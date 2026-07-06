from __future__ import annotations

import argparse
import json
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

from src.modelling.tone_conditioned_vits import ToneConditionedVitsModel
from src.training.tone_vits_collator import ToneVitsBatchCollator
from src.training.tone_vits_dataset import ToneVitsManifestDataset
from src.training.tone_vits_losses import masked_waveform_mse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Debug/proxy training scaffold for tone-conditioned VITS. "
            "This does not yet implement the full VITS loss."
        )
    )

    parser.add_argument(
        "--model-id",
        default="FarmerlineML/main_twi_TTS",
        help="HF model id or local checkpoint path.",
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/dev_set_tone_conditioning_gpu_check.csv",
        help="Tone-conditioning manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/tone_vits_debug",
        help="Local checkpoint/debug output directory. This should remain gitignored.",
    )

    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tone-sequence-column", default="gemini_tone_sequence")
    parser.add_argument("--input-ids-column", default="input_ids")
    parser.add_argument("--attention-mask-column", default="attention_mask")
    parser.add_argument("--tone-ids-column", default="tone_ids")
    parser.add_argument("--audio-column", default=None)
    parser.add_argument("--sample-id-column", default=None)
    parser.add_argument("--audio-root", default=None)

    parser.add_argument("--target-sampling-rate", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--overfit-n-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--dry-run-batch",
        action="store_true",
        help="Load one batch, run inference once, print shapes, then exit.",
    )
    parser.add_argument(
        "--allow-missing-audio",
        action="store_true",
        help=(
            "Allow synthetic zero audio if audio paths are unavailable. "
            "Use this for dry-run only."
        ),
    )
    parser.add_argument(
        "--allow-synthetic-audio-targets",
        action="store_true",
        help=(
            "Permit proxy training even when some audio is synthetic zeros. "
            "This is only for debugging and should not be used for real results."
        ),
    )
    parser.add_argument("--synthetic-audio-seconds", type=float, default=1.0)

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--tone-embedding-scale",
        type=float,
        default=1.0,
        help=(
            "Use 1.0 for trainability. Using 0.0 intentionally disables the "
            "tone embedding contribution."
        ),
    )
    parser.add_argument(
        "--freeze-base",
        action="store_true",
        help="Freeze pretrained VITS weights before selecting trainable tone params.",
    )
    parser.add_argument(
        "--train-tone-adapter-only",
        action="store_true",
        help="Train only the tone embedding table in the current wrapper.",
    )
    parser.add_argument(
        "--deterministic-inference",
        action="store_true",
        help="Set VITS inference noise scales to zero where exposed.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save adapter checkpoint every N steps. 0 disables intermediate saves.",
    )

    return parser.parse_args()


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _set_deterministic_inference(model: ToneConditionedVitsModel) -> None:
    # These attributes exist on HF VITS inference models. Guard with hasattr so
    # this script stays robust across minor Transformers changes.
    for attr in ["noise_scale", "noise_scale_duration"]:
        if hasattr(model.vits, attr):
            setattr(model.vits, attr, 0.0)


def _configure_trainable_params(
    model: ToneConditionedVitsModel,
    *,
    freeze_base: bool,
    train_tone_adapter_only: bool,
) -> list[torch.nn.Parameter]:
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    if train_tone_adapter_only:
        if not hasattr(model.vits.text_encoder, "tone_embedding"):
            raise AttributeError(
                "Expected model.vits.text_encoder.tone_embedding to exist."
            )

        for param in model.parameters():
            param.requires_grad = False

        model.vits.text_encoder.tone_embedding.weight.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]

    if not params:
        raise RuntimeError(
            "No trainable parameters selected. Use --train-tone-adapter-only "
            "or avoid freezing all parameters."
        )

    return params


def _save_debug_checkpoint(
    model: ToneConditionedVitsModel,
    output_dir: Path,
    *,
    step: int,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    text_encoder = model.vits.text_encoder
    payload = {
        "step": int(step),
        "model_id": args.model_id,
        "tone_embedding_scale": float(args.tone_embedding_scale),
        "tone_embedding_state_dict": (
            text_encoder.tone_embedding.state_dict()
            if hasattr(text_encoder, "tone_embedding")
            else None
        ),
        "args": vars(args),
    }

    torch.save(payload, output_dir / f"tone_adapter_step_{step:06d}.pt")
    model.save_tone_metadata(output_dir)


def _print_batch_summary(batch: dict[str, Any]) -> None:
    print("Batch summary")
    print("-------------")
    for key in [
        "input_ids",
        "attention_mask",
        "tone_ids",
        "audio_values",
        "audio_attention_mask",
        "audio_lengths",
        "has_real_audio",
    ]:
        value = batch[key]
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"{key}: {value}")

    print(f"sample_ids: {batch['sample_ids']}")
    print(f"audio_paths: {batch['audio_paths']}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    max_rows = args.overfit_n_samples if args.overfit_n_samples else args.max_rows

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    dataset = ToneVitsManifestDataset(
        args.manifest,
        tokenizer=tokenizer,
        text_column=args.text_column,
        tone_sequence_column=args.tone_sequence_column,
        input_ids_column=args.input_ids_column,
        attention_mask_column=args.attention_mask_column,
        tone_ids_column=args.tone_ids_column,
        audio_column=args.audio_column,
        sample_id_column=args.sample_id_column,
        audio_root=args.audio_root,
        target_sampling_rate=args.target_sampling_rate,
        max_rows=max_rows,
        allow_missing_audio=args.allow_missing_audio or args.dry_run_batch,
        synthetic_audio_seconds=args.synthetic_audio_seconds,
    )

    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(dataset.summary(), f, indent=2, ensure_ascii=False)

    print("Dataset summary")
    print("---------------")
    print(json.dumps(dataset.summary(), indent=2, ensure_ascii=False))

    collator = ToneVitsBatchCollator(input_pad_id=0)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.dry_run_batch,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    model = ToneConditionedVitsModel.from_pretrained(
        args.model_id,
        tone_embedding_scale=args.tone_embedding_scale,
    )
    model.to(device)

    if args.deterministic_inference:
        _set_deterministic_inference(model)

    first_batch = next(iter(loader))
    _print_batch_summary(first_batch)

    first_batch = _move_batch_to_device(first_batch, device)

    if args.dry_run_batch:
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=first_batch["input_ids"],
                attention_mask=first_batch["attention_mask"],
                tone_ids=first_batch["tone_ids"],
            )

        print("Dry-run model output")
        print("--------------------")
        print(f"waveform shape: {tuple(outputs.waveform.shape)}")
        print(f"sequence_lengths: {outputs.sequence_lengths.detach().cpu().tolist()}")
        print(f"spectrogram shape: {tuple(outputs.spectrogram.shape)}")
        print("Dry-run completed successfully.")
        return

    if args.max_steps <= 0:
        raise ValueError(
            "Nothing to do: pass --dry-run-batch or set --max-steps > 0."
        )

    params = _configure_trainable_params(
        model,
        freeze_base=args.freeze_base,
        train_tone_adapter_only=args.train_tone_adapter_only,
    )

    n_trainable = sum(p.numel() for p in params)
    print(f"Trainable parameters: {n_trainable:,}")

    optimiser = torch.optim.AdamW(params, lr=args.lr)

    model.train()
    running = cycle(loader)

    progress = tqdm(range(1, args.max_steps + 1), desc="proxy-train")

    for step in progress:
        batch = next(running)
        batch = _move_batch_to_device(batch, device)

        if (
            not args.allow_synthetic_audio_targets
            and not torch.all(batch["has_real_audio"]).item()
        ):
            raise RuntimeError(
                "At least one batch item has synthetic/missing audio. "
                "For dry-run use --dry-run-batch. For real proxy training, provide "
                "valid audio paths. For debugging only, pass "
                "--allow-synthetic-audio-targets."
            )

        optimiser.zero_grad(set_to_none=True)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            tone_ids=batch["tone_ids"],
        )

        loss = masked_waveform_mse(
            outputs.waveform,
            batch["audio_values"],
            batch["audio_attention_mask"],
        )

        loss.backward()

        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)

        optimiser.step()

        progress.set_postfix({"loss": f"{loss.item():.6f}"})

        if args.save_every and step % args.save_every == 0:
            _save_debug_checkpoint(model, output_dir, step=step, args=args)

    _save_debug_checkpoint(model, output_dir, step=args.max_steps, args=args)

    with (output_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "max_steps": args.max_steps,
                "final_proxy_loss": float(loss.item()),
                "trainable_parameters": int(n_trainable),
                "warning": (
                    "This is a proxy waveform-MSE debug loop, not the full "
                    "VITS training objective."
                ),
            },
            f,
            indent=2,
        )

    print("Training scaffold run completed.")
    print(f"Saved debug adapter files under: {output_dir}")


if __name__ == "__main__":
    main()
