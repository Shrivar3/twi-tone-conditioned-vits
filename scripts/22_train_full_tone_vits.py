from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

from src.tone.tone_vocab import ToneVocab
from src.training.full_vits.checkpoints import load_checkpoint, save_checkpoint
from src.training.full_vits.config import (
    audio_config_from_dict,
    model_config_from_dict,
    training_config_from_dict,
)
from src.training.full_vits.data import FullVitsBatchCollator, FullVitsFilelistDataset
from src.training.full_vits.models import MultiPeriodDiscriminator, ToneConditionedVitsGenerator
from src.training.full_vits.train_loop import (
    move_batch_to_device,
    training_step,
    write_validation_samples,
)
from src.utils.paths import ensure_dir, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train full tone-conditioned VITS.")
    parser.add_argument("--config", default="configs/full_vits_tone_debug.yaml")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--disable-tone-conditioning",
        action="store_true",
        help="Run a no-tone ablation while keeping the same full VITS objective.",
    )
    return parser.parse_args()


def _tokenizer_pad_id(tokenizer: Any) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    return int(pad_id) if pad_id is not None else 0


def _tokenizer_vocab_size(tokenizer: Any) -> int:
    try:
        return int(len(tokenizer))
    except TypeError:
        if hasattr(tokenizer, "get_vocab"):
            return len(tokenizer.get_vocab())
        raise


def _make_loader(
    filelist: str,
    *,
    tokenizer: Any,
    audio_config,
    data_cfg: dict[str, Any],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = FullVitsFilelistDataset(
        filelist,
        tokenizer=tokenizer,
        audio_config=audio_config,
        audio_root=data_cfg.get("audio_root"),
        max_rows=data_cfg.get("max_rows"),
        allow_missing_audio=bool(data_cfg.get("allow_missing_audio", False)),
        synthetic_audio_seconds=float(data_cfg.get("synthetic_audio_seconds", 1.0)),
    )
    collator = FullVitsBatchCollator(text_pad_id=_tokenizer_pad_id(tokenizer))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )


def main() -> None:
    args = parse_args()
    raw_cfg = load_yaml(args.config)
    set_seed(int(raw_cfg.get("project", {}).get("seed", 42)))

    data_cfg = raw_cfg["data"]
    audio_config = audio_config_from_dict(raw_cfg.get("audio"))
    model_config = model_config_from_dict(raw_cfg.get("model"))
    train_config = training_config_from_dict(raw_cfg.get("training"))

    if args.max_steps is not None:
        train_config.max_steps = int(args.max_steps)
    if args.disable_tone_conditioning:
        model_config.use_tone_conditioning = False
        model_config.tone_conditioning_mode = "none"

    output_dir = ensure_dir(args.output_dir or raw_cfg.get("output_dir", "checkpoints/full_vits_tone_debug"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    tokenizer = AutoTokenizer.from_pretrained(
        raw_cfg.get("model_id", "FarmerlineML/main_twi_TTS"),
        trust_remote_code=True,
    )
    model_config.n_vocab = int(raw_cfg.get("model", {}).get("n_vocab") or _tokenizer_vocab_size(tokenizer))
    model_config.num_tones = len(ToneVocab().labels)
    model_config.spec_channels = audio_config.filter_length // 2 + 1
    if model_config.upsample_factor != audio_config.hop_length:
        raise ValueError(
            "VITS decoder upsample factor must match the spectrogram hop length: "
            f"upsample_factor={model_config.upsample_factor}, "
            f"hop_length={audio_config.hop_length}."
        )

    train_loader = _make_loader(
        data_cfg["train_filelist"],
        tokenizer=tokenizer,
        audio_config=audio_config,
        data_cfg=data_cfg,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        shuffle=True,
    )
    val_loader = None
    if data_cfg.get("val_filelist"):
        val_loader = _make_loader(
            data_cfg["val_filelist"],
            tokenizer=tokenizer,
            audio_config=audio_config,
            data_cfg=data_cfg,
            batch_size=min(train_config.batch_size, 2),
            num_workers=0,
            shuffle=False,
        )

    generator = ToneConditionedVitsGenerator(model_config).to(device)
    discriminator = MultiPeriodDiscriminator.from_model_config(model_config).to(device)

    optimizer_g = torch.optim.AdamW(
        generator.parameters(),
        lr=train_config.learning_rate,
        betas=train_config.betas,
        eps=train_config.eps,
    )
    optimizer_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=train_config.learning_rate,
        betas=train_config.betas,
        eps=train_config.eps,
    )

    start_step = 0
    start_epoch = 0
    if args.resume_from_checkpoint:
        payload = load_checkpoint(
            args.resume_from_checkpoint,
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            map_location=device,
        )
        start_step = int(payload.get("step", 0))
        start_epoch = int(payload.get("epoch", 0))
        print(f"Resumed from {args.resume_from_checkpoint} at step {start_step}.")

    config_snapshot = {
        **raw_cfg,
        "resolved_audio_config": asdict(audio_config),
        "resolved_model_config": asdict(model_config),
        "resolved_training_config": asdict(train_config),
    }
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)

    running = cycle(train_loader)
    progress = tqdm(
        range(start_step + 1, train_config.max_steps + 1),
        desc="full-tone-vits",
    )

    epoch = start_epoch
    for step in progress:
        batch = move_batch_to_device(next(running), device)
        metrics = training_step(
            batch,
            generator=generator,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            audio_config=audio_config,
            loss_weights=train_config.loss_weights,
            grad_clip=train_config.grad_clip,
        )

        if step % train_config.log_every == 0 or step == 1:
            progress.set_postfix(
                {
                    "g": f"{metrics['loss_g']:.3f}",
                    "d": f"{metrics['loss_d']:.3f}",
                    "mel": f"{metrics['loss_mel']:.3f}",
                    "dur": f"{metrics['loss_duration']:.3f}",
                }
            )

        if val_loader is not None and train_config.validate_every and step % train_config.validate_every == 0:
            val_batch = move_batch_to_device(next(iter(val_loader)), device)
            written = write_validation_samples(
                generator,
                val_batch,
                output_dir=output_dir / "validation_samples",
                step=step,
                sampling_rate=audio_config.sampling_rate,
            )
            print(f"Wrote {len(written)} validation samples for step {step}.")

        if train_config.save_every and step % train_config.save_every == 0:
            ckpt_path = output_dir / f"full_tone_vits_step_{step:08d}.pt"
            save_checkpoint(
                ckpt_path,
                step=step,
                epoch=epoch,
                generator=generator,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                config=config_snapshot,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    final_path = output_dir / "full_tone_vits_final.pt"
    save_checkpoint(
        final_path,
        step=train_config.max_steps,
        epoch=epoch,
        generator=generator,
        discriminator=discriminator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        config=config_snapshot,
    )
    print(f"Training complete. Saved final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
