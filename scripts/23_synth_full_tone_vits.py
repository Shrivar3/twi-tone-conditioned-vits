from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.tone.token_tone_alignment import build_token_tone_encoding
from src.tone.tone_vocab import ToneVocab
from src.training.full_vits.config import (
    audio_config_from_dict,
    model_config_from_dict,
)
from src.training.full_vits.models import ToneConditionedVitsGenerator
from src.utils.paths import ensure_dir, ensure_parent, load_yaml, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize audio from a full tone-conditioned VITS checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint saved by scripts/22_train_full_tone_vits.py.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="CSV manifest with utt_id, text, and gemini_tone_sequence columns.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for generated WAV files.",
    )
    parser.add_argument(
        "--output-manifest",
        required=True,
        help="Output CSV with full_tone_vits_audio_path added.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional training config YAML or resolved_config.json override.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional tokenizer model id override.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional row limit for quick synthesis checks.",
    )
    parser.add_argument(
        "--disable-tone-conditioning",
        action="store_true",
        help="Disable tone conditioning for no-tone ablation synthesis.",
    )
    return parser.parse_args()


def _load_config_file(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    if config_path.suffix.lower() == ".json":
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return load_yaml(config_path)


def _checkpoint_path(path: str | Path) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return resolve_path(raw_path)


def _config_from_checkpoint(
    checkpoint_path: Path,
    payload: dict[str, Any],
    *,
    config_override: str | None = None,
) -> dict[str, Any]:
    if config_override is not None:
        return _load_config_file(config_override)

    if isinstance(payload.get("config"), dict):
        return payload["config"]

    resolved_config = checkpoint_path.parent / "resolved_config.json"
    if resolved_config.exists():
        return _load_config_file(resolved_config)

    raise ValueError(
        "Could not find training config in checkpoint payload or "
        f"{resolved_config}. Pass --config explicitly."
    )


def _tokenizer_vocab_size(tokenizer: Any) -> int:
    try:
        return int(len(tokenizer))
    except TypeError:
        if hasattr(tokenizer, "get_vocab"):
            return len(tokenizer.get_vocab())
        raise


def _model_id(config: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    if config.get("model_id"):
        return str(config["model_id"])
    return "FarmerlineML/main_twi_TTS"


def _build_generator(
    checkpoint_path: Path,
    *,
    payload: dict[str, Any],
    config: dict[str, Any],
    tokenizer: Any,
    device: torch.device,
    disable_tone_conditioning: bool,
) -> tuple[ToneConditionedVitsGenerator, int]:
    audio_cfg_source = config.get("resolved_audio_config") or config.get("audio")
    model_cfg_source = config.get("resolved_model_config") or config.get("model")
    audio_config = audio_config_from_dict(audio_cfg_source)
    model_config = model_config_from_dict(model_cfg_source)

    model_config.n_vocab = int(
        (model_cfg_source or {}).get("n_vocab") or _tokenizer_vocab_size(tokenizer)
    )
    model_config.num_tones = int(
        (model_cfg_source or {}).get("num_tones") or len(ToneVocab().labels)
    )
    model_config.spec_channels = int(
        (model_cfg_source or {}).get("spec_channels")
        or audio_config.filter_length // 2 + 1
    )

    if disable_tone_conditioning:
        model_config.use_tone_conditioning = False
        model_config.tone_conditioning_mode = "none"

    generator = ToneConditionedVitsGenerator(model_config).to(device)
    state_dict = payload.get("generator")
    if state_dict is None:
        raise ValueError(f"Checkpoint does not contain generator weights: {checkpoint_path}")

    load_result = generator.load_state_dict(
        state_dict,
        strict=not disable_tone_conditioning,
    )
    if disable_tone_conditioning:
        if load_result.missing_keys:
            print(f"Missing generator keys after ablation load: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"Unexpected generator keys after ablation load: {load_result.unexpected_keys}")

    generator.eval()
    return generator, int(audio_config.sampling_rate)


def _validate_manifest(df: pd.DataFrame, manifest: str) -> None:
    required = ["utt_id", "text", "gemini_tone_sequence"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Manifest {manifest} is missing columns: {missing}")


def _tone_sequence(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _build_inputs(
    *,
    text: str,
    tone_sequence: str | None,
    tokenizer: Any,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoding = build_token_tone_encoding(
        text=text,
        tokenizer=tokenizer,
        tone_sequence=tone_sequence,
    )
    return {
        "text_ids": torch.tensor([encoding.input_ids], dtype=torch.long, device=device),
        "text_lengths": torch.tensor([len(encoding.input_ids)], dtype=torch.long, device=device),
        "tone_ids": torch.tensor([encoding.tone_ids], dtype=torch.long, device=device),
    }


def synthesize_manifest(
    *,
    checkpoint: str,
    manifest: str,
    output_dir: str,
    output_manifest: str,
    config: str | None = None,
    model_id: str | None = None,
    device: str | torch.device | None = None,
    max_samples: int | None = None,
    disable_tone_conditioning: bool = False,
) -> pd.DataFrame:
    checkpoint_path = _checkpoint_path(checkpoint)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    payload = torch.load(checkpoint_path, map_location=torch_device)
    training_config = _config_from_checkpoint(
        checkpoint_path,
        payload,
        config_override=config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        _model_id(training_config, model_id),
        trust_remote_code=True,
    )
    generator, sampling_rate = _build_generator(
        checkpoint_path,
        payload=payload,
        config=training_config,
        tokenizer=tokenizer,
        device=torch_device,
        disable_tone_conditioning=disable_tone_conditioning,
    )

    df = pd.read_csv(resolve_path(manifest))
    _validate_manifest(df, manifest)
    if max_samples is not None:
        df = df.head(int(max_samples)).copy()

    out_dir = ensure_dir(output_dir)
    rows: list[dict[str, Any]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Synthesizing full-tone VITS"):
        utt_id = str(row["utt_id"])
        text = str(row["text"])
        tone_sequence = _tone_sequence(row["gemini_tone_sequence"])
        inputs = _build_inputs(
            text=text,
            tone_sequence=tone_sequence,
            tokenizer=tokenizer,
            device=torch_device,
        )

        with torch.no_grad():
            outputs = generator.infer(**inputs)

        wav = outputs["waveform"].squeeze().detach().cpu().numpy().astype(np.float32)
        audio_path = out_dir / f"{utt_id}.wav"
        sf.write(audio_path, wav, sampling_rate)

        out = row.to_dict()
        out["full_tone_vits_audio_path"] = str(audio_path)
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(ensure_parent(output_manifest), index=False)
    return out_df


def main() -> None:
    args = parse_args()
    out_df = synthesize_manifest(
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        output_dir=args.output_dir,
        output_manifest=args.output_manifest,
        config=args.config,
        model_id=args.model_id,
        device=args.device,
        max_samples=args.max_samples,
        disable_tone_conditioning=args.disable_tone_conditioning,
    )
    print(f"Wrote {len(out_df)} synthesized rows to {ensure_parent(args.output_manifest)}")


if __name__ == "__main__":
    main()
