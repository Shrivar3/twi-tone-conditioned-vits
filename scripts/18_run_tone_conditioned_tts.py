from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.modelling.tone_conditioned_vits import ToneConditionedVitsModel
from src.tone.token_tone_alignment import build_tone_conditioned_inputs
from src.utils.paths import ensure_dir, ensure_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tone-conditioned VITS TTS on a CSV manifest.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="CSV manifest with utt_id, text, and gemini_tone_sequence or tone_ids_json.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for generated WAV files.",
    )
    parser.add_argument(
        "--output-manifest",
        required=True,
        help="CSV path for the manifest with generated audio paths.",
    )
    parser.add_argument(
        "--model-id",
        default="FarmerlineML/main_twi_TTS",
        help="HF model id or local model path.",
    )
    parser.add_argument(
        "--adapter-checkpoint",
        default=None,
        help="Optional .pt checkpoint containing tone_embedding_state_dict.",
    )
    parser.add_argument(
        "--tone-embedding-scale",
        type=float,
        default=1.0,
        help="Scale applied to the tone embedding contribution.",
    )
    parser.add_argument(
        "--system-name",
        default="tone_vits",
        help="Prefix for the generated audio path column.",
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
    return parser.parse_args()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _optional_text(value: Any) -> str | None:
    if _is_missing(value):
        return None

    text = str(value).strip()
    return text or None


def _parse_tone_ids_json(value: Any, *, utt_id: str) -> list[int] | None:
    if _is_missing(value):
        return None

    if isinstance(value, list):
        return [int(x) for x in value]

    if isinstance(value, np.ndarray):
        return [int(x) for x in value.tolist()]

    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)

    if not isinstance(parsed, list):
        raise ValueError(
            f"tone_ids_json for utt_id={utt_id} must parse to a list, got {type(parsed)}."
        )

    return [int(x) for x in parsed]


def _load_tone_adapter(
    model: ToneConditionedVitsModel,
    checkpoint_path: str,
    *,
    device: torch.device,
) -> None:
    payload = torch.load(checkpoint_path, map_location=device)
    tone_embedding_state_dict = payload["tone_embedding_state_dict"]

    if tone_embedding_state_dict is None:
        raise ValueError(
            f"Checkpoint does not contain tone embedding weights: {checkpoint_path}"
        )

    model.vits.text_encoder.tone_embedding.load_state_dict(tone_embedding_state_dict)


def _validate_manifest(df: pd.DataFrame, manifest_csv: str) -> None:
    missing = [column for column in ["utt_id", "text"] if column not in df.columns]
    if missing:
        raise ValueError(f"Manifest {manifest_csv} is missing columns: {missing}")

    if "gemini_tone_sequence" not in df.columns and "tone_ids_json" not in df.columns:
        raise ValueError(
            f"Manifest {manifest_csv} must contain gemini_tone_sequence or tone_ids_json."
        )


def synthesise_manifest(
    manifest_csv: str,
    output_dir: str,
    output_manifest: str,
    *,
    model_id: str,
    adapter_checkpoint: str | None = None,
    tone_embedding_scale: float = 1.0,
    system_name: str = "tone_vits",
    device: str | torch.device | None = None,
    max_samples: int | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    _validate_manifest(df, manifest_csv)

    if max_samples is not None:
        df = df.head(int(max_samples)).copy()

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = ToneConditionedVitsModel.from_pretrained(
        model_id,
        tone_embedding_scale=tone_embedding_scale,
        trust_remote_code=True,
    )
    model.to(torch_device)

    if adapter_checkpoint is not None:
        _load_tone_adapter(model, adapter_checkpoint, device=torch_device)

    model.eval()

    out_dir = ensure_dir(output_dir)
    audio_column = f"{system_name}_audio_path"
    sampling_rate = int(getattr(model.config, "sampling_rate", 22050))

    rows: list[dict[str, Any]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Synthesising {system_name}"):
        utt_id = str(row["utt_id"])
        text = str(row["text"])
        tone_sequence = _optional_text(row.get("gemini_tone_sequence"))

        inputs = build_tone_conditioned_inputs(
            text=text,
            tokenizer=tokenizer,
            tone_sequence=tone_sequence,
            device=torch_device,
        )

        tone_ids = _parse_tone_ids_json(row.get("tone_ids_json"), utt_id=utt_id)
        if tone_ids is not None:
            input_len = int(inputs["input_ids"].shape[-1])
            if len(tone_ids) != input_len:
                raise ValueError(
                    f"tone_ids_json length {len(tone_ids)} does not match input_ids "
                    f"length {input_len} for utt_id={utt_id}."
                )
            inputs["tone_ids"] = torch.tensor(
                [tone_ids],
                dtype=torch.long,
                device=torch_device,
            )

        with torch.no_grad():
            outputs = model(**inputs)

        wav = outputs.waveform[0].detach().cpu().numpy().astype(np.float32)
        audio_path = out_dir / f"{utt_id}.wav"
        sf.write(audio_path, wav, sampling_rate)

        out = row.to_dict()
        out[audio_column] = str(audio_path)
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_path = ensure_parent(output_manifest)
    out_df.to_csv(out_path, index=False)

    return out_df


def main() -> None:
    args = parse_args()

    out_df = synthesise_manifest(
        manifest_csv=args.manifest,
        output_dir=args.output_dir,
        output_manifest=args.output_manifest,
        model_id=args.model_id,
        adapter_checkpoint=args.adapter_checkpoint,
        tone_embedding_scale=args.tone_embedding_scale,
        system_name=args.system_name,
        device=args.device,
        max_samples=args.max_samples,
    )

    print(f"Wrote {len(out_df)} rows to {Path(args.output_manifest)}")
    print(f"Audio column: {args.system_name}_audio_path")


if __name__ == "__main__":
    main()
