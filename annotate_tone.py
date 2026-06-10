from __future__ import annotations

import pandas as pd
from tqdm import tqdm
from transformers import pipeline

from src.utils.paths import ensure_parent


def transcribe_audio_manifest(
    manifest_with_audio_csv: str,
    output_csv: str,
    asr_model_id: str = "openai/whisper-small",
    language: str | None = "twi",
    task: str = "transcribe",
) -> pd.DataFrame:
    df = pd.read_csv(manifest_with_audio_csv)
    if "baseline_tts_audio_path" not in df.columns:
        raise ValueError("Manifest must contain baseline_tts_audio_path")

    asr = pipeline(
        "automatic-speech-recognition",
        model=asr_model_id,
        device=0,  # use GPU if available; change to -1 for CPU
    )

    rows = []
    generate_kwargs = {}
    if language:
        generate_kwargs["language"] = language
    if task:
        generate_kwargs["task"] = task

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Running ASR"):
        audio_path = row["baseline_tts_audio_path"]
        try:
            result = asr(audio_path, generate_kwargs=generate_kwargs)
            transcript = result.get("text", "") if isinstance(result, dict) else str(result)
            error = ""
        except Exception as exc:  # keep going and inspect failures later
            transcript = ""
            error = repr(exc)
        out = row.to_dict()
        out["asr_transcript"] = transcript
        out["asr_error"] = error
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_path = ensure_parent(output_csv)
    out_df.to_csv(out_path, index=False)
    return out_df
