from __future__ import annotations

import pandas as pd
from jiwer import cer, wer

from src.data.text_normalise import load_normalisation_config, normalise_text
from src.utils.paths import ensure_parent, resolve_path


def compute_roundtrip_wer(
    transcripts_csv: str,
    output_csv: str,
    normalisation_config_path: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(transcripts_csv)
    if "text" not in df.columns:
        raise ValueError("Expected reference column `text` in transcripts CSV")
    if "asr_transcript" not in df.columns:
        raise ValueError("Expected hypothesis column `asr_transcript` in transcripts CSV")

    cfg = load_normalisation_config(resolve_path(normalisation_config_path)) if normalisation_config_path else None
    df["reference_norm"] = df["text"].fillna("").map(lambda x: normalise_text(x, cfg))
    df["asr_norm"] = df["asr_transcript"].fillna("").map(lambda x: normalise_text(x, cfg))

    df["wer"] = [wer(r, h) for r, h in zip(df["reference_norm"], df["asr_norm"])]
    df["cer"] = [cer(r, h) for r, h in zip(df["reference_norm"], df["asr_norm"])]

    out_path = ensure_parent(output_csv)
    df.to_csv(out_path, index=False)
    return df


def summarise_wer(df: pd.DataFrame) -> dict:
    return {
        "n_samples": int(len(df)),
        "mean_wer": float(df["wer"].mean()),
        "median_wer": float(df["wer"].median()),
        "mean_cer": float(df["cer"].mean()),
        "median_cer": float(df["cer"].median()),
    }
