from __future__ import annotations

import pandas as pd


def summarise_native_validation(validation_csv: str) -> dict:
    """Summarise a completed native-speaker validation sheet.

    Expects `auto_annotation_correct` to contain True/False, yes/no, 1/0, or blank.
    """
    df = pd.read_csv(validation_csv)
    if "auto_annotation_correct" not in df.columns:
        raise ValueError("Missing column: auto_annotation_correct")

    cleaned = df["auto_annotation_correct"].astype(str).str.lower().str.strip()
    valid = cleaned[cleaned.isin(["true", "false", "yes", "no", "1", "0"])]
    correct = valid.isin(["true", "yes", "1"])

    return {
        "n_reviewed": int(len(valid)),
        "n_correct": int(correct.sum()),
        "accuracy": float(correct.mean()) if len(valid) else None,
        "n_unreviewed": int(len(df) - len(valid)),
    }
