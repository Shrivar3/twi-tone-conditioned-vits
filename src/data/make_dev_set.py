from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from datasets import Audio, load_dataset

from src.utils.paths import ensure_parent


def _as_float(value: Any):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _disable_audio_decoding(ds):
    """Prevent Hugging Face Datasets from decoding audio via torchcodec.

    In Codespaces, we only need text/duration metadata for the dev manifest.
    Decoding audio can trigger torchcodec/FFmpeg errors and uses much more disk.
    """
    try:
        features = getattr(ds, "features", None)
        if features is not None and "audio" in features:
            ds = ds.cast_column("audio", Audio(decode=False))
    except Exception as exc:
        print(f"Warning: could not disable audio decoding: {exc}")

    return ds


def make_dev_rows(config: dict) -> list[dict]:
    dataset_cfg = config["dataset"]
    dev_cfg = config["dev_set"]

    hf_dataset_id = dataset_cfg["hf_dataset_id"]
    split = dev_cfg.get("split", dataset_cfg.get("eval_split", "test"))

    text_col = dataset_cfg.get("text_column", "text")
    duration_col = dataset_cfg.get("duration_column", "duration")

    n_wer = int(dev_cfg.get("n_wer_samples", 100))
    n_mos = int(dev_cfg.get("n_mos_samples", 50))
    n_tone = int(dev_cfg.get("n_tone_validation_samples", 100))

    min_duration = float(dev_cfg.get("min_duration_seconds", 0.0))
    max_duration = float(dev_cfg.get("max_duration_seconds", 999999.0))

    max_stream_examples = int(dev_cfg.get("max_stream_examples", max(1000, n_wer * 10)))

    print(f"Loading {hf_dataset_id!r}, split={split!r}, streaming=True")
    ds = load_dataset(hf_dataset_id, split=split, streaming=True)
    ds = _disable_audio_decoding(ds)

    rows: list[dict] = []

    for stream_index, row in enumerate(ds):
        if stream_index >= max_stream_examples:
            break

        text = str(row.get(text_col, "")).strip()
        if not text:
            continue

        duration = _as_float(row.get(duration_col))
        if duration is not None:
            if duration < min_duration or duration > max_duration:
                continue

        rows.append(
            {
                "utt_id": f"dev_{len(rows):04d}",
                "split": split,
                "hf_index": stream_index,
                "text": text,
                "duration": "" if duration is None else duration,
                "wer_subset": True,
                "mos_subset": len(rows) < n_mos,
                "tone_validation_subset": len(rows) < n_tone,
            }
        )

        if len(rows) >= n_wer:
            break

    if not rows:
        raise RuntimeError(
            "No dev-set rows were collected. Check split name, text column, "
            "duration column, and duration filters."
        )

    if len(rows) < n_wer:
        print(
            f"Warning: requested {n_wer} rows but only collected {len(rows)} "
            f"within max_stream_examples={max_stream_examples}."
        )

    return rows


def save_dev_set(config: dict, output_csv: str | Path | None = None) -> Path:
    out_path = ensure_parent(output_csv or config["dev_set"]["output_csv"])
    rows = make_dev_rows(config)

    fieldnames = [
        "utt_id",
        "split",
        "hf_index",
        "text",
        "duration",
        "wer_subset",
        "mos_subset",
        "tone_validation_subset",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {out_path}")
    return out_path
