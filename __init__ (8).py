from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.load_hf_dataset import load_twi_dataset, split_to_dataframe
from src.utils.paths import ensure_parent


def make_dev_set(config: dict) -> pd.DataFrame:
    dataset_cfg = config["dataset"]
    dev_cfg = config["dev_set"]

    ds = load_twi_dataset(
        dataset_cfg["hf_dataset_id"],
        target_sampling_rate=None,  # manifest creation does not need audio decoding
    )

    split = dev_cfg.get("split", dataset_cfg.get("eval_split", "test"))
    df = split_to_dataframe(ds, split).reset_index().rename(columns={"index": "hf_index"})

    text_col = dataset_cfg.get("text_column", "text")
    duration_col = dataset_cfg.get("duration_column", "duration")

    df = df[df[text_col].notna()]
    df = df[df[text_col].astype(str).str.strip().str.len() > 0]

    if duration_col in df.columns:
        df = df[df[duration_col].between(
            dev_cfg.get("min_duration_seconds", 0.0),
            dev_cfg.get("max_duration_seconds", float("inf")),
            inclusive="both",
        )]
        n_bins = int(dev_cfg.get("duration_bins", 5))
        df["duration_bin"] = pd.qcut(df[duration_col], q=n_bins, duplicates="drop")
        per_bin = max(1, int(dev_cfg["n_wer_samples"]) // max(1, df["duration_bin"].nunique()))
        sampled = (
            df.groupby("duration_bin", group_keys=False)
            .sample(n=min(per_bin, df.groupby("duration_bin").size().min()), random_state=config["project"].get("seed", 42))
        )
        # If rounding undershoots, top up randomly.
        if len(sampled) < int(dev_cfg["n_wer_samples"]):
            remainder = df.drop(sampled.index, errors="ignore")
            top_up = remainder.sample(
                n=min(int(dev_cfg["n_wer_samples"]) - len(sampled), len(remainder)),
                random_state=config["project"].get("seed", 42),
            )
            sampled = pd.concat([sampled, top_up], ignore_index=False)
    else:
        sampled = df.sample(
            n=min(int(dev_cfg["n_wer_samples"]), len(df)),
            random_state=config["project"].get("seed", 42),
        )

    dev = sampled.sample(frac=1, random_state=config["project"].get("seed", 42)).reset_index(drop=True)
    dev = dev.head(int(dev_cfg["n_wer_samples"]))

    dev["utt_id"] = [f"dev_{i:04d}" for i in range(len(dev))]
    dev["split"] = split
    dev["wer_subset"] = True
    dev["mos_subset"] = False
    dev.loc[: int(dev_cfg["n_mos_samples"]) - 1, "mos_subset"] = True
    dev["tone_validation_subset"] = False
    dev.loc[: int(dev_cfg["n_tone_validation_samples"]) - 1, "tone_validation_subset"] = True

    keep_cols = ["utt_id", "split", "hf_index", text_col]
    if duration_col in dev.columns:
        keep_cols.append(duration_col)
    keep_cols += ["wer_subset", "mos_subset", "tone_validation_subset"]

    out = dev[keep_cols].copy()
    if text_col != "text":
        out = out.rename(columns={text_col: "text"})
    if duration_col != "duration" and duration_col in out.columns:
        out = out.rename(columns={duration_col: "duration"})
    return out


def save_dev_set(config: dict, output_csv: str | Path | None = None) -> Path:
    dev = make_dev_set(config)
    out_path = ensure_parent(output_csv or config["dev_set"]["output_csv"])
    dev.to_csv(out_path, index=False)
    return out_path
