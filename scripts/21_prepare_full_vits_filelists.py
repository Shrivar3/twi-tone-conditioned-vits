from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.paths import ensure_dir, ensure_parent, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare full-VITS train/val/test filelists from a Farmerline manifest.",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="CSV manifest containing audio_path, text, and gemini_tone_sequence.",
    )
    parser.add_argument("--output-dir", default="data/filelists/full_vits")
    parser.add_argument("--train-output", default=None)
    parser.add_argument("--val-output", default=None)
    parser.add_argument("--test-output", default=None)
    parser.add_argument("--audio-column", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tone-column", default="gemini_tone_sequence")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def _first_existing_column(
    columns: list[str],
    requested: str | None,
    candidates: list[str],
) -> str | None:
    if requested and requested in columns:
        return requested
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _clean_filelist_field(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    return " ".join(text.replace("|", " ").split())


def _with_audio_root(path: Any, audio_root: str | None) -> str:
    value = _clean_filelist_field(path)
    if not audio_root:
        return value
    candidate = Path(value)
    if candidate.is_absolute():
        return value
    return str(Path(audio_root) / candidate)


def _split_manifest(
    df: pd.DataFrame,
    *,
    split_column: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    if split_column in df.columns:
        split_values = df[split_column].astype(str).str.lower()
        train = df[split_values.isin(["train", "training"])].copy()
        val = df[split_values.isin(["val", "valid", "validation", "dev"])].copy()
        test = df[split_values.isin(["test", "eval", "evaluation"])].copy()
        if not train.empty:
            rng = random.Random(seed)
            if val.empty and len(train) >= 3:
                val_count = max(1, int(round(len(train) * val_fraction)))
                val_indices = rng.sample(list(train.index), k=min(val_count, len(train) - 1))
                val = train.loc[val_indices].copy()
                train = train.drop(index=val_indices)
            if test.empty and len(train) >= 3:
                test_count = max(1, int(round(len(train) * test_fraction)))
                test_indices = rng.sample(list(train.index), k=min(test_count, len(train) - 1))
                test = train.loc[test_indices].copy()
                train = train.drop(index=test_indices)
            return {"train": train, "val": val, "test": test}

    rows = list(df.index)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n = len(rows)
    n_test = max(1 if n >= 3 else 0, int(round(n * test_fraction)))
    n_val = max(1 if n >= 3 else 0, int(round(n * val_fraction)))
    n_train = max(0, n - n_val - n_test)

    train_idx = rows[:n_train]
    val_idx = rows[n_train : n_train + n_val]
    test_idx = rows[n_train + n_val :]
    return {
        "train": df.loc[train_idx].copy(),
        "val": df.loc[val_idx].copy(),
        "test": df.loc[test_idx].copy(),
    }


def _write_filelist(
    rows: pd.DataFrame,
    path: str | Path,
    *,
    audio_column: str,
    text_column: str,
    tone_column: str,
    audio_root: str | None,
) -> Path:
    out_path = ensure_parent(path)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for _, row in rows.iterrows():
            audio_path = _with_audio_root(row[audio_column], audio_root)
            text = _clean_filelist_field(row[text_column])
            tone_sequence = _clean_filelist_field(row.get(tone_column, ""))
            f.write(f"{audio_path}|{text}|{tone_sequence}\n")
    return out_path


def prepare_filelists(
    manifest: str,
    *,
    output_dir: str = "data/filelists/full_vits",
    train_output: str | None = None,
    val_output: str | None = None,
    test_output: str | None = None,
    audio_column: str | None = None,
    text_column: str = "text",
    tone_column: str = "gemini_tone_sequence",
    split_column: str = "split",
    audio_root: str | None = None,
    val_fraction: float = 0.05,
    test_fraction: float = 0.05,
    seed: int = 42,
    max_rows: int | None = None,
) -> dict[str, Path]:
    manifest_path = resolve_path(manifest)
    df = pd.read_csv(manifest_path)
    if max_rows is not None:
        df = df.head(int(max_rows)).copy()

    columns = list(df.columns)
    resolved_audio_column = _first_existing_column(
        columns,
        audio_column,
        ["audio_path", "path", "wav_path", "file_path", "audio_filepath"],
    )
    if resolved_audio_column is None:
        raise ValueError(
            "Could not find an audio path column. Pass --audio-column explicitly."
        )
    if text_column not in columns:
        raise ValueError(f"Missing text column: {text_column}")
    if tone_column not in columns:
        raise ValueError(f"Missing tone column: {tone_column}")

    df = df.dropna(subset=[resolved_audio_column, text_column]).copy()
    if df.empty:
        raise ValueError("No rows remain after dropping missing audio/text values.")

    splits = _split_manifest(
        df,
        split_column=split_column,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    out_dir = ensure_dir(output_dir)
    outputs = {
        "train": train_output or str(out_dir / "train.txt"),
        "val": val_output or str(out_dir / "val.txt"),
        "test": test_output or str(out_dir / "test.txt"),
    }

    written = {}
    for split_name, split_df in splits.items():
        written[split_name] = _write_filelist(
            split_df,
            outputs[split_name],
            audio_column=resolved_audio_column,
            text_column=text_column,
            tone_column=tone_column,
            audio_root=audio_root,
        )

    return written


def main() -> None:
    args = parse_args()
    written = prepare_filelists(**vars(args))

    for split_name, path in written.items():
        n_rows = sum(1 for _ in path.open("r", encoding="utf-8"))
        print(f"{split_name}: {n_rows} rows -> {path}")


if __name__ == "__main__":
    main()
