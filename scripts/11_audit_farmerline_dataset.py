from __future__ import annotations

import argparse
import csv
import os
import sys
import math
import re
import unicodedata
from collections import Counter
from itertools import islice
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Iterator

from datasets import Audio, load_dataset


DEFAULT_DATASET_ID = "FarmerlineML/Twi_TTS2026_dataset"
TOKEN_RE = re.compile(r"\S+")


def _iter_rows(
    dataset_id: str,
    split: str,
    streaming: bool,
    limit: int | None,
    audio_column: str | None,
    decode_audio: bool,
) -> Iterator[dict]:
    ds = load_dataset(dataset_id, split=split, streaming=streaming)

    # Important: recent `datasets` versions decode Audio columns with TorchCodec.
    # In lightweight CPU/Codespaces environments this can fail because FFmpeg or
    # compatible TorchCodec binaries are missing. For text/duration audits we do
    # not need waveform arrays, so keep audio as path/bytes metadata by default.
    if audio_column and not decode_audio:
        try:
            ds = ds.cast_column(audio_column, Audio(decode=False))
        except Exception as exc:
            print(f"Warning: could not disable decoding for audio column {audio_column!r}: {exc}")

    iterator: Iterable[dict] = ds
    if limit is not None:
        iterator = islice(iterator, limit)
    yield from iterator


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    idx = (len(values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def _write_key_value_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def _write_counter_csv(path: Path, field_name: str, counter: Counter[str], extra_fn=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        base_fields = [field_name, "count"]
        extra_fields = ["unicode_name", "unicode_category"] if extra_fn else []
        writer = csv.DictWriter(f, fieldnames=base_fields + extra_fields)
        writer.writeheader()
        for key, count in counter.most_common():
            row = {field_name: key, "count": count}
            if extra_fn:
                row.update(extra_fn(key))
            writer.writerow(row)


def _char_info(ch: str) -> dict[str, str]:
    return {
        "unicode_name": unicodedata.name(ch, "UNKNOWN"),
        "unicode_category": unicodedata.category(ch),
    }


def audit_farmerline_dataset(
    dataset_id: str,
    split: str,
    text_column: str,
    duration_column: str,
    audio_column: str | None,
    decode_audio: bool,
    streaming: bool,
    limit: int | None,
    summary_output: str | Path,
    char_output: str | Path,
    token_output: str | Path,
    outlier_output: str | Path,
) -> None:
    summary_output = Path(summary_output)
    char_output = Path(char_output)
    token_output = Path(token_output)
    outlier_output = Path(outlier_output)
    outlier_output.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    empty_text = 0
    missing_duration = 0
    durations: list[float] = []
    text_lengths: list[int] = []
    token_lengths: list[int] = []
    chars: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    duplicate_texts: Counter[str] = Counter()
    outlier_rows: list[dict[str, str]] = []

    for idx, row in enumerate(_iter_rows(dataset_id, split, streaming, limit, audio_column, decode_audio)):
        n_rows += 1
        text = str(row.get(text_column, "") or "").strip()
        duration_raw = row.get(duration_column)
        duration = None
        if duration_raw not in (None, ""):
            try:
                duration = float(duration_raw)
                if math.isfinite(duration):
                    durations.append(duration)
            except Exception:
                duration = None
        if duration is None:
            missing_duration += 1

        if not text:
            empty_text += 1
            continue

        text_lengths.append(len(text))
        toks = TOKEN_RE.findall(text)
        token_lengths.append(len(toks))
        chars.update(text)
        tokens.update(tok.lower() for tok in toks)
        duplicate_texts[text] += 1

        if duration is not None and toks:
            chars_per_second = len(text) / max(duration, 1e-9)
            tokens_per_second = len(toks) / max(duration, 1e-9)
            if chars_per_second > 25 or chars_per_second < 2 or tokens_per_second > 5:
                outlier_rows.append(
                    {
                        "row_index": str(idx),
                        "duration": f"{duration:.3f}",
                        "chars": str(len(text)),
                        "tokens": str(len(toks)),
                        "chars_per_second": f"{chars_per_second:.3f}",
                        "tokens_per_second": f"{tokens_per_second:.3f}",
                        "text": text[:240],
                    }
                )

    duplicated_text_count = sum(1 for count in duplicate_texts.values() if count > 1)
    duplicated_row_count = sum(count for count in duplicate_texts.values() if count > 1)

    rows = [
        ("dataset_id", dataset_id),
        ("split", split),
        ("streaming", str(streaming)),
        ("row_limit", str(limit)),
        ("audio_column", str(audio_column)),
        ("decode_audio", str(decode_audio)),
        ("rows_scanned", str(n_rows)),
        ("empty_text_rows", str(empty_text)),
        ("missing_duration_rows", str(missing_duration)),
        ("unique_characters", str(len(chars))),
        ("unique_lowercase_tokens", str(len(tokens))),
        ("duplicated_unique_texts", str(duplicated_text_count)),
        ("duplicated_text_rows", str(duplicated_row_count)),
    ]

    for name, values in [("duration_seconds", durations), ("text_chars", text_lengths), ("text_tokens", token_lengths)]:
        if values:
            rows.extend(
                [
                    (f"{name}_min", f"{min(values):.6g}"),
                    (f"{name}_p05", f"{_percentile(values, 0.05):.6g}"),
                    (f"{name}_median", f"{median(values):.6g}"),
                    (f"{name}_mean", f"{mean(values):.6g}"),
                    (f"{name}_p95", f"{_percentile(values, 0.95):.6g}"),
                    (f"{name}_max", f"{max(values):.6g}"),
                ]
            )

    _write_key_value_csv(summary_output, rows)
    _write_counter_csv(char_output, "character", chars, _char_info)
    _write_counter_csv(token_output, "token", tokens)

    with outlier_output.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "row_index",
            "duration",
            "chars",
            "tokens",
            "chars_per_second",
            "tokens_per_second",
            "text",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(outlier_rows[:500])

    print(f"Saved dataset summary to {summary_output}")
    print(f"Saved character inventory to {char_output}")
    print(f"Saved token inventory to {token_output}")
    print(f"Saved first {min(len(outlier_rows), 500)} text/duration outliers to {outlier_output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Farmerline Twi TTS dataset without decoding audio by default."
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--duration-column", default="duration")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--decode-audio", action="store_true", help="Decode waveform arrays. Off by default to avoid TorchCodec/FFmpeg issues.")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hard-exit", action=argparse.BooleanOptionalAction, default=True, help="Exit with os._exit(0) after writing outputs to avoid Codespaces Torch/CUDA shutdown aborts. Use --no-hard-exit for normal Python shutdown.")
    parser.add_argument("--summary-output", default="results/farmerline_dataset_audit_summary.csv")
    parser.add_argument("--char-output", default="results/farmerline_character_inventory.csv")
    parser.add_argument("--token-output", default="results/farmerline_token_inventory.csv")
    parser.add_argument("--outlier-output", default="results/farmerline_text_duration_outliers.csv")
    args = parser.parse_args()

    audit_farmerline_dataset(
        dataset_id=args.dataset_id,
        split=args.split,
        text_column=args.text_column,
        duration_column=args.duration_column,
        audio_column=args.audio_column,
        decode_audio=args.decode_audio,
        streaming=args.streaming,
        limit=args.limit,
        summary_output=args.summary_output,
        char_output=args.char_output,
        token_output=args.token_output,
        outlier_output=args.outlier_output,
    )
    _hard_exit_if_requested(args.hard_exit)


def _hard_exit_if_requested(enabled: bool) -> None:
    """Bypass fragile extension-module finalizers in Codespaces after outputs are written.

    Some lightweight Codespaces images can abort during Python shutdown after
    Hugging Face Datasets imports Torch/CUDA-related extension modules. The CSV
    outputs are already flushed by this point; os._exit(0) avoids the shutdown
    path without hiding real errors from the audit itself.
    """
    if not enabled:
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
