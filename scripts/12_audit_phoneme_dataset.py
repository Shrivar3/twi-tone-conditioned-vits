from __future__ import annotations

import argparse
import csv
import os
import sys
import math
import re
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Iterator

from datasets import Audio, load_dataset


DEFAULT_DATASET_ID = "ghananlpcommunity/asante-twi-bible-speech-phonemes"
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

    # Keep Audio columns undecoded by default. This avoids importing TorchCodec,
    # which often fails in Codespaces/lightweight CPU environments when FFmpeg
    # libraries or a matching TorchCodec build are unavailable.
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


def _audio_duration(row: dict, duration_column: str | None, audio_column: str | None) -> float | None:
    if duration_column and row.get(duration_column) not in (None, ""):
        try:
            value = float(row[duration_column])
            return value if math.isfinite(value) else None
        except Exception:
            return None

    if not audio_column:
        return None

    audio = row.get(audio_column)
    if isinstance(audio, dict):
        if audio.get("duration") is not None:
            try:
                value = float(audio["duration"])
                return value if math.isfinite(value) else None
            except Exception:
                pass
        if audio.get("array") is not None and audio.get("sampling_rate"):
            try:
                return len(audio["array"]) / float(audio["sampling_rate"])
            except Exception:
                return None
    return None


def _write_counter_csv(path: Path, field_name: str, counter: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(counter.values()) or 1
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[field_name, "count", "share"])
        writer.writeheader()
        for key, count in counter.most_common():
            writer.writerow({field_name: key, "count": count, "share": f"{count / total:.8f}"})


def audit_phoneme_dataset(
    dataset_id: str,
    split: str,
    text_column: str,
    phoneme_column: str,
    speaker_column: str,
    duration_column: str | None,
    audio_column: str | None,
    decode_audio: bool,
    streaming: bool,
    limit: int | None,
    summary_output: str | Path,
    phoneme_output: str | Path,
    speaker_output: str | Path,
    long_clip_output: str | Path,
) -> None:
    summary_output = Path(summary_output)
    phoneme_output = Path(phoneme_output)
    speaker_output = Path(speaker_output)
    long_clip_output = Path(long_clip_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    long_clip_output.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    empty_text = 0
    empty_phonemes = 0
    missing_duration = 0
    durations: list[float] = []
    text_token_lengths: list[int] = []
    phoneme_token_lengths: list[int] = []
    phoneme_counter: Counter[str] = Counter()
    speaker_counts: Counter[str] = Counter()
    speaker_duration: dict[str, float] = defaultdict(float)
    long_rows: list[dict[str, str]] = []

    for idx, row in enumerate(_iter_rows(dataset_id, split, streaming, limit, audio_column, decode_audio)):
        n_rows += 1
        text = str(row.get(text_column, "") or "").strip()
        phonemes = str(row.get(phoneme_column, "") or "").strip()
        speaker = str(row.get(speaker_column, "") or "unknown")
        duration = _audio_duration(row, duration_column, audio_column)

        if not text:
            empty_text += 1
        if not phonemes:
            empty_phonemes += 1
        if duration is None:
            missing_duration += 1

        text_tokens = TOKEN_RE.findall(text)
        phoneme_tokens = TOKEN_RE.findall(phonemes)
        text_token_lengths.append(len(text_tokens))
        phoneme_token_lengths.append(len(phoneme_tokens))
        phoneme_counter.update(phoneme_tokens)
        speaker_counts[speaker] += 1

        if duration is not None:
            durations.append(duration)
            speaker_duration[speaker] += duration
            if duration > 20:
                long_rows.append(
                    {
                        "row_index": str(idx),
                        "speaker_id": speaker,
                        "duration_seconds": f"{duration:.3f}",
                        "text_tokens": str(len(text_tokens)),
                        "phoneme_tokens": str(len(phoneme_tokens)),
                        "text": text[:220],
                        "phonemes": phonemes[:220],
                    }
                )

    summary_rows: list[tuple[str, str]] = [
        ("dataset_id", dataset_id),
        ("split", split),
        ("streaming", str(streaming)),
        ("row_limit", str(limit)),
        ("audio_column", str(audio_column)),
        ("decode_audio", str(decode_audio)),
        ("rows_scanned", str(n_rows)),
        ("empty_text_rows", str(empty_text)),
        ("empty_phoneme_rows", str(empty_phonemes)),
        ("missing_duration_rows", str(missing_duration)),
        ("unique_phoneme_tokens", str(len(phoneme_counter))),
        ("unique_speakers", str(len(speaker_counts))),
    ]

    for name, values in [
        ("duration_seconds", durations),
        ("text_tokens", text_token_lengths),
        ("phoneme_tokens", phoneme_token_lengths),
    ]:
        if values:
            summary_rows.extend(
                [
                    (f"{name}_min", f"{min(values):.6g}"),
                    (f"{name}_p05", f"{_percentile(values, 0.05):.6g}"),
                    (f"{name}_median", f"{median(values):.6g}"),
                    (f"{name}_mean", f"{mean(values):.6g}"),
                    (f"{name}_p95", f"{_percentile(values, 0.95):.6g}"),
                    (f"{name}_max", f"{max(values):.6g}"),
                ]
            )

    with summary_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(summary_rows)

    _write_counter_csv(phoneme_output, "phoneme", phoneme_counter)

    with speaker_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["speaker_id", "rows", "duration_hours"])
        writer.writeheader()
        for speaker, count in speaker_counts.most_common():
            writer.writerow(
                {
                    "speaker_id": speaker,
                    "rows": count,
                    "duration_hours": f"{speaker_duration.get(speaker, 0.0) / 3600:.6f}",
                }
            )

    with long_clip_output.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "row_index",
            "speaker_id",
            "duration_seconds",
            "text_tokens",
            "phoneme_tokens",
            "text",
            "phonemes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(long_rows[:500])

    print(f"Saved phoneme dataset summary to {summary_output}")
    print(f"Saved phoneme inventory to {phoneme_output}")
    print(f"Saved speaker summary to {speaker_output}")
    print(f"Saved first {min(len(long_rows), 500)} long clips to {long_clip_output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the Asante Twi Bible speech phoneme dataset without decoding audio by default."
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--phoneme-column", default="phonemes")
    parser.add_argument("--speaker-column", default="speaker_id")
    parser.add_argument("--duration-column", default=None)
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--decode-audio", action="store_true", help="Decode waveform arrays. Off by default to avoid TorchCodec/FFmpeg issues.")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hard-exit", action=argparse.BooleanOptionalAction, default=True, help="Exit with os._exit(0) after writing outputs to avoid Codespaces Torch/CUDA shutdown aborts. Use --no-hard-exit for normal Python shutdown.")
    parser.add_argument("--summary-output", default="results/phoneme_dataset_audit_summary.csv")
    parser.add_argument("--phoneme-output", default="results/phoneme_inventory.csv")
    parser.add_argument("--speaker-output", default="results/phoneme_speaker_summary.csv")
    parser.add_argument("--long-clip-output", default="results/phoneme_long_clip_candidates.csv")
    args = parser.parse_args()

    audit_phoneme_dataset(
        dataset_id=args.dataset_id,
        split=args.split,
        text_column=args.text_column,
        phoneme_column=args.phoneme_column,
        speaker_column=args.speaker_column,
        duration_column=args.duration_column,
        audio_column=args.audio_column,
        decode_audio=args.decode_audio,
        streaming=args.streaming,
        limit=args.limit,
        summary_output=args.summary_output,
        phoneme_output=args.phoneme_output,
        speaker_output=args.speaker_output,
        long_clip_output=args.long_clip_output,
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
