from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

from datasets import Audio, load_dataset


DEFAULT_DATASET_ID = "ghananlpcommunity/asante-twi-bible-speech-phonemes"


def _iter_split(
    dataset_id: str,
    split: str,
    streaming: bool,
    limit_per_split: int | None,
    audio_column: str | None,
    decode_audio: bool,
) -> Iterator[dict]:
    ds = load_dataset(dataset_id, split=split, streaming=streaming)

    # Keep audio as metadata by default. This avoids TorchCodec/FFmpeg crashes in
    # lightweight CPU environments when we only need text/phoneme fields.
    if audio_column and not decode_audio:
        try:
            ds = ds.cast_column(audio_column, Audio(decode=False))
        except Exception as exc:
            print(f"Warning: could not disable decoding for audio column {audio_column!r}: {exc}")

    iterator: Iterable[dict] = ds
    if limit_per_split is not None:
        iterator = islice(iterator, limit_per_split)
    yield from iterator


def build_phoneme_inventory(
    dataset_id: str,
    splits: list[str],
    phoneme_column: str,
    text_column: str,
    streaming: bool,
    limit_per_split: int | None,
    audio_column: str | None,
    decode_audio: bool,
    output_csv: str | Path,
) -> Path:
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)

    for split in splits:
        print(f"Scanning {dataset_id} split={split}")
        for row in _iter_split(dataset_id, split, streaming, limit_per_split, audio_column, decode_audio):
            phoneme_string = str(row.get(phoneme_column, "") or "").strip()
            text = str(row.get(text_column, "") or "").strip()
            for phoneme in phoneme_string.split():
                counts[phoneme] += 1
                if text and len(examples[phoneme]) < 3:
                    examples[phoneme].append(text[:120])

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    total = sum(counts.values()) or 1

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["phoneme", "count", "share", "example_texts"],
        )
        writer.writeheader()
        for phoneme, count in counts.most_common():
            writer.writerow(
                {
                    "phoneme": phoneme,
                    "count": count,
                    "share": f"{count / total:.8f}",
                    "example_texts": " || ".join(examples.get(phoneme, [])),
                }
            )

    print(f"Saved phoneme inventory to {output_csv}")
    return output_csv


def _hard_exit_if_requested(enabled: bool) -> None:
    """Bypass fragile extension-module finalizers in Codespaces after outputs are written."""
    if not enabled:
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a phoneme inventory from the Asante Twi phoneme dataset."
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--phoneme-column", default="phonemes")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--decode-audio", action="store_true", help="Decode waveform arrays. Off by default to avoid TorchCodec/FFmpeg issues.")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--hard-exit", action=argparse.BooleanOptionalAction, default=True, help="Exit with os._exit(0) after writing outputs to avoid Codespaces Torch/CUDA shutdown aborts. Use --no-hard-exit for normal Python shutdown.")
    parser.add_argument("--output", default="results/phoneme_inventory.csv")
    args = parser.parse_args()

    build_phoneme_inventory(
        dataset_id=args.dataset_id,
        splits=args.splits,
        phoneme_column=args.phoneme_column,
        text_column=args.text_column,
        streaming=args.streaming,
        limit_per_split=args.limit_per_split,
        audio_column=args.audio_column,
        decode_audio=args.decode_audio,
        output_csv=args.output,
    )
    _hard_exit_if_requested(args.hard_exit)


if __name__ == "__main__":
    main()
