from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import unicodedata
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

from datasets import Audio, load_dataset


FARMERLINE_DATASET_ID = "FarmerlineML/Twi_TTS2026_dataset"
PHONEME_DATASET_ID = "ghananlpcommunity/asante-twi-bible-speech-phonemes"
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

    # Keep audio as metadata by default. This avoids TorchCodec/FFmpeg crashes in
    # lightweight CPU environments when we only need text fields.
    if audio_column and not decode_audio:
        try:
            ds = ds.cast_column(audio_column, Audio(decode=False))
        except Exception as exc:
            print(f"Warning: could not disable decoding for audio column {audio_column!r}: {exc}")

    iterator: Iterable[dict] = ds
    if limit is not None:
        iterator = islice(iterator, limit)
    yield from iterator


def _collect_text_stats(
    dataset_id: str,
    split: str,
    text_column: str,
    streaming: bool,
    limit: int | None,
    audio_column: str | None,
    decode_audio: bool,
) -> tuple[Counter[str], Counter[str]]:
    chars: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    for row in _iter_rows(dataset_id, split, streaming, limit, audio_column, decode_audio):
        text = str(row.get(text_column, "") or "").strip()
        chars.update(text)
        tokens.update(tok.lower() for tok in TOKEN_RE.findall(text))
    return chars, tokens


def _write_overlap_csv(
    path: Path,
    item_name: str,
    farmerline_counts: Counter[str],
    phoneme_counts: Counter[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_items = sorted(set(farmerline_counts) | set(phoneme_counts))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                item_name,
                "farmerline_count",
                "phoneme_dataset_count",
                "in_farmerline",
                "in_phoneme_dataset",
                "unicode_name",
                "unicode_category",
            ],
        )
        writer.writeheader()
        for item in all_items:
            unicode_name = ""
            unicode_category = ""
            if item_name == "character" and len(item) == 1:
                unicode_name = unicodedata.name(item, "UNKNOWN")
                unicode_category = unicodedata.category(item)
            writer.writerow(
                {
                    item_name: item,
                    "farmerline_count": farmerline_counts.get(item, 0),
                    "phoneme_dataset_count": phoneme_counts.get(item, 0),
                    "in_farmerline": item in farmerline_counts,
                    "in_phoneme_dataset": item in phoneme_counts,
                    "unicode_name": unicode_name,
                    "unicode_category": unicode_category,
                }
            )


def compare_vocab(
    farmerline_dataset_id: str,
    farmerline_split: str,
    phoneme_dataset_id: str,
    phoneme_split: str,
    text_column: str,
    streaming: bool,
    farmerline_limit: int | None,
    phoneme_limit: int | None,
    audio_column: str | None,
    decode_audio: bool,
    char_output: str | Path,
    token_output: str | Path,
    summary_output: str | Path,
) -> None:
    print("Collecting Farmerline text statistics")
    farmer_chars, farmer_tokens = _collect_text_stats(
        farmerline_dataset_id, farmerline_split, text_column, streaming, farmerline_limit, audio_column, decode_audio
    )
    print("Collecting phoneme-dataset text statistics")
    phoneme_chars, phoneme_tokens = _collect_text_stats(
        phoneme_dataset_id, phoneme_split, text_column, streaming, phoneme_limit, audio_column, decode_audio
    )

    char_output = Path(char_output)
    token_output = Path(token_output)
    summary_output = Path(summary_output)
    _write_overlap_csv(char_output, "character", farmer_chars, phoneme_chars)
    _write_overlap_csv(token_output, "token", farmer_tokens, phoneme_tokens)

    farmer_only_chars = set(farmer_chars) - set(phoneme_chars)
    phoneme_only_chars = set(phoneme_chars) - set(farmer_chars)
    farmer_only_tokens = set(farmer_tokens) - set(phoneme_tokens)
    phoneme_only_tokens = set(phoneme_tokens) - set(farmer_tokens)

    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with summary_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(
            [
                ("farmerline_dataset_id", farmerline_dataset_id),
                ("farmerline_split", farmerline_split),
                ("phoneme_dataset_id", phoneme_dataset_id),
                ("phoneme_split", phoneme_split),
                ("farmerline_unique_chars", len(farmer_chars)),
                ("phoneme_dataset_unique_chars", len(phoneme_chars)),
                ("farmerline_only_chars", "".join(sorted(farmer_only_chars))),
                ("phoneme_dataset_only_chars", "".join(sorted(phoneme_only_chars))),
                ("farmerline_unique_tokens", len(farmer_tokens)),
                ("phoneme_dataset_unique_tokens", len(phoneme_tokens)),
                ("farmerline_only_token_count", len(farmer_only_tokens)),
                ("phoneme_dataset_only_token_count", len(phoneme_only_tokens)),
            ]
        )

    print(f"Saved character overlap to {char_output}")
    print(f"Saved token overlap to {token_output}")
    print(f"Saved comparison summary to {summary_output}")


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
        description="Compare Farmerline Twi text vocabulary against the Asante Twi phoneme dataset transcript vocabulary."
    )
    parser.add_argument("--farmerline-dataset-id", default=FARMERLINE_DATASET_ID)
    parser.add_argument("--farmerline-split", default="train")
    parser.add_argument("--phoneme-dataset-id", default=PHONEME_DATASET_ID)
    parser.add_argument("--phoneme-split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--decode-audio", action="store_true", help="Decode waveform arrays. Off by default to avoid TorchCodec/FFmpeg issues.")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--farmerline-limit", type=int, default=None)
    parser.add_argument("--phoneme-limit", type=int, default=None)
    parser.add_argument("--hard-exit", action=argparse.BooleanOptionalAction, default=True, help="Exit with os._exit(0) after writing outputs to avoid Codespaces Torch/CUDA shutdown aborts. Use --no-hard-exit for normal Python shutdown.")
    parser.add_argument("--char-output", default="results/farmerline_vs_phoneme_character_overlap.csv")
    parser.add_argument("--token-output", default="results/farmerline_vs_phoneme_token_overlap.csv")
    parser.add_argument("--summary-output", default="results/farmerline_vs_phoneme_vocab_summary.csv")
    args = parser.parse_args()

    compare_vocab(
        farmerline_dataset_id=args.farmerline_dataset_id,
        farmerline_split=args.farmerline_split,
        phoneme_dataset_id=args.phoneme_dataset_id,
        phoneme_split=args.phoneme_split,
        text_column=args.text_column,
        streaming=args.streaming,
        farmerline_limit=args.farmerline_limit,
        phoneme_limit=args.phoneme_limit,
        audio_column=args.audio_column,
        decode_audio=args.decode_audio,
        char_output=args.char_output,
        token_output=args.token_output,
        summary_output=args.summary_output,
    )
    _hard_exit_if_requested(args.hard_exit)


if __name__ == "__main__":
    main()
