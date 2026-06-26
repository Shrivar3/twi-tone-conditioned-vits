# scripts/15_build_tone_conditioning_manifest.py
from __future__ import annotations

import argparse

import pandas as pd
from transformers import AutoTokenizer

from src.tone.token_tone_alignment import (
    build_token_tone_encoding,
    tone_ids_to_json,
)
from src.utils.paths import ensure_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build token-level tone_id manifest for tone-conditioned VITS.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV containing at least a text column.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--model-id",
        default="FarmerlineML/main_twi_TTS",
        help="HF model/tokenizer id or local model path.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Column containing Twi text.",
    )
    parser.add_argument(
        "--tone-column",
        default="tone_sequence_gemini",
        help=(
            "Column containing token-level tone labels. "
            "If missing, all tones become UNK/NONE."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick Codespaces checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    if args.text_column not in df.columns:
        raise ValueError(f"Missing text column: {args.text_column}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    rows = []
    for _, row in df.iterrows():
        text = str(row.get(args.text_column, ""))
        tone_sequence = row.get(args.tone_column, None)

        if pd.isna(tone_sequence):
            tone_sequence = None
        else:
            tone_sequence = str(tone_sequence)

        encoding = build_token_tone_encoding(
            text=text,
            tokenizer=tokenizer,
            tone_sequence=tone_sequence,
        )

        out = row.to_dict()
        out["normalised_text_for_tone_alignment"] = encoding.normalised_text_for_alignment
        out["input_id_len"] = len(encoding.input_ids)
        out["char_tone_len"] = len(encoding.char_tone_ids)
        out["tone_id_len"] = len(encoding.tone_ids)
        out["tone_alignment_status"] = encoding.alignment_status
        out["tone_id_summary"] = encoding.tone_summary
        out["tone_ids_json"] = tone_ids_to_json(encoding.tone_ids)
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_path = ensure_parent(args.output)
    out_df.to_csv(out_path, index=False)

    print(f"Wrote {len(out_df)} rows to {out_path}")
    print(out_df["tone_alignment_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
