# Prepare native-speaker tone validation pack.
#
# This script takes Gemini tone annotations and creates a reviewer-friendly
# token-level validation sheet.
#
# The sheet includes:
# - the original Twi/Akan text;
# - an automatic English gloss of the full text;
# - token-level English glosses where available;
# - Gemini candidate tone labels;
# - a reviewer_note_prompt column explaining what the reviewer should check;
# - blank reviewer columns for corrected native tone labels, meaning notes,
#   better native wording, and general reviewer notes.
#
# Usage:
#   PYTHONPATH=. python scripts/08_prepare_native_validation_pack.py
#
# Optional:
#   PYTHONPATH=. python scripts/08_prepare_native_validation_pack.py \
#     --input data/manifests/gemini_tone_annotated_dev.csv \
#     --output data/manifests/native_tone_validation_sheet.csv

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def safe_str(value: Any) -> str:
    """Convert a possibly missing CSV value to a clean string."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value)


def parse_items_json(value: Any) -> list[dict[str, Any]]:
    """Parse Gemini token-level JSON if it exists."""
    text = safe_str(value).strip()

    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    return [item for item in parsed if isinstance(item, dict)]


def split_pipe_field(value: Any) -> list[str]:
    """Split fields stored as 'a | b | c'."""
    text = safe_str(value).strip()

    if not text:
        return []

    return [part.strip() for part in text.split("|")]


def make_reviewer_note_prompt(
    content_text_twi: str,
    english_gloss_auto: str,
    token: str,
    token_english_gloss_auto: str,
    candidate_tone: str,
    candidate_reason: str,
) -> str:
    """Create a reviewer guidance note for each token row."""

    token_meaning_part = ""
    if token_english_gloss_auto:
        token_meaning_part = f" The automatic token gloss is: '{token_english_gloss_auto}'."

    reason_part = ""
    if candidate_reason:
        reason_part = f" Gemini reason: {candidate_reason}"

    return (
        f"Full Twi/Akan text: {content_text_twi} "
        f"Automatic English gloss of full text: {english_gloss_auto} "
        f"Token under review: '{token}'."
        f"{token_meaning_part} "
        f"Candidate tone label: {candidate_tone}."
        f"{reason_part} "
        "Please use reviewer_notes especially to correct the English meaning, "
        "suggest the right native word or phrase, flag unnatural Twi/Akan wording, "
        "explain context-dependent meanings, or explain why the candidate tone is wrong."
    )


def build_native_validation_sheet(
    input_csv: Path,
    output_csv: Path,
    text_column: str,
) -> Path:
    """Build token-level native validation CSV."""

    df = pd.read_csv(input_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict[str, Any]] = []

    for source_row_index, row in df.iterrows():
        utt_id = safe_str(row.get("utt_id", source_row_index))
        content_text_twi = safe_str(row.get(text_column, ""))

        # New preferred full-sentence English gloss column.
        english_gloss_auto = safe_str(row.get("gemini_english_gloss_auto", ""))

        # Backward-compatible fallback in case the older file used another name.
        if not english_gloss_auto:
            english_gloss_auto = safe_str(row.get("english_gloss_auto", ""))

        sentence_confidence = safe_str(row.get("gemini_sentence_confidence", ""))
        needs_native_review = safe_str(row.get("gemini_needs_native_review", ""))
        gemini_overall_comment = safe_str(row.get("gemini_overall_comment", ""))

        items = parse_items_json(row.get("gemini_items_json", ""))

        # Backward-compatible fallbacks for older Gemini annotation files.
        fallback_tokens = safe_str(row.get("gemini_tokens", "")).split()
        fallback_tones = safe_str(row.get("gemini_tone_sequence", "")).split()
        fallback_confidences = safe_str(row.get("gemini_token_confidences", "")).split()
        fallback_token_glosses = split_pipe_field(row.get("gemini_token_english_glosses", ""))

        if not items:
            n_items = max(
                len(fallback_tokens),
                len(fallback_tones),
                len(fallback_confidences),
                len(fallback_token_glosses),
            )

            for i in range(n_items):
                items.append(
                    {
                        "token": fallback_tokens[i] if i < len(fallback_tokens) else "",
                        "english_gloss_auto": (
                            fallback_token_glosses[i]
                            if i < len(fallback_token_glosses)
                            else ""
                        ),
                        "tone_sequence": fallback_tones[i] if i < len(fallback_tones) else "",
                        "confidence": (
                            fallback_confidences[i]
                            if i < len(fallback_confidences)
                            else ""
                        ),
                        "reason": "",
                    }
                )

        for token_index, item in enumerate(items):
            token = safe_str(item.get("token", ""))
            token_english_gloss_auto = safe_str(item.get("english_gloss_auto", ""))
            candidate_tone = safe_str(item.get("tone_sequence", ""))
            candidate_confidence = safe_str(item.get("confidence", ""))
            candidate_reason = safe_str(item.get("reason", ""))

            output_rows.append(
                {
                    "utt_id": utt_id,
                    "source_row_index": source_row_index,
                    "token_index": token_index,
                    "content_text_twi": content_text_twi,
                    "content_text_english_gloss_auto": english_gloss_auto,
                    "token": token,
                    "token_english_gloss_auto": token_english_gloss_auto,
                    "candidate_tone": candidate_tone,
                    "candidate_confidence": candidate_confidence,
                    "candidate_reason": candidate_reason,
                    "sentence_confidence": sentence_confidence,
                    "needs_native_review": needs_native_review,
                    "gemini_overall_comment": gemini_overall_comment,
                    "reviewer_note_prompt": make_reviewer_note_prompt(
                        content_text_twi=content_text_twi,
                        english_gloss_auto=english_gloss_auto,
                        token=token,
                        token_english_gloss_auto=token_english_gloss_auto,
                        candidate_tone=candidate_tone,
                        candidate_reason=candidate_reason,
                    ),
                    "native_tone_label": "",
                    "native_corrected_token": "",
                    "native_corrected_twi_sentence": "",
                    "native_english_meaning_note": "",
                    "auto_annotation_correct": "",
                    "review_status": "",
                    "reviewer_notes": "",
                }
            )

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_csv, index=False)

    print(f"Wrote {len(output_df)} token-level validation rows to {output_csv}")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare native-speaker Twi/Akan tone validation sheet."
    )

    parser.add_argument(
        "--input",
        default="data/manifests/gemini_tone_annotated_dev.csv",
        help="Input Gemini annotation CSV.",
    )

    parser.add_argument(
        "--output",
        default="data/manifests/native_tone_validation_sheet.csv",
        help="Output native-speaker validation CSV.",
    )

    parser.add_argument(
        "--text-column",
        default="text",
        help="Column containing original Twi/Akan text.",
    )

    args = parser.parse_args()

    build_native_validation_sheet(
        input_csv=Path(args.input),
        output_csv=Path(args.output),
        text_column=args.text_column,
    )


if __name__ == "__main__":
    main()
