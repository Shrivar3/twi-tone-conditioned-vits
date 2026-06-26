from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


TRUE_VALUES = {"true", "yes", "1", "y"}
FALSE_VALUES = {"false", "no", "0", "n"}
VALID_REVIEW_VALUES = TRUE_VALUES | FALSE_VALUES


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def _parse_items_json(value: Any) -> list[dict[str, Any]]:
    text = _safe_str(value).strip()

    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    return [item for item in parsed if isinstance(item, dict)]


def _split_pipe_glosses(value: Any) -> list[str]:
    text = _safe_str(value)

    if not text:
        return []

    return [part.strip() for part in text.split("|")]


def _reviewer_note_prompt(
    content_text_twi: str,
    english_gloss_auto: str,
    token: str,
    token_english_gloss_auto: str,
    candidate_tone: str,
) -> str:
    return (
        f"Auto English gloss of full Twi text: {english_gloss_auto}. "
        f"Token under review: '{token}'"
        f"{f' = {token_english_gloss_auto}' if token_english_gloss_auto else ''}. "
        f"Candidate tone: {candidate_tone}. "
        "Please use reviewer_notes to correct the English meaning, flag unnatural Twi wording, "
        "give the right native word/phrase if needed, and correct the tone label if the candidate is wrong. "
        f"Full Twi text: {content_text_twi}"
    )


def build_native_validation_sheet(
    gemini_csv: str | Path,
    output_csv: str | Path,
    text_column: str = "text",
) -> Path:
    """Create a token-level native-speaker validation sheet.

    The sheet keeps the original Twi text untouched, adds Gemini's automatic English gloss,
    and gives the reviewer a dedicated prompt plus blank reviewer fields.
    """

    gemini_csv = Path(gemini_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(gemini_csv)
    out_rows: list[dict[str, Any]] = []

    for source_row_index, row in df.iterrows():
        utt_id = _safe_str(row.get("utt_id", source_row_index))
        content_text_twi = _safe_str(row.get(text_column, ""))
        english_gloss_auto = _safe_str(row.get("gemini_english_gloss_auto", ""))
        sentence_confidence = _safe_str(row.get("gemini_sentence_confidence", ""))
        needs_native_review = _safe_str(row.get("gemini_needs_native_review", ""))
        gemini_overall_comment = _safe_str(row.get("gemini_overall_comment", ""))

        items = _parse_items_json(row.get("gemini_items_json", ""))
        fallback_tokens = _safe_str(row.get("gemini_tokens", "")).split()
        fallback_tones = _safe_str(row.get("gemini_tone_sequence", "")).split()
        fallback_confidences = _safe_str(row.get("gemini_token_confidences", "")).split()
        fallback_glosses = _split_pipe_glosses(row.get("gemini_token_english_glosses", ""))

        if not items:
            items = []
            n_tokens = max(
                len(fallback_tokens),
                len(fallback_tones),
                len(fallback_confidences),
                len(fallback_glosses),
            )

            for i in range(n_tokens):
                items.append(
                    {
                        "token": fallback_tokens[i] if i < len(fallback_tokens) else "",
                        "english_gloss_auto": fallback_glosses[i] if i < len(fallback_glosses) else "",
                        "tone_sequence": fallback_tones[i] if i < len(fallback_tones) else "",
                        "confidence": fallback_confidences[i] if i < len(fallback_confidences) else "",
                        "reason": "",
                    }
                )

        for token_index, item in enumerate(items):
            token = _safe_str(item.get("token", ""))
            token_english_gloss_auto = _safe_str(item.get("english_gloss_auto", ""))
            candidate_tone = _safe_str(item.get("tone_sequence", ""))
            candidate_confidence = _safe_str(item.get("confidence", ""))
            candidate_reason = _safe_str(item.get("reason", ""))

            out_rows.append(
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
                    "reviewer_note_prompt": _reviewer_note_prompt(
                        content_text_twi=content_text_twi,
                        english_gloss_auto=english_gloss_auto,
                        token=token,
                        token_english_gloss_auto=token_english_gloss_auto,
                        candidate_tone=candidate_tone,
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

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(output_csv, index=False)

    print(f"Saved native validation sheet to {output_csv}")
    return output_csv


def summarise_native_validation(validation_csv: str | Path) -> dict[str, Any]:
    """Summarise a completed native-speaker validation sheet.

    Expects auto_annotation_correct to contain True/False, yes/no, 1/0, or blank.
    """

    df = pd.read_csv(validation_csv)

    if "auto_annotation_correct" not in df.columns:
        raise ValueError("Missing column: auto_annotation_correct")

    cleaned = df["auto_annotation_correct"].astype(str).str.lower().str.strip()
    valid = cleaned[cleaned.isin(VALID_REVIEW_VALUES)]
    correct = valid.isin(TRUE_VALUES)

    summary: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_reviewed": int(len(valid)),
        "n_correct": int(correct.sum()),
        "accuracy": float(correct.mean()) if len(valid) else None,
        "n_unreviewed": int(len(df) - len(valid)),
    }

    if "review_status" in df.columns:
        status_counts = (
            df["review_status"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", "blank")
            .value_counts()
            .to_dict()
        )
        summary["review_status_counts"] = {
            str(key): int(value) for key, value in status_counts.items()
        }

    if "reviewer_notes" in df.columns:
        notes = df["reviewer_notes"].fillna("").astype(str).str.strip()
        summary["n_rows_with_reviewer_notes"] = int((notes != "").sum())

    if "native_english_meaning_note" in df.columns:
        meaning_notes = df["native_english_meaning_note"].fillna("").astype(str).str.strip()
        summary["n_rows_with_native_english_meaning_note"] = int(
            (meaning_notes != "").sum()
        )

    return summary
