from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.utils.paths import ensure_parent

# Combining marks used by many orthographies/transcriptions.
COMBINING_ACUTE = "\u0301"       # high-like mark
COMBINING_GRAVE = "\u0300"       # low-like mark
COMBINING_CIRCUMFLEX = "\u0302"  # falling/contour-like mark in many schemes
COMBINING_MACRON = "\u0304"      # mid/level marker in some schemes; we keep as unknown by default

TOKEN_RE = re.compile(r"[\wɛƐɔƆáàâāéèêēíìîīóòôōúùûūńǹḿ]+", flags=re.UNICODE)


@dataclass(frozen=True)
class ToneLabels:
    high: str = "H"
    low: str = "L"
    falling: str = "F"
    unknown: str = "UNK"


def character_tone(char: str, labels: ToneLabels = ToneLabels()) -> str | None:
    """Infer tone from a single Unicode character if it has a tone diacritic.

    This is intentionally conservative. Plain Twi orthography may not encode tone,
    so unmarked vowels/consonants should normally remain unknown rather than being
    guessed aggressively.
    """
    decomposed = unicodedata.normalize("NFD", char)
    if COMBINING_CIRCUMFLEX in decomposed:
        return labels.falling
    if COMBINING_ACUTE in decomposed:
        return labels.high
    if COMBINING_GRAVE in decomposed:
        return labels.low
    return None


def tokenise(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text))


def annotate_token(token: str, labels: ToneLabels = ToneLabels()) -> str:
    tones = [t for ch in token for t in [character_tone(ch, labels)] if t is not None]
    if not tones:
        return labels.unknown
    return "-".join(tones)


def annotate_text(text: str, labels: ToneLabels = ToneLabels()) -> tuple[list[str], list[str], float]:
    tokens = tokenise(text)
    token_tones = [annotate_token(tok, labels) for tok in tokens]
    if not tokens:
        return [], [], 0.0
    confidence = sum(t != labels.unknown for t in token_tones) / len(token_tones)
    return tokens, token_tones, confidence


def annotate_manifest(
    input_csv: str,
    output_csv: str,
    unknown_label: str = "UNK",
    high_label: str = "H",
    low_label: str = "L",
    falling_label: str = "F",
) -> pd.DataFrame:
    labels = ToneLabels(high=high_label, low=low_label, falling=falling_label, unknown=unknown_label)
    df = pd.read_csv(input_csv)

    rows = []
    for _, row in df.iterrows():
        tokens, tones, confidence = annotate_text(row.get("text", ""), labels)
        out = row.to_dict()
        out["tokens"] = " ".join(tokens)
        out["tone_sequence_auto"] = " ".join(tones)
        out["tone_confidence"] = confidence
        out["needs_native_review"] = confidence < 0.8
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_path = ensure_parent(output_csv)
    out_df.to_csv(out_path, index=False)
    return out_df


def make_native_validation_sheet(
    tone_df: pd.DataFrame,
    output_csv: str,
    subset_column: str = "tone_validation_subset",
) -> pd.DataFrame:
    if subset_column in tone_df.columns:
        val = tone_df[tone_df[subset_column].astype(bool)].copy()
    else:
        val = tone_df.copy()

    val["native_tone_sequence"] = ""
    val["native_reviewer_id"] = ""
    val["native_review_notes"] = ""
    val["auto_annotation_correct"] = ""

    columns = [
        "utt_id",
        "text",
        "tokens",
        "tone_sequence_auto",
        "tone_confidence",
        "native_tone_sequence",
        "auto_annotation_correct",
        "native_reviewer_id",
        "native_review_notes",
    ]
    columns = [c for c in columns if c in val.columns]
    val = val[columns]
    out_path = ensure_parent(output_csv)
    val.to_csv(out_path, index=False)
    return val
