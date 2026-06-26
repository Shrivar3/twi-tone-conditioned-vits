# src/tone/tone_vocab.py
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

# Keep this intentionally small at first.
# PAD is used only for padded tone_id tensors.
# NONE is used for consonants, spaces, punctuation, inserted VITS blanks, etc.
DEFAULT_TONE_LABELS = [
    "PAD",
    "NONE",
    "UNK",
    "H",
    "L",
    "M",
    "F",
    "R",
    "HL",
    "LH",
    "DOWNSTEP",
]

TOKEN_RE = re.compile(
    r"[\wɛƐɔƆáàâāéèêēíìîīóòôōúùûūńǹḿ]+",
    flags=re.UNICODE,
)

VOWELS = set(
    "aeɛioɔu"
    "áàâāéèêēíìîīóòôōúùûū"
    "AEƐIOƆU"
    "ÁÀÂĀÉÈÊĒÍÌÎĪÓÒÔŌÚÙÛŪ"
)


@dataclass(frozen=True)
class ToneVocab:
    labels: tuple[str, ...] = tuple(DEFAULT_TONE_LABELS)

    @property
    def label_to_id(self) -> dict[str, int]:
        return {label: idx for idx, label in enumerate(self.labels)}

    @property
    def id_to_label(self) -> dict[int, str]:
        return {idx: label for idx, label in enumerate(self.labels)}

    @property
    def pad_id(self) -> int:
        return self.label_to_id["PAD"]

    @property
    def none_id(self) -> int:
        return self.label_to_id["NONE"]

    @property
    def unk_id(self) -> int:
        return self.label_to_id["UNK"]

    def encode(self, label: str | None) -> int:
        return self.label_to_id.get(normalise_tone_label(label), self.unk_id)

    def decode(self, idx: int) -> str:
        return self.id_to_label.get(int(idx), "UNK")


def normalise_tone_label(label: str | None) -> str:
    """Map Gemini/native-review labels into a small stable tone vocabulary."""
    if label is None:
        return "UNK"

    value = str(label).strip().upper()
    if not value:
        return "UNK"

    value = value.replace(" ", "")
    value = value.replace("_", "-")
    value = value.replace("→", "-")
    value = value.replace("/", "-")

    aliases = {
        "HIGH": "H",
        "HI": "H",
        "HIGHTONE": "H",
        "LOW": "L",
        "LOTONE": "L",
        "LOWTONE": "L",
        "MID": "M",
        "MIDTONE": "M",
        "FALL": "F",
        "FALLING": "F",
        "FALLINGTONE": "F",
        "RISE": "R",
        "RISING": "R",
        "RISINGTONE": "R",
        "UNKNOWN": "UNK",
        "UNKOWN": "UNK",
        "UNMARKED": "UNK",
        "NA": "UNK",
        "N/A": "UNK",
        "NONE": "NONE",
        "NO-TONE": "NONE",
        "NOTONE": "NONE",
        "0": "NONE",
        "-": "NONE",
        "!H": "DOWNSTEP",
        "DOWNSTEP": "DOWNSTEP",
        "D": "DOWNSTEP",
    }

    if value in aliases:
        return aliases[value]

    # Common contour spellings.
    if value in {"H-L", "HL", "HIGH-LOW"}:
        return "HL"
    if value in {"L-H", "LH", "LOW-HIGH"}:
        return "LH"

    if value in DEFAULT_TONE_LABELS:
        return value

    return "UNK"


def split_contour_label(label: str | None) -> list[str]:
    """Split a token-level label into vowel-level pieces where possible."""
    clean = normalise_tone_label(label)

    if clean == "HL":
        return ["H", "L"]
    if clean == "LH":
        return ["L", "H"]
    if clean in {"H", "L", "M", "F", "R", "DOWNSTEP", "UNK", "NONE"}:
        return [clean]

    return ["UNK"]


def is_vowel(char: str) -> bool:
    return char in VOWELS


def tokenise_text(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text))


def character_tone_labels_from_token_sequence(
    text: str,
    token_tone_sequence: str | Iterable[str] | None,
    *,
    default_label: str = "UNK",
) -> list[str]:
    """Expand token-level tone labels to one label per raw character.

    This is conservative:
    - vowels receive the token tone;
    - consonants receive NONE;
    - spaces/punctuation receive NONE;
    - missing or malformed labels become UNK.

    Example
    -------
    text = "me kɔ"
    token_tone_sequence = "H L"

    Output roughly:
    m -> NONE
    e -> H
    space -> NONE
    k -> NONE
    ɔ -> L
    """
    text = str(text)

    if token_tone_sequence is None:
        token_labels: list[str] = []
    elif isinstance(token_tone_sequence, str):
        token_labels = [x for x in token_tone_sequence.split() if x.strip()]
    else:
        token_labels = [str(x) for x in token_tone_sequence]

    matches = list(TOKEN_RE.finditer(text))
    char_labels = ["NONE"] * len(text)

    if not matches:
        return char_labels

    if len(token_labels) == 1 and len(matches) > 1:
        token_labels = token_labels * len(matches)

    for token_idx, match in enumerate(matches):
        token = match.group(0)
        tone_label = token_labels[token_idx] if token_idx < len(token_labels) else default_label
        tone_pieces = split_contour_label(tone_label)

        vowel_offsets = [i for i, ch in enumerate(token) if is_vowel(ch)]

        if not vowel_offsets:
            continue

        if len(tone_pieces) == 1:
            expanded = tone_pieces * len(vowel_offsets)
        else:
            # Assign contour pieces in order; reuse the final piece if there
            # are more vowels than contour components.
            expanded = [
                tone_pieces[min(i, len(tone_pieces) - 1)]
                for i in range(len(vowel_offsets))
            ]

        for offset, label in zip(vowel_offsets, expanded):
            char_labels[match.start() + offset] = normalise_tone_label(label)

    return char_labels


def summarise_tone_ids(tone_ids: Iterable[int], tone_vocab: ToneVocab | None = None) -> str:
    tone_vocab = tone_vocab or ToneVocab()
    counts: dict[str, int] = {}

    for idx in tone_ids:
        label = tone_vocab.decode(int(idx))
        counts[label] = counts.get(label, 0) + 1

    return ";".join(f"{label}:{count}" for label, count in sorted(counts.items()))
