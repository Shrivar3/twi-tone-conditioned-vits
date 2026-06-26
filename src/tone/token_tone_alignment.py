# src/tone/token_tone_alignment.py
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from typing import Any

import torch

from src.tone.tone_vocab import (
    ToneVocab,
    character_tone_labels_from_token_sequence,
    summarise_tone_ids,
)


@dataclass
class TokenToneEncoding:
    input_ids: list[int]
    attention_mask: list[int]
    tone_ids: list[int]
    normalised_text_for_alignment: str
    char_tone_ids: list[int]
    alignment_status: str
    tone_summary: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "tone_ids": self.tone_ids,
            "normalised_text_for_alignment": self.normalised_text_for_alignment,
            "char_tone_ids": self.char_tone_ids,
            "alignment_status": self.alignment_status,
            "tone_summary": self.tone_summary,
        }


def _as_list(value: Any) -> list[int]:
    """Handle tokenizer outputs that may be list[int] or list[list[int]]."""
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list):
        return list(value[0])
    return list(value)


def tokenizer_vocab(tokenizer: Any) -> dict[str, int]:
    if hasattr(tokenizer, "get_vocab"):
        return dict(tokenizer.get_vocab())
    if hasattr(tokenizer, "vocab"):
        return dict(tokenizer.vocab)
    raise TypeError("Tokenizer does not expose get_vocab() or vocab.")


def normalise_for_farmerline_vits_alignment(text: str, tokenizer: Any) -> str:
    """Approximate the character stream that Farmerline's VitsTokenizer sees.

    Farmerline's tokenizer_config has normalize=true and phonemize=false.
    This function does a conservative local approximation:
    - Unicode NFKC normalisation;
    - lowercasing;
    - keeping only characters present in the model vocabulary;
    - keeping spaces only if the vocab has a space token.

    The smoke-test script reports alignment mismatches so we can adjust this
    once we observe exact tokenizer behaviour on GPU/Codespaces.
    """
    vocab = tokenizer_vocab(tokenizer)
    text = unicodedata.normalize("NFKC", str(text)).lower()

    chars: list[str] = []
    previous_space = False

    for ch in text:
        if ch in vocab:
            if ch.isspace():
                if " " in vocab and not previous_space:
                    chars.append(" ")
                    previous_space = True
            else:
                chars.append(ch)
                previous_space = False
        elif ch.isspace() and " " in vocab and not previous_space:
            chars.append(" ")
            previous_space = True

    return "".join(chars).strip()


def encode_char_tones(
    text: str,
    tone_sequence: str | None,
    tokenizer: Any,
    tone_vocab: ToneVocab | None = None,
) -> tuple[str, list[int]]:
    tone_vocab = tone_vocab or ToneVocab()

    aligned_text = normalise_for_farmerline_vits_alignment(text, tokenizer)
    char_labels = character_tone_labels_from_token_sequence(
        aligned_text,
        tone_sequence,
        default_label="UNK",
    )

    char_tone_ids = [tone_vocab.encode(label) for label in char_labels]

    return aligned_text, char_tone_ids


def align_char_tones_to_vits_input_ids(
    input_ids: list[int],
    char_tone_ids: list[int],
    tone_vocab: ToneVocab | None = None,
) -> tuple[list[int], str]:
    """Expand char-level tones to VITS token positions.

    VITS tokenizers often insert blank token 0 between text symbols when
    add_blank=true. Farmerline's tokenizer_config has add_blank=true. The exact
    blank pattern can vary, so we handle common cases by length.

    We do not rely on token id 0 to detect blanks because Farmerline's vocab
    also maps "a" to id 0, and its pad token is "a". That is precisely why
    position-based alignment is safer here.
    """
    tone_vocab = tone_vocab or ToneVocab()
    none_id = tone_vocab.none_id

    n_input = len(input_ids)
    n_chars = len(char_tone_ids)

    if n_chars == 0:
        return [none_id] * n_input, "empty-char-tones"

    if n_input == n_chars:
        return list(char_tone_ids), "direct"

    if n_input == 2 * n_chars + 1:
        # blank, char, blank, char, ..., blank
        tone_ids: list[int] = []
        for tone_id in char_tone_ids:
            tone_ids.append(none_id)
            tone_ids.append(tone_id)
        tone_ids.append(none_id)
        return tone_ids, "vits-blank-before-between-after"

    if n_input == 2 * n_chars - 1:
        # char, blank, char, blank, ..., char
        tone_ids = []
        for idx, tone_id in enumerate(char_tone_ids):
            tone_ids.append(tone_id)
            if idx < n_chars - 1:
                tone_ids.append(none_id)
        return tone_ids, "vits-blank-between"

    if n_input == 2 * n_chars:
        # ambiguous, but common enough to support: blank, char repeated
        tone_ids = []
        for tone_id in char_tone_ids:
            tone_ids.append(none_id)
            tone_ids.append(tone_id)
        return tone_ids, "vits-blank-before-each-char"

    # Fallback: length mismatch. Keep model runnable, but report loudly.
    # This should be audited before training.
    tone_ids = []
    for j in range(n_input):
        src_idx = round(j * (n_chars - 1) / max(n_input - 1, 1))
        tone_ids.append(char_tone_ids[src_idx])

    return tone_ids, f"length-mismatch-input-{n_input}-chars-{n_chars}-fallback-resampled"


def build_token_tone_encoding(
    text: str,
    tokenizer: Any,
    tone_sequence: str | None = None,
    tone_vocab: ToneVocab | None = None,
) -> TokenToneEncoding:
    tone_vocab = tone_vocab or ToneVocab()

    encoded = tokenizer(str(text), return_attention_mask=True)
    input_ids = _as_list(encoded.get("input_ids"))
    attention_mask = _as_list(encoded.get("attention_mask"))

    if not attention_mask:
        attention_mask = [1] * len(input_ids)

    aligned_text, char_tone_ids = encode_char_tones(
        text=text,
        tone_sequence=tone_sequence,
        tokenizer=tokenizer,
        tone_vocab=tone_vocab,
    )

    tone_ids, alignment_status = align_char_tones_to_vits_input_ids(
        input_ids=input_ids,
        char_tone_ids=char_tone_ids,
        tone_vocab=tone_vocab,
    )

    if len(tone_ids) != len(input_ids):
        raise ValueError(
            f"tone_ids length {len(tone_ids)} != input_ids length {len(input_ids)}"
        )

    return TokenToneEncoding(
        input_ids=input_ids,
        attention_mask=attention_mask,
        tone_ids=tone_ids,
        normalised_text_for_alignment=aligned_text,
        char_tone_ids=char_tone_ids,
        alignment_status=alignment_status,
        tone_summary=summarise_tone_ids(tone_ids, tone_vocab=tone_vocab),
    )


def build_tone_conditioned_inputs(
    text: str,
    tokenizer: Any,
    tone_sequence: str | None = None,
    *,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    encoding = build_token_tone_encoding(
        text=text,
        tokenizer=tokenizer,
        tone_sequence=tone_sequence,
    )

    out = {
        "input_ids": torch.tensor([encoding.input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([encoding.attention_mask], dtype=torch.long),
        "tone_ids": torch.tensor([encoding.tone_ids], dtype=torch.long),
    }

    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}

    return out


def tone_ids_to_json(tone_ids: list[int]) -> str:
    return json.dumps([int(x) for x in tone_ids], ensure_ascii=False)
