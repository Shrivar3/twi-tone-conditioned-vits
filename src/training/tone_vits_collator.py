from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from src.tone.tone_vocab import ToneVocab
from src.training.tone_vits_dataset import ToneVitsSample


class ToneVitsBatchCollator:
    """Pad text/tone/audio fields for tone-conditioned VITS batches."""

    def __init__(
        self,
        *,
        input_pad_id: int = 0,
        tone_pad_id: int | None = None,
        audio_pad_value: float = 0.0,
    ) -> None:
        tone_vocab = ToneVocab()
        self.input_pad_id = int(input_pad_id)
        self.tone_pad_id = tone_vocab.pad_id if tone_pad_id is None else int(tone_pad_id)
        self.audio_pad_value = float(audio_pad_value)

    @staticmethod
    def _pad_1d_long(
        values: list[torch.LongTensor],
        *,
        pad_value: int,
    ) -> torch.LongTensor:
        max_len = max(int(x.numel()) for x in values)
        out = torch.full(
            (len(values), max_len),
            fill_value=int(pad_value),
            dtype=torch.long,
        )
        for i, x in enumerate(values):
            out[i, : x.numel()] = x
        return out

    @staticmethod
    def _pad_1d_float(
        values: list[torch.FloatTensor],
        *,
        pad_value: float,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        max_len = max(int(x.numel()) for x in values)
        out = torch.full(
            (len(values), max_len),
            fill_value=float(pad_value),
            dtype=torch.float32,
        )
        mask = torch.zeros((len(values), max_len), dtype=torch.float32)
        lengths = torch.zeros((len(values),), dtype=torch.long)

        for i, x in enumerate(values):
            n = int(x.numel())
            out[i, :n] = x
            mask[i, :n] = 1.0
            lengths[i] = n

        return out, mask, lengths

    def __call__(self, samples: list[ToneVitsSample]) -> dict[str, Any]:
        input_ids = self._pad_1d_long(
            [s.input_ids for s in samples],
            pad_value=self.input_pad_id,
        )

        attention_mask = self._pad_1d_long(
            [s.attention_mask for s in samples],
            pad_value=0,
        )

        tone_ids = self._pad_1d_long(
            [s.tone_ids for s in samples],
            pad_value=self.tone_pad_id,
        )

        audio_values, audio_attention_mask, audio_lengths = self._pad_1d_float(
            [s.audio_values for s in samples],
            pad_value=self.audio_pad_value,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tone_ids": tone_ids,
            "audio_values": audio_values,
            "audio_attention_mask": audio_attention_mask,
            "audio_lengths": audio_lengths,
            "sample_ids": [s.sample_id for s in samples],
            "texts": [s.text for s in samples],
            "audio_paths": [s.audio_path for s in samples],
            "has_real_audio": torch.tensor(
                [bool(s.has_real_audio) for s in samples],
                dtype=torch.bool,
            ),
            "raw_samples": [asdict(s) for s in samples],
        }
