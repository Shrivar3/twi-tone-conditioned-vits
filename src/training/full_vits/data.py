from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.tone.token_tone_alignment import build_token_tone_encoding
from src.tone.tone_vocab import ToneVocab
from src.training.full_vits.commons import pad_1d, pad_2d
from src.training.full_vits.config import AudioConfig
from src.training.full_vits.mel import spectrogram_torch
from src.utils.paths import resolve_path


@dataclass
class FullVitsSample:
    sample_id: str
    audio_path: str
    text: str
    tone_sequence: str
    text_ids: torch.LongTensor
    tone_ids: torch.LongTensor
    spectrogram: torch.FloatTensor
    waveform: torch.FloatTensor
    text_length: int
    spec_length: int
    waveform_length: int


def _parse_filelist_line(line: str, *, line_number: int) -> tuple[str, str, str]:
    parts = line.rstrip("\n").split("|", maxsplit=2)
    if len(parts) < 2:
        raise ValueError(
            f"Filelist line {line_number} must be audio_path|text|gemini_tone_sequence."
        )
    if len(parts) == 2:
        parts.append("")
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def _resolve_audio_path(path: str, *, filelist_dir: Path, audio_root: Path | None) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw

    if audio_root is not None:
        candidate = audio_root / raw
        if candidate.exists():
            return candidate

    candidate = filelist_dir / raw
    if candidate.exists():
        return candidate

    return resolve_path(raw)


def _load_audio(path: Path, *, sampling_rate: int) -> torch.FloatTensor:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != sampling_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)
    return torch.tensor(np.asarray(audio, dtype=np.float32))


def _synthetic_waveform(*, sampling_rate: int, seconds: float) -> torch.FloatTensor:
    n = max(1, int(float(seconds) * int(sampling_rate)))
    t = torch.arange(n, dtype=torch.float32) / float(sampling_rate)
    return 0.05 * torch.sin(2.0 * torch.pi * 220.0 * t)


class FullVitsFilelistDataset(Dataset):
    """VITS filelist dataset with Farmerline tokenisation and tone IDs.

    Filelist format:
        audio_path|text|gemini_tone_sequence
    """

    def __init__(
        self,
        filelist_path: str | Path,
        *,
        tokenizer: Any,
        audio_config: AudioConfig | None = None,
        audio_root: str | Path | None = None,
        max_rows: int | None = None,
        allow_missing_audio: bool = False,
        synthetic_audio_seconds: float = 1.0,
    ) -> None:
        self.filelist_path = resolve_path(filelist_path)
        if not self.filelist_path.exists():
            raise FileNotFoundError(f"Filelist not found: {self.filelist_path}")

        self.filelist_dir = self.filelist_path.parent
        self.tokenizer = tokenizer
        self.audio_config = audio_config or AudioConfig()
        self.audio_root = resolve_path(audio_root) if audio_root else None
        self.allow_missing_audio = bool(allow_missing_audio)
        self.synthetic_audio_seconds = float(synthetic_audio_seconds)

        with self.filelist_path.open("r", encoding="utf-8") as f:
            rows = [
                _parse_filelist_line(line, line_number=i + 1)
                for i, line in enumerate(f)
                if line.strip()
            ]

        if max_rows is not None:
            rows = rows[: int(max_rows)]
        if not rows:
            raise ValueError(f"Filelist is empty: {self.filelist_path}")

        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def _load_waveform(self, audio_path: str) -> tuple[torch.FloatTensor, Path]:
        resolved = _resolve_audio_path(
            audio_path,
            filelist_dir=self.filelist_dir,
            audio_root=self.audio_root,
        )

        if resolved.exists():
            return _load_audio(resolved, sampling_rate=self.audio_config.sampling_rate), resolved

        if not self.allow_missing_audio:
            raise FileNotFoundError(f"Audio file not found: {resolved}")

        return (
            _synthetic_waveform(
                sampling_rate=self.audio_config.sampling_rate,
                seconds=self.synthetic_audio_seconds,
            ),
            resolved,
        )

    def __getitem__(self, idx: int) -> FullVitsSample:
        audio_path, text, tone_sequence = self.rows[idx]
        waveform, resolved_audio_path = self._load_waveform(audio_path)

        encoding = build_token_tone_encoding(
            text=text,
            tokenizer=self.tokenizer,
            tone_sequence=tone_sequence,
        )

        spectrogram = spectrogram_torch(
            waveform,
            filter_length=self.audio_config.filter_length,
            hop_length=self.audio_config.hop_length,
            win_length=self.audio_config.win_length,
        ).squeeze(0)

        return FullVitsSample(
            sample_id=f"{self.filelist_path.stem}_{idx:06d}",
            audio_path=str(resolved_audio_path),
            text=text,
            tone_sequence=tone_sequence,
            text_ids=torch.tensor(encoding.input_ids, dtype=torch.long),
            tone_ids=torch.tensor(encoding.tone_ids, dtype=torch.long),
            spectrogram=spectrogram.float(),
            waveform=waveform.float(),
            text_length=len(encoding.input_ids),
            spec_length=int(spectrogram.shape[-1]),
            waveform_length=int(waveform.numel()),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "filelist_path": str(self.filelist_path),
            "n_rows": len(self.rows),
            "audio_root": str(self.audio_root) if self.audio_root else None,
            "sampling_rate": self.audio_config.sampling_rate,
            "allow_missing_audio": self.allow_missing_audio,
        }


class FullVitsBatchCollator:
    def __init__(
        self,
        *,
        text_pad_id: int = 0,
        tone_pad_id: int | None = None,
        spec_pad_value: float = 0.0,
        waveform_pad_value: float = 0.0,
    ) -> None:
        tone_vocab = ToneVocab()
        self.text_pad_id = int(text_pad_id)
        self.tone_pad_id = tone_vocab.pad_id if tone_pad_id is None else int(tone_pad_id)
        self.spec_pad_value = float(spec_pad_value)
        self.waveform_pad_value = float(waveform_pad_value)

    def __call__(self, samples: list[FullVitsSample]) -> dict[str, Any]:
        text_ids, text_lengths = pad_1d(
            [sample.text_ids for sample in samples],
            pad_value=self.text_pad_id,
        )
        tone_ids, tone_lengths = pad_1d(
            [sample.tone_ids for sample in samples],
            pad_value=self.tone_pad_id,
        )
        spectrograms, spec_lengths = pad_2d(
            [sample.spectrogram for sample in samples],
            pad_value=self.spec_pad_value,
        )
        waveforms, waveform_lengths = pad_1d(
            [sample.waveform for sample in samples],
            pad_value=self.waveform_pad_value,
        )

        if not torch.equal(text_lengths, tone_lengths):
            raise ValueError("Text and tone lengths must match after collation.")

        return {
            "text_ids": text_ids.long(),
            "tone_ids": tone_ids.long(),
            "text_lengths": text_lengths.long(),
            "spectrograms": spectrograms.float(),
            "spec_lengths": spec_lengths.long(),
            "waveforms": waveforms.float(),
            "waveform_lengths": waveform_lengths.long(),
            "sample_ids": [sample.sample_id for sample in samples],
            "audio_paths": [sample.audio_path for sample in samples],
            "texts": [sample.text for sample in samples],
            "tone_sequences": [sample.tone_sequence for sample in samples],
        }
