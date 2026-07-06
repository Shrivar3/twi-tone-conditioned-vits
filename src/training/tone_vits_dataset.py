from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.tone.token_tone_alignment import build_token_tone_encoding


@dataclass
class ToneVitsSample:
    """Single sample for tone-conditioned VITS debugging/training."""

    row_index: int
    sample_id: str
    text: str
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    tone_ids: torch.LongTensor
    audio_values: torch.FloatTensor
    audio_sampling_rate: int
    audio_path: str | None
    has_real_audio: bool


def _parse_int_list(value: Any, *, field_name: str) -> list[int]:
    """Parse JSON/list-like CSV fields into list[int]."""
    if isinstance(value, list):
        return [int(x) for x in value]

    if isinstance(value, np.ndarray):
        return [int(x) for x in value.tolist()]

    if pd.isna(value):
        raise ValueError(f"{field_name} is missing.")

    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} is empty.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)

    if not isinstance(parsed, list):
        raise ValueError(f"{field_name} must parse to a list, got {type(parsed)}.")

    return [int(x) for x in parsed]


def _first_existing_column(
    columns: list[str],
    requested: str | None,
    candidates: list[str],
) -> str | None:
    if requested and requested in columns:
        return requested

    for col in candidates:
        if col in columns:
            return col

    return None


def _resolve_audio_path(
    raw_path: Any,
    *,
    manifest_dir: Path,
    audio_root: Path | None,
) -> Path | None:
    if raw_path is None or pd.isna(raw_path):
        return None

    value = str(raw_path).strip()
    if not value:
        return None

    path = Path(value)
    if path.is_absolute():
        return path

    if audio_root is not None:
        candidate = audio_root / path
        if candidate.exists():
            return candidate

    candidate = manifest_dir / path
    if candidate.exists():
        return candidate

    if audio_root is not None:
        return audio_root / path

    return candidate


def _load_audio(
    audio_path: Path,
    *,
    target_sampling_rate: int,
) -> torch.FloatTensor:
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != target_sampling_rate:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=target_sampling_rate,
        )

    audio = np.asarray(audio, dtype=np.float32)
    return torch.from_numpy(audio)


class ToneVitsManifestDataset(Dataset):
    """CSV-backed dataset for tone-conditioned VITS experiments.

    This dataset supports two modes:

    1. Precomputed mode:
       The manifest already contains input_ids / attention_mask / tone_ids.

    2. Build-on-load mode:
       The manifest contains text + tone sequence, and a tokenizer is supplied.
       The dataset builds input_ids and tone_ids using the repo's alignment code.

    Audio is optional for dry-run debugging. Real training/proxy training should use
    real audio paths.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        tokenizer: Any | None = None,
        text_column: str = "text",
        tone_sequence_column: str = "gemini_tone_sequence",
        input_ids_column: str | None = "input_ids",
        attention_mask_column: str | None = "attention_mask",
        tone_ids_column: str | None = "tone_ids",
        audio_column: str | None = None,
        sample_id_column: str | None = None,
        audio_root: str | Path | None = None,
        target_sampling_rate: int = 16000,
        max_rows: int | None = None,
        allow_missing_audio: bool = False,
        synthetic_audio_seconds: float = 1.0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest_dir = self.manifest_path.parent
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.tone_sequence_column = tone_sequence_column
        self.target_sampling_rate = int(target_sampling_rate)
        self.allow_missing_audio = bool(allow_missing_audio)
        self.synthetic_audio_seconds = float(synthetic_audio_seconds)
        self.audio_root = Path(audio_root) if audio_root else None

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        df = pd.read_csv(self.manifest_path)
        if max_rows is not None:
            df = df.head(int(max_rows)).copy()

        if df.empty:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")

        columns = list(df.columns)

        self.input_ids_column = _first_existing_column(
            columns,
            input_ids_column,
            ["input_ids", "vits_input_ids", "token_ids"],
        )
        self.attention_mask_column = _first_existing_column(
            columns,
            attention_mask_column,
            ["attention_mask", "vits_attention_mask"],
        )
        self.tone_ids_column = _first_existing_column(
            columns,
            tone_ids_column,
            ["tone_ids", "vits_tone_ids"],
        )
        self.audio_column = _first_existing_column(
            columns,
            audio_column,
            [
                "audio_path",
                "path",
                "wav_path",
                "file_path",
                "audio_filepath",
                "audio",
            ],
        )
        self.sample_id_column = _first_existing_column(
            columns,
            sample_id_column,
            [
                "sample_id",
                "utterance_id",
                "sentence_id",
                "id",
                "row_id",
            ],
        )

        has_precomputed_ids = (
            self.input_ids_column is not None
            and self.tone_ids_column is not None
        )

        if not has_precomputed_ids:
            if tokenizer is None:
                raise ValueError(
                    "Manifest does not contain precomputed input_ids/tone_ids. "
                    "Pass a tokenizer so they can be built from text + tone sequence."
                )
            if text_column not in columns:
                raise ValueError(f"Missing text column: {text_column}")
            if tone_sequence_column not in columns:
                raise ValueError(
                    f"Missing tone sequence column: {tone_sequence_column}"
                )

        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _build_ids_from_text(self, row: pd.Series) -> tuple[list[int], list[int], list[int]]:
        encoding = build_token_tone_encoding(
            text=str(row[self.text_column]),
            tokenizer=self.tokenizer,
            tone_sequence=str(row[self.tone_sequence_column]),
        )
        return encoding.input_ids, encoding.attention_mask, encoding.tone_ids

    def _load_or_create_audio(self, row: pd.Series) -> tuple[torch.FloatTensor, str | None, bool]:
        audio_path: Path | None = None

        if self.audio_column is not None:
            audio_path = _resolve_audio_path(
                row[self.audio_column],
                manifest_dir=self.manifest_dir,
                audio_root=self.audio_root,
            )

        if audio_path is not None and audio_path.exists():
            audio = _load_audio(
                audio_path,
                target_sampling_rate=self.target_sampling_rate,
            )
            return audio, str(audio_path), True

        if not self.allow_missing_audio:
            details = (
                f"row={row.name}, audio_column={self.audio_column}, "
                f"audio_path={audio_path}"
            )
            raise FileNotFoundError(
                "Audio file is missing. For dry-run only, pass "
                "--allow-missing-audio. Details: " + details
            )

        n = max(1, int(self.synthetic_audio_seconds * self.target_sampling_rate))
        return torch.zeros(n, dtype=torch.float32), None, False

    def __getitem__(self, idx: int) -> ToneVitsSample:
        row = self.df.iloc[idx]

        text = str(row[self.text_column]) if self.text_column in self.df.columns else ""

        if self.input_ids_column and self.tone_ids_column:
            input_ids = _parse_int_list(
                row[self.input_ids_column],
                field_name=self.input_ids_column,
            )
            tone_ids = _parse_int_list(
                row[self.tone_ids_column],
                field_name=self.tone_ids_column,
            )
            if self.attention_mask_column:
                attention_mask = _parse_int_list(
                    row[self.attention_mask_column],
                    field_name=self.attention_mask_column,
                )
            else:
                attention_mask = [1] * len(input_ids)
        else:
            input_ids, attention_mask, tone_ids = self._build_ids_from_text(row)

        if len(input_ids) != len(tone_ids):
            raise ValueError(
                f"input_ids and tone_ids length mismatch at row {idx}: "
                f"{len(input_ids)} vs {len(tone_ids)}"
            )

        if len(attention_mask) != len(input_ids):
            raise ValueError(
                f"attention_mask and input_ids length mismatch at row {idx}: "
                f"{len(attention_mask)} vs {len(input_ids)}"
            )

        audio, audio_path, has_real_audio = self._load_or_create_audio(row)

        if self.sample_id_column:
            sample_id = str(row[self.sample_id_column])
        else:
            sample_id = f"row_{idx:05d}"

        return ToneVitsSample(
            row_index=idx,
            sample_id=sample_id,
            text=text,
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            tone_ids=torch.tensor(tone_ids, dtype=torch.long),
            audio_values=audio.float(),
            audio_sampling_rate=self.target_sampling_rate,
            audio_path=audio_path,
            has_real_audio=has_real_audio,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "n_rows": len(self.df),
            "input_ids_column": self.input_ids_column,
            "attention_mask_column": self.attention_mask_column,
            "tone_ids_column": self.tone_ids_column,
            "text_column": self.text_column,
            "tone_sequence_column": self.tone_sequence_column,
            "audio_column": self.audio_column,
            "sample_id_column": self.sample_id_column,
            "target_sampling_rate": self.target_sampling_rate,
            "allow_missing_audio": self.allow_missing_audio,
        }
