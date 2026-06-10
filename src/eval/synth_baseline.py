from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, VitsModel

from src.utils.paths import ensure_dir


class BaselineSynthesiser:
    """Generic Transformers-style VITS synthesiser.

    If `FarmerlineML/twi-tts-2026` uses a custom class, adapt only this class.
    The rest of the Week 1 evaluation pipeline should not need to change.
    """

    def __init__(self, model_id: str, device: str | None = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = VitsModel.from_pretrained(model_id, trust_remote_code=True).to(self.device)
        self.model.eval()

    @property
    def sampling_rate(self) -> int:
        return int(getattr(self.model.config, "sampling_rate", 22050))

    @torch.inference_mode()
    def synthesise_one(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output = self.model(**inputs).waveform
        wav = output.squeeze().detach().cpu().numpy()
        return wav.astype(np.float32)


def synthesise_manifest(
    manifest_csv: str,
    model_id: str,
    output_dir: str,
    max_samples: int | None = None,
    mos_subset_only: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    if mos_subset_only and "mos_subset" in df.columns:
        df = df[df["mos_subset"].astype(bool)].copy()
    if max_samples is not None:
        df = df.head(int(max_samples)).copy()

    out_dir = ensure_dir(output_dir)
    synthesiser = BaselineSynthesiser(model_id)

    rows: list[dict[str, Any]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Synthesising baseline TTS"):
        utt_id = row["utt_id"]
        text = str(row["text"])
        wav = synthesiser.synthesise_one(text)
        audio_path = out_dir / f"{utt_id}.wav"
        sf.write(audio_path, wav, synthesiser.sampling_rate)
        out = row.to_dict()
        out["baseline_tts_audio_path"] = str(audio_path)
        rows.append(out)

    return pd.DataFrame(rows)
