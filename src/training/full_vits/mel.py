from __future__ import annotations

from functools import lru_cache

import librosa
import numpy as np
import torch
import torch.nn.functional as F


def dynamic_range_compression(x: torch.Tensor, clip_val: float = 1e-5) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val))


def spectrogram_torch(
    y: torch.Tensor,
    *,
    filter_length: int,
    hop_length: int,
    win_length: int,
    center: bool = True,
) -> torch.Tensor:
    if y.ndim == 1:
        y = y.unsqueeze(0)
    if y.ndim != 2:
        raise ValueError(f"Expected waveform [batch, time], got {tuple(y.shape)}")

    if y.size(-1) < filter_length:
        y = F.pad(y, (0, filter_length - y.size(-1)))

    window = torch.hann_window(win_length, device=y.device, dtype=y.dtype)
    spec = torch.stft(
        y,
        n_fft=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    return torch.abs(spec).clamp_min(1e-7)


@lru_cache(maxsize=16)
def _mel_basis(
    sampling_rate: int,
    filter_length: int,
    n_mel_channels: int,
    mel_fmin: float,
    mel_fmax: float | None,
) -> np.ndarray:
    return librosa.filters.mel(
        sr=sampling_rate,
        n_fft=filter_length,
        n_mels=n_mel_channels,
        fmin=mel_fmin,
        fmax=mel_fmax,
    ).astype(np.float32)


def spec_to_mel_torch(
    spec: torch.Tensor,
    *,
    sampling_rate: int,
    filter_length: int,
    n_mel_channels: int,
    mel_fmin: float = 0.0,
    mel_fmax: float | None = None,
) -> torch.Tensor:
    basis = torch.tensor(
        _mel_basis(
            sampling_rate,
            filter_length,
            n_mel_channels,
            mel_fmin,
            mel_fmax,
        ),
        dtype=spec.dtype,
        device=spec.device,
    )
    mel = torch.matmul(basis, spec)
    return dynamic_range_compression(mel)


def mel_spectrogram_torch(
    y: torch.Tensor,
    *,
    sampling_rate: int,
    filter_length: int,
    hop_length: int,
    win_length: int,
    n_mel_channels: int,
    mel_fmin: float = 0.0,
    mel_fmax: float | None = None,
    center: bool = True,
) -> torch.Tensor:
    spec = spectrogram_torch(
        y,
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        center=center,
    )
    return spec_to_mel_torch(
        spec,
        sampling_rate=sampling_rate,
        filter_length=filter_length,
        n_mel_channels=n_mel_channels,
        mel_fmin=mel_fmin,
        mel_fmax=mel_fmax,
    )
