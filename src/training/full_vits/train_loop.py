from __future__ import annotations

from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torch.nn.functional as F

from src.training.full_vits.commons import crop_to_match, slice_waveform_segments
from src.training.full_vits.config import AudioConfig
from src.training.full_vits.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from src.training.full_vits.mel import mel_spectrogram_torch
from src.training.full_vits.models import MultiPeriodDiscriminator, ToneConditionedVitsGenerator


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _waveform_segment(
    batch: dict[str, Any],
    ids_slice: torch.Tensor,
    *,
    segment_size: int,
    hop_length: int,
) -> torch.Tensor:
    return slice_waveform_segments(
        batch["waveforms"],
        ids_slice,
        segment_size=segment_size,
        hop_length=hop_length,
    )


def _mel_loss(
    y: torch.Tensor,
    y_hat: torch.Tensor,
    audio_config: AudioConfig,
) -> torch.Tensor:
    y, y_hat = crop_to_match(y, y_hat)
    mel_y = mel_spectrogram_torch(
        y.squeeze(1),
        sampling_rate=audio_config.sampling_rate,
        filter_length=audio_config.filter_length,
        hop_length=audio_config.hop_length,
        win_length=audio_config.win_length,
        n_mel_channels=audio_config.n_mel_channels,
        mel_fmin=audio_config.mel_fmin,
        mel_fmax=audio_config.mel_fmax,
    )
    mel_y_hat = mel_spectrogram_torch(
        y_hat.squeeze(1),
        sampling_rate=audio_config.sampling_rate,
        filter_length=audio_config.filter_length,
        hop_length=audio_config.hop_length,
        win_length=audio_config.win_length,
        n_mel_channels=audio_config.n_mel_channels,
        mel_fmin=audio_config.mel_fmin,
        mel_fmax=audio_config.mel_fmax,
    )
    mel_y, mel_y_hat = crop_to_match(mel_y, mel_y_hat)
    return F.l1_loss(mel_y, mel_y_hat)


def training_step(
    batch: dict[str, Any],
    *,
    generator: ToneConditionedVitsGenerator,
    discriminator: MultiPeriodDiscriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    audio_config: AudioConfig,
    loss_weights: dict[str, float],
    grad_clip: float | None = None,
) -> dict[str, float]:
    generator.train()
    discriminator.train()

    outputs = generator(
        text_ids=batch["text_ids"],
        text_lengths=batch["text_lengths"],
        tone_ids=batch["tone_ids"],
        spectrogram=batch["spectrograms"],
        spec_lengths=batch["spec_lengths"],
    )

    y = _waveform_segment(
        batch,
        outputs["ids_slice"],
        segment_size=generator.config.segment_size,
        hop_length=audio_config.hop_length,
    )
    y_hat = outputs["y_hat"]
    y, y_hat = crop_to_match(y, y_hat)

    optimizer_d.zero_grad(set_to_none=True)
    y_d_rs, y_d_gs, _, _ = discriminator(y, y_hat.detach())
    loss_disc, _, _ = discriminator_loss(y_d_rs, y_d_gs)
    loss_disc.backward()
    if grad_clip and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
    optimizer_d.step()

    optimizer_g.zero_grad(set_to_none=True)
    y_d_rs, y_d_gs, fmap_rs, fmap_gs = discriminator(y, y_hat)
    loss_gen_adv, _ = generator_loss(y_d_gs)
    loss_fm = feature_loss(fmap_rs, fmap_gs)
    loss_mel = _mel_loss(y, y_hat, audio_config)
    loss_kl = kl_loss(
        outputs["z_p"],
        outputs["logs_q"],
        outputs["m_p"],
        outputs["logs_p"],
        outputs["y_mask"],
    )
    loss_duration = outputs["duration_loss"]

    total_g = (
        float(loss_weights.get("generator_adv", 1.0)) * loss_gen_adv
        + float(loss_weights.get("feature_matching", 2.0)) * loss_fm
        + float(loss_weights.get("mel", 45.0)) * loss_mel
        + float(loss_weights.get("kl", 1.0)) * loss_kl
        + float(loss_weights.get("duration", 1.0)) * loss_duration
    )
    total_g.backward()
    if grad_clip and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
    optimizer_g.step()

    return {
        "loss_d": float(loss_disc.detach().cpu()),
        "loss_g": float(total_g.detach().cpu()),
        "loss_gen_adv": float(loss_gen_adv.detach().cpu()),
        "loss_feature_matching": float(loss_fm.detach().cpu()),
        "loss_mel": float(loss_mel.detach().cpu()),
        "loss_kl": float(loss_kl.detach().cpu()),
        "loss_duration": float(loss_duration.detach().cpu()),
    }


@torch.no_grad()
def write_validation_samples(
    generator: ToneConditionedVitsGenerator,
    batch: dict[str, Any],
    *,
    output_dir: str | Path,
    step: int,
    sampling_rate: int,
    max_samples: int = 2,
) -> list[Path]:
    generator.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = generator.infer(
        text_ids=batch["text_ids"][:max_samples],
        text_lengths=batch["text_lengths"][:max_samples],
        tone_ids=batch["tone_ids"][:max_samples],
    )
    waveforms = outputs["waveform"].squeeze(1).detach().cpu()

    written = []
    for idx, wav in enumerate(waveforms):
        sample_id = batch["sample_ids"][idx] if "sample_ids" in batch else f"sample_{idx:02d}"
        path = output_dir / f"step_{step:08d}_{sample_id}.wav"
        sf.write(path, wav.numpy(), sampling_rate)
        written.append(path)

    return written
