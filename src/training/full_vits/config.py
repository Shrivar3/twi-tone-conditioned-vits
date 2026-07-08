from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AudioConfig:
    sampling_rate: int = 22050
    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mel_channels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float | None = None


@dataclass
class ModelConfig:
    n_vocab: int = 256
    num_tones: int = 11
    spec_channels: int = 513
    segment_size: int = 8192
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.1
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list[list[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    upsample_rates: list[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: list[int] = field(default_factory=lambda: [16, 16, 4, 4])
    tone_conditioning_mode: str = "concat_projection"
    tone_embedding_scale: float = 1.0
    use_tone_conditioning: bool = True
    discriminator_base_channels: int = 16
    discriminator_periods: list[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])

    @property
    def spec_segment_size(self) -> int:
        return max(1, int(self.segment_size) // max(1, int(self.upsample_factor)))

    @property
    def upsample_factor(self) -> int:
        factor = 1
        for rate in self.upsample_rates:
            factor *= int(rate)
        return factor


@dataclass
class TrainingConfig:
    batch_size: int = 2
    num_workers: int = 0
    max_steps: int = 1000
    learning_rate: float = 2e-4
    betas: tuple[float, float] = (0.8, 0.99)
    eps: float = 1e-9
    grad_clip: float | None = 5.0
    save_every: int = 1000
    validate_every: int = 1000
    log_every: int = 10
    seed: int = 42
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "mel": 45.0,
            "kl": 1.0,
            "duration": 1.0,
            "feature_matching": 2.0,
            "generator_adv": 1.0,
        }
    )


def _filter_dataclass_kwargs(cls: type, values: dict[str, Any] | None) -> dict[str, Any]:
    values = values or {}
    names = set(cls.__dataclass_fields__.keys())
    return {key: value for key, value in values.items() if key in names}


def audio_config_from_dict(values: dict[str, Any] | None) -> AudioConfig:
    return AudioConfig(**_filter_dataclass_kwargs(AudioConfig, values))


def model_config_from_dict(values: dict[str, Any] | None) -> ModelConfig:
    return ModelConfig(**_filter_dataclass_kwargs(ModelConfig, values))


def training_config_from_dict(values: dict[str, Any] | None) -> TrainingConfig:
    kwargs = _filter_dataclass_kwargs(TrainingConfig, values)
    if "betas" in kwargs:
        kwargs["betas"] = tuple(kwargs["betas"])
    return TrainingConfig(**kwargs)
