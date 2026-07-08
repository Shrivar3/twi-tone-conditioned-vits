from __future__ import annotations

# Architecture follows the MIT-licensed VITS training structure by Jaehyeon Kim:
# text prior, posterior encoder, residual coupling flow, duration predictor,
# HiFi-GAN-style decoder, and multi-period discriminator.

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from src.training.full_vits.commons import (
    generate_path,
    maximum_path,
    rand_slice_segments,
    sequence_mask,
)
from src.training.full_vits.config import ModelConfig


LRELU_SLOPE = 0.1


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d)):
        nn.init.normal_(module.weight, 0.0, 0.01)


class ConvStack(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        *,
        kernel_size: int,
        n_layers: int,
        p_dropout: float,
    ) -> None:
        super().__init__()
        layers = []
        in_channels = channels
        for _ in range(n_layers):
            layers.append(
                weight_norm(
                    nn.Conv1d(
                        in_channels,
                        hidden_channels,
                        kernel_size,
                        padding=get_padding(kernel_size),
                    )
                )
            )
            layers.append(nn.LeakyReLU(LRELU_SLOPE))
            layers.append(nn.Dropout(p_dropout))
            in_channels = hidden_channels
        self.layers = nn.ModuleList(layers)
        self.proj = weight_norm(nn.Conv1d(hidden_channels, channels, 1))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        residual = x
        for layer in self.layers:
            if isinstance(layer, nn.Conv1d):
                x = layer(x * x_mask)
            else:
                x = layer(x)
        return (self.proj(x * x_mask) + residual) * x_mask


class ToneAwareTextEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tone_conditioning_mode = config.tone_conditioning_mode
        self.use_tone_conditioning = bool(config.use_tone_conditioning)
        self.tone_embedding_scale = float(config.tone_embedding_scale)

        self.text_embedding = nn.Embedding(config.n_vocab, config.hidden_channels)
        self.tone_embedding = nn.Embedding(config.num_tones, config.hidden_channels, padding_idx=0)

        nn.init.normal_(self.text_embedding.weight, 0.0, config.hidden_channels**-0.5)
        nn.init.normal_(self.tone_embedding.weight, 0.0, config.hidden_channels**-0.5)

        if self.tone_conditioning_mode == "concat_projection":
            self.tone_projection = nn.Linear(config.hidden_channels * 2, config.hidden_channels)
        elif self.tone_conditioning_mode in {"none", "additive"}:
            self.tone_projection = None
        else:
            raise ValueError(
                "tone_conditioning_mode must be one of: none, additive, concat_projection"
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_channels,
            nhead=config.n_heads,
            dim_feedforward=config.filter_channels,
            dropout=config.p_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.proj = nn.Conv1d(config.hidden_channels, config.inter_channels * 2, 1)

    def _condition_embeddings(
        self,
        text_hidden: torch.Tensor,
        tone_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            not self.use_tone_conditioning
            or self.tone_conditioning_mode == "none"
            or tone_ids is None
        ):
            return text_hidden

        tone_hidden = self.tone_embedding(tone_ids) * self.tone_embedding_scale
        if self.tone_conditioning_mode == "additive":
            return text_hidden + tone_hidden

        return self.tone_projection(torch.cat([text_hidden, tone_hidden], dim=-1))

    def forward(
        self,
        text_ids: torch.Tensor,
        text_lengths: torch.Tensor,
        tone_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.text_embedding(text_ids) * math.sqrt(self.config.hidden_channels)
        hidden = self._condition_embeddings(hidden, tone_ids)

        padding_mask = ~sequence_mask(text_lengths, text_ids.size(1))
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        hidden = hidden.transpose(1, 2)

        x_mask = sequence_mask(text_lengths, text_ids.size(1)).unsqueeze(1).to(hidden.dtype)
        stats = self.proj(hidden * x_mask) * x_mask
        m, logs = torch.split(stats, self.config.inter_channels, dim=1)
        return hidden * x_mask, m, logs, x_mask


class PosteriorEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.pre = nn.Conv1d(config.spec_channels, config.hidden_channels, 1)
        self.enc = ConvStack(
            config.hidden_channels,
            config.hidden_channels,
            kernel_size=5,
            n_layers=8,
            p_dropout=config.p_dropout,
        )
        self.proj = nn.Conv1d(config.hidden_channels, config.inter_channels * 2, 1)
        self.inter_channels = config.inter_channels

    def forward(
        self,
        spectrogram: torch.Tensor,
        spec_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y_mask = sequence_mask(spec_lengths, spectrogram.size(2)).unsqueeze(1).to(spectrogram.dtype)
        hidden = self.pre(spectrogram) * y_mask
        hidden = self.enc(hidden, y_mask)
        stats = self.proj(hidden) * y_mask
        m, logs = torch.split(stats, self.inter_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * y_mask
        return z, m, logs, y_mask


class DurationPredictor(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.drop = nn.Dropout(config.p_dropout)
        self.conv_1 = nn.Conv1d(
            config.hidden_channels,
            config.hidden_channels,
            config.kernel_size,
            padding=get_padding(config.kernel_size),
        )
        self.norm_1 = nn.GroupNorm(1, config.hidden_channels)
        self.conv_2 = nn.Conv1d(
            config.hidden_channels,
            config.hidden_channels,
            config.kernel_size,
            padding=get_padding(config.kernel_size),
        )
        self.norm_2 = nn.GroupNorm(1, config.hidden_channels)
        self.proj = nn.Conv1d(config.hidden_channels, 1, 1)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = x.detach()
        x = self.conv_1(x * x_mask)
        x = self.norm_1(F.relu(x))
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = self.norm_2(F.relu(x))
        x = self.drop(x)
        return self.proj(x * x_mask) * x_mask


class ResidualCouplingLayer(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, kernel_size: int, n_layers: int) -> None:
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("ResidualCouplingLayer requires an even channel count.")

        self.half_channels = channels // 2
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = ConvStack(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            n_layers=n_layers,
            p_dropout=0.0,
        )
        self.proj = nn.Conv1d(hidden_channels, self.half_channels * 2, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        *,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        x0, x1 = torch.split(x, [self.half_channels, self.half_channels], dim=1)
        hidden = self.pre(x0) * x_mask
        hidden = self.enc(hidden, x_mask)
        stats = self.proj(hidden) * x_mask
        m, logs = torch.split(stats, [self.half_channels, self.half_channels], dim=1)
        logs = torch.tanh(logs)

        if not reverse:
            x1 = (m + x1 * torch.exp(logs)) * x_mask
            x = torch.cat([x0, x1], dim=1)
            logdet = torch.sum(logs * x_mask, dim=[1, 2])
            return x, logdet

        x1 = ((x1 - m) * torch.exp(-logs)) * x_mask
        return torch.cat([x0, x1], dim=1)


class Flip(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        *,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        x = torch.flip(x, dims=[1]) * x_mask
        if reverse:
            return x
        return x, torch.zeros(x.size(0), dtype=x.dtype, device=x.device)


class ResidualCouplingBlock(nn.Module):
    def __init__(self, config: ModelConfig, n_flows: int = 4) -> None:
        super().__init__()
        flows: list[nn.Module] = []
        for _ in range(n_flows):
            flows.append(
                ResidualCouplingLayer(
                    config.inter_channels,
                    config.hidden_channels,
                    kernel_size=5,
                    n_layers=4,
                )
            )
            flows.append(Flip())
        self.flows = nn.ModuleList(flows)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        *,
        reverse: bool = False,
    ) -> torch.Tensor:
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, reverse=False)
            return x

        for flow in reversed(self.flows):
            x = flow(x, x_mask, reverse=True)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: list[int]) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=dilation,
                        padding=get_padding(kernel_size, dilation),
                    )
                )
                for dilation in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            residual = x
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = conv(x)
            x = x + residual
        return x

    def remove_weight_norm(self) -> None:
        for conv in self.convs:
            remove_weight_norm(conv)


class HifiGanGenerator(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = weight_norm(
            nn.Conv1d(config.inter_channels, config.upsample_initial_channel, 7, padding=3)
        )

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for i, (rate, kernel_size) in enumerate(
            zip(config.upsample_rates, config.upsample_kernel_sizes)
        ):
            in_channels = config.upsample_initial_channel // (2**i)
            out_channels = config.upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        rate,
                        padding=(kernel_size - rate) // 2,
                    )
                )
            )
            for res_kernel, res_dilations in zip(
                config.resblock_kernel_sizes,
                config.resblock_dilation_sizes,
            ):
                self.resblocks.append(ResBlock(out_channels, res_kernel, res_dilations))

        final_channels = config.upsample_initial_channel // (2**self.num_upsamples)
        self.conv_post = weight_norm(nn.Conv1d(final_channels, 1, 7, padding=3, bias=False))
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pre(x)
        for i, upsample in enumerate(self.ups):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = upsample(x)
            xs = None
            for j in range(self.num_kernels):
                res = self.resblocks[i * self.num_kernels + j](x)
                xs = res if xs is None else xs + res
            x = xs / self.num_kernels
        x = F.leaky_relu(x, LRELU_SLOPE)
        return torch.tanh(self.conv_post(x))

    def remove_weight_norm(self) -> None:
        remove_weight_norm(self.conv_pre)
        for upsample in self.ups:
            remove_weight_norm(upsample)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_post)


class ToneConditionedVitsGenerator(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.enc_p = ToneAwareTextEncoder(config)
        self.enc_q = PosteriorEncoder(config)
        self.flow = ResidualCouplingBlock(config)
        self.dp = DurationPredictor(config)
        self.dec = HifiGanGenerator(config)

    def forward(
        self,
        text_ids: torch.Tensor,
        text_lengths: torch.Tensor,
        tone_ids: torch.Tensor,
        spectrogram: torch.Tensor,
        spec_lengths: torch.Tensor,
    ) -> dict[str, Any]:
        x, m_p, logs_p, x_mask = self.enc_p(text_ids, text_lengths, tone_ids)
        z, m_q, logs_q, y_mask = self.enc_q(spectrogram, spec_lengths)
        z_p = self.flow(z, y_mask)

        with torch.no_grad():
            s_p_sq_r = torch.exp(-2.0 * logs_p)
            neg_cent1 = torch.sum(
                -0.5 * math.log(2.0 * math.pi) - logs_p,
                dim=1,
                keepdim=True,
            )
            neg_cent2 = torch.matmul(-0.5 * (z_p**2).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), m_p * s_p_sq_r)
            neg_cent4 = torch.sum(-0.5 * (m_p**2) * s_p_sq_r, dim=1, keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
            attn = maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

        w = attn.sum(dim=2)
        logw_target = torch.log(w + 1e-6) * x_mask
        logw = self.dp(x, x_mask)
        duration_loss = torch.sum((logw - logw_target) ** 2 * x_mask) / torch.sum(
            x_mask
        ).clamp_min(1.0)

        m_p_aligned = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p_aligned = torch.matmul(
            attn.squeeze(1),
            logs_p.transpose(1, 2),
        ).transpose(1, 2)

        z_slice, ids_slice = rand_slice_segments(
            z,
            spec_lengths,
            self.config.spec_segment_size,
        )
        y_hat = self.dec(z_slice)

        return {
            "y_hat": y_hat,
            "duration_loss": duration_loss,
            "attn": attn,
            "ids_slice": ids_slice,
            "x_mask": x_mask,
            "y_mask": y_mask,
            "z": z,
            "z_p": z_p,
            "m_p": m_p_aligned,
            "logs_p": logs_p_aligned,
            "m_q": m_q,
            "logs_q": logs_q,
        }

    @torch.no_grad()
    def infer(
        self,
        text_ids: torch.Tensor,
        text_lengths: torch.Tensor,
        tone_ids: torch.Tensor,
        *,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        max_len: int | None = None,
    ) -> dict[str, Any]:
        x, m_p, logs_p, x_mask = self.enc_p(text_ids, text_lengths, tone_ids)
        logw = self.dp(x, x_mask)
        w = torch.exp(logw) * x_mask * float(length_scale)
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, dim=[1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, int(y_lengths.max().item())).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(2) * y_mask.unsqueeze(-1)
        attn = generate_path(w_ceil, attn_mask)
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * float(noise_scale)
        z = self.flow(z_p, y_mask, reverse=True)
        waveform = self.dec((z * y_mask)[:, :, :max_len])
        return {
            "waveform": waveform,
            "attn": attn,
            "y_mask": y_mask,
            "z": z,
            "z_p": z_p,
            "m_p": m_p,
            "logs_p": logs_p,
        }


class DiscriminatorP(nn.Module):
    def __init__(
        self,
        period: int,
        *,
        base_channels: int = 16,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.period = period
        norm = spectral_norm if use_spectral_norm else weight_norm
        channels = [
            base_channels,
            base_channels * 4,
            base_channels * 16,
            base_channels * 32,
            base_channels * 32,
        ]
        self.convs = nn.ModuleList()
        in_channels = 1
        for idx, out_channels in enumerate(channels):
            layer_stride = stride if idx < len(channels) - 1 else 1
            self.convs.append(
                norm(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        (kernel_size, 1),
                        (layer_stride, 1),
                        padding=(get_padding(kernel_size), 0),
                    )
                )
            )
            in_channels = out_channels
        self.conv_post = norm(nn.Conv2d(in_channels, 1, (3, 1), padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), mode="constant")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class DiscriminatorS(nn.Module):
    def __init__(self, *, base_channels: int = 16, use_spectral_norm: bool = False) -> None:
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        channels = [
            base_channels,
            base_channels * 4,
            base_channels * 16,
            base_channels * 32,
            base_channels * 32,
            base_channels * 32,
        ]
        kernels = [15, 41, 41, 41, 5, 3]
        strides = [1, 4, 4, 4, 1, 1]

        self.convs = nn.ModuleList()
        in_channels = 1
        for out_channels, kernel_size, stride in zip(channels, kernels, strides):
            self.convs.append(
                norm(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=get_padding(kernel_size),
                    )
                )
            )
            in_channels = out_channels
        self.conv_post = norm(nn.Conv1d(in_channels, 1, 3, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(
        self,
        *,
        base_channels: int = 16,
        periods: list[int] | None = None,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        periods = periods or [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList(
            [DiscriminatorS(base_channels=base_channels, use_spectral_norm=use_spectral_norm)]
            + [
                DiscriminatorP(
                    period,
                    base_channels=base_channels,
                    use_spectral_norm=use_spectral_norm,
                )
                for period in periods
            ]
        )

    @classmethod
    def from_model_config(cls, config: ModelConfig) -> "MultiPeriodDiscriminator":
        return cls(
            base_channels=config.discriminator_base_channels,
            periods=config.discriminator_periods,
        )

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for discriminator in self.discriminators:
            y_d_r, fmap_r = discriminator(y)
            y_d_g, fmap_g = discriminator(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
