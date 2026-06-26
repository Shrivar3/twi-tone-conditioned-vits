# src/modelling/tone_conditioned_vits.py
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from transformers import VitsModel
from transformers.models.vits.modeling_vits import (
    VitsModelOutput,
    VitsTextEncoderOutput,
)

from src.tone.tone_vocab import ToneVocab


class ToneConditionedTextEncoder(nn.Module):
    """Wrap a HF VITS text encoder with additive tone embeddings.

    Baseline VITS:
        hidden = char_embedding(input_ids)

    Tone-conditioned VITS:
        hidden = char_embedding(input_ids) + tone_embedding(tone_ids)

    This keeps the original Farmerline checkpoint weights intact and only adds
    a small new embedding table that can be trained during finetuning.
    """

    def __init__(
        self,
        base_text_encoder: nn.Module,
        *,
        num_tones: int,
        tone_embedding_scale: float = 1.0,
        tone_padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.base_text_encoder = base_text_encoder
        self.config = base_text_encoder.config
        self.tone_embedding = nn.Embedding(
            num_embeddings=num_tones,
            embedding_dim=self.config.hidden_size,
            padding_idx=tone_padding_idx,
        )
        self.tone_embedding_scale = float(tone_embedding_scale)

    def get_input_embeddings(self):
        return self.base_text_encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.base_text_encoder.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.FloatTensor,
        tone_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], VitsTextEncoderOutput]:
        if tone_ids is None:
            tone_ids = torch.zeros_like(input_ids)

        if tone_ids.shape != input_ids.shape:
            raise ValueError(
                f"tone_ids shape {tuple(tone_ids.shape)} must match "
                f"input_ids shape {tuple(input_ids.shape)}"
            )

        base = self.base_text_encoder

        hidden_states = base.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)
        hidden_states = hidden_states + self.tone_embedding(tone_ids) * self.tone_embedding_scale

        encoder_outputs = base.encoder(
            hidden_states=hidden_states,
            padding_mask=padding_mask,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = (
            encoder_outputs[0] if not return_dict else encoder_outputs.last_hidden_state
        )

        stats = base.project(last_hidden_state.transpose(1, 2)).transpose(1, 2)
        stats = stats * padding_mask

        prior_means, prior_log_variances = torch.split(
            stats,
            self.config.flow_size,
            dim=2,
        )

        if not return_dict:
            return (last_hidden_state, prior_means, prior_log_variances) + encoder_outputs[1:]

        return VitsTextEncoderOutput(
            last_hidden_state=last_hidden_state,
            prior_means=prior_means,
            prior_log_variances=prior_log_variances,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ToneConditionedVitsModel(nn.Module):
    """Inference-compatible wrapper around HF VitsModel with tone_ids support.

    This wrapper is intentionally small. It is meant to prove the architecture,
    produce smoke-test audio, and prepare for finetuning.

    Note: HF Transformers VITS currently does not implement the full training
    loss path. For real finetuning, use this module as the architectural target
    and implement/adapt a VITS training loop.
    """

    def __init__(
        self,
        base_model: VitsModel,
        *,
        tone_vocab: ToneVocab | None = None,
        tone_embedding_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.vits = base_model
        self.tone_vocab = tone_vocab or ToneVocab()

        self.vits.text_encoder = ToneConditionedTextEncoder(
            self.vits.text_encoder,
            num_tones=len(self.tone_vocab.labels),
            tone_embedding_scale=tone_embedding_scale,
            tone_padding_idx=self.tone_vocab.pad_id,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        *,
        tone_vocab: ToneVocab | None = None,
        tone_embedding_scale: float = 1.0,
        **kwargs: Any,
    ) -> "ToneConditionedVitsModel":
        base_model = VitsModel.from_pretrained(model_id_or_path, **kwargs)
        return cls(
            base_model,
            tone_vocab=tone_vocab,
            tone_embedding_scale=tone_embedding_scale,
        )

    @property
    def config(self):
        return self.vits.config

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        tone_ids: Optional[torch.Tensor] = None,
        speaker_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[Any], VitsModelOutput]:
        if labels is not None:
            raise NotImplementedError(
                "This wrapper currently supports inference/smoke tests. "
                "Training needs a VITS training loop with spectrogram, duration, "
                "flow, KL, reconstruction, and adversarial losses."
            )

        if input_ids is None:
            raise ValueError("input_ids is required.")

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None:
            input_padding_mask = attention_mask.unsqueeze(-1).float()
        else:
            input_padding_mask = torch.ones_like(input_ids).unsqueeze(-1).float()

        if tone_ids is None:
            tone_ids = torch.zeros_like(input_ids)

        if self.config.num_speakers > 1 and speaker_id is not None:
            if not 0 <= speaker_id < self.config.num_speakers:
                raise ValueError(
                    f"Set speaker_id in the range 0-{self.config.num_speakers - 1}."
                )

            if isinstance(speaker_id, int):
                speaker_id = torch.full(
                    size=(1,),
                    fill_value=speaker_id,
                    device=input_ids.device,
                )

            speaker_embeddings = self.vits.embed_speaker(speaker_id).unsqueeze(-1)
        else:
            speaker_embeddings = None

        text_encoder_output = self.vits.text_encoder(
            input_ids=input_ids,
            padding_mask=input_padding_mask,
            tone_ids=tone_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = (
            text_encoder_output[0]
            if not return_dict
            else text_encoder_output.last_hidden_state
        )
        hidden_states = hidden_states.transpose(1, 2)

        input_padding_mask = input_padding_mask.transpose(1, 2)

        prior_means = (
            text_encoder_output[1]
            if not return_dict
            else text_encoder_output.prior_means
        )
        prior_log_variances = (
            text_encoder_output[2]
            if not return_dict
            else text_encoder_output.prior_log_variances
        )

        if self.config.use_stochastic_duration_prediction:
            log_duration = self.vits.duration_predictor(
                hidden_states,
                input_padding_mask,
                speaker_embeddings,
                reverse=True,
                noise_scale=self.vits.noise_scale_duration,
            )
        else:
            log_duration = self.vits.duration_predictor(
                hidden_states,
                input_padding_mask,
                speaker_embeddings,
            )

        length_scale = 1.0 / self.vits.speaking_rate
        duration = torch.ceil(torch.exp(log_duration) * input_padding_mask * length_scale)

        predicted_lengths = torch.clamp_min(torch.sum(duration, [1, 2]), 1).long()

        indices = torch.arange(
            predicted_lengths.max(),
            dtype=predicted_lengths.dtype,
            device=predicted_lengths.device,
        )
        output_padding_mask = indices.unsqueeze(0) < predicted_lengths.unsqueeze(1)
        output_padding_mask = output_padding_mask.unsqueeze(1).to(input_padding_mask.dtype)

        attn_mask = torch.unsqueeze(input_padding_mask, 2) * torch.unsqueeze(
            output_padding_mask,
            -1,
        )

        batch_size, _, output_length, input_length = attn_mask.shape

        cum_duration = torch.cumsum(duration, -1).view(batch_size * input_length, 1)
        indices = torch.arange(
            output_length,
            dtype=duration.dtype,
            device=duration.device,
        )

        valid_indices = indices.unsqueeze(0) < cum_duration
        valid_indices = valid_indices.to(attn_mask.dtype).view(
            batch_size,
            input_length,
            output_length,
        )

        padded_indices = valid_indices - torch.nn.functional.pad(
            valid_indices,
            [0, 0, 1, 0, 0, 0],
        )[:, :-1]

        attn = padded_indices.unsqueeze(1).transpose(2, 3) * attn_mask

        prior_means = torch.matmul(attn.squeeze(1), prior_means).transpose(1, 2)
        prior_log_variances = torch.matmul(
            attn.squeeze(1),
            prior_log_variances,
        ).transpose(1, 2)

        prior_latents = (
            prior_means
            + torch.randn_like(prior_means)
            * torch.exp(prior_log_variances)
            * self.vits.noise_scale
        )

        latents = self.vits.flow(
            prior_latents,
            output_padding_mask,
            speaker_embeddings,
            reverse=True,
        )

        spectrogram = latents * output_padding_mask
        waveform = self.vits.decoder(spectrogram, speaker_embeddings)
        waveform = waveform.squeeze(1)

        sequence_lengths = predicted_lengths * np.prod(self.config.upsample_rates)

        if not return_dict:
            outputs = (waveform, sequence_lengths, spectrogram) + text_encoder_output[3:]
            return outputs

        return VitsModelOutput(
            waveform=waveform,
            sequence_lengths=sequence_lengths,
            spectrogram=spectrogram,
            hidden_states=text_encoder_output.hidden_states,
            attentions=text_encoder_output.attentions,
        )

    def save_tone_metadata(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "tone_labels": list(self.tone_vocab.labels),
            "tone_embedding_scale": self.vits.text_encoder.tone_embedding_scale,
            "base_model_type": "VitsModel",
        }

        with (output_dir / "tone_conditioning_config.json").open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
