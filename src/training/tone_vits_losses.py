from __future__ import annotations

import torch
import torch.nn.functional as F


def crop_or_pad_waveform(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Make prediction and target waveforms the same length.

    This is only for the debug/proxy loop. It is not a substitute for the full
    VITS reconstruction, KL, duration, flow, and adversarial training losses.
    """
    if prediction.ndim != 2:
        raise ValueError(f"prediction must be [batch, time], got {prediction.shape}")
    if target.ndim != 2:
        raise ValueError(f"target must be [batch, time], got {target.shape}")

    pred_len = prediction.shape[1]
    target_len = target.shape[1]
    out_len = max(pred_len, target_len)

    if pred_len < out_len:
        prediction = F.pad(prediction, (0, out_len - pred_len))
    if target_len < out_len:
        target = F.pad(target, (0, out_len - target_len))

    if target_mask is not None:
        if target_mask.ndim != 2:
            raise ValueError(f"target_mask must be [batch, time], got {target_mask.shape}")
        if target_mask.shape[1] < out_len:
            target_mask = F.pad(target_mask, (0, out_len - target_mask.shape[1]))
        elif target_mask.shape[1] > out_len:
            target_mask = target_mask[:, :out_len]

    return prediction, target, target_mask


def masked_waveform_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Simple waveform MSE for a debug/proxy overfit loop.

    This lets us test whether gradients can flow through the tone-conditioned
    inference path. It is deliberately labelled as a proxy loss because proper
    VITS fine-tuning needs the original VITS training objectives.
    """
    prediction, target, target_mask = crop_or_pad_waveform(
        prediction,
        target,
        target_mask,
    )

    loss = (prediction - target) ** 2

    if target_mask is None:
        return loss.mean()

    target_mask = target_mask.to(loss.dtype)
    denom = target_mask.sum().clamp_min(1.0)
    return (loss * target_mask).sum() / denom
