from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    if max_len is None:
        max_len = int(lengths.max().item())
    ids = torch.arange(max_len, device=lengths.device)
    return ids.unsqueeze(0) < lengths.unsqueeze(1)


def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int) -> torch.Tensor:
    ret = torch.zeros(
        x.size(0),
        x.size(1),
        segment_size,
        dtype=x.dtype,
        device=x.device,
    )
    for i, idx in enumerate(ids_str.tolist()):
        idx = int(idx)
        segment = x[i, :, idx : idx + segment_size]
        ret[i, :, : segment.size(-1)] = segment
    return ret


def rand_slice_segments(
    x: torch.Tensor,
    x_lengths: torch.Tensor,
    segment_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids_str_max = torch.clamp(x_lengths - segment_size, min=0)
    ids_str = torch.zeros_like(x_lengths)
    for i, max_start in enumerate(ids_str_max.tolist()):
        if int(max_start) > 0:
            ids_str[i] = torch.randint(
                low=0,
                high=int(max_start) + 1,
                size=(1,),
                device=x.device,
            )
    return slice_segments(x, ids_str, segment_size), ids_str


def slice_waveform_segments(
    waveform: torch.Tensor,
    ids_str: torch.Tensor,
    segment_size: int,
    hop_length: int,
) -> torch.Tensor:
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(1)
    starts = ids_str * int(hop_length)
    return slice_segments(waveform, starts, segment_size)


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    duration = duration.long()
    b, _, t_y, t_x = mask.shape
    path = torch.zeros_like(mask)
    cum_duration = torch.cumsum(duration, dim=2)

    for batch_idx in range(b):
        prev = 0
        for text_idx in range(t_x):
            end = int(cum_duration[batch_idx, 0, text_idx].item())
            end = min(end, t_y)
            if end > prev:
                path[batch_idx, 0, prev:end, text_idx] = 1.0
            prev = end

    return path * mask


def maximum_path(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch monotonic alignment search.

    The original VITS repo uses a Cython MAS extension. This implementation is
    slower but keeps smoke/debug runs dependency-light and preserves the same
    monotonic dynamic-programming objective.
    """
    if value.ndim != 3:
        raise ValueError(f"value must be [batch, spec_time, text_time], got {value.shape}")
    if mask.shape != value.shape:
        raise ValueError(f"mask shape {mask.shape} must match value shape {value.shape}")

    b, t_y, t_x = value.shape
    path = torch.zeros_like(value)
    neg_inf = -1e9

    with torch.no_grad():
        for batch_idx in range(b):
            y_len = int(mask[batch_idx].any(dim=1).sum().item())
            x_len = int(mask[batch_idx].any(dim=0).sum().item())
            if y_len <= 0 or x_len <= 0:
                continue

            scores = value[batch_idx, :y_len, :x_len]
            dp = scores.new_full((y_len, x_len), neg_inf)
            back = torch.zeros((y_len, x_len), dtype=torch.bool, device=value.device)
            dp[0, 0] = scores[0, 0]

            for y_idx in range(1, y_len):
                max_x = min(y_idx, x_len - 1)
                for x_idx in range(max_x + 1):
                    stay = dp[y_idx - 1, x_idx]
                    advance = dp[y_idx - 1, x_idx - 1] if x_idx > 0 else scores.new_tensor(neg_inf)
                    if advance > stay:
                        dp[y_idx, x_idx] = advance + scores[y_idx, x_idx]
                        back[y_idx, x_idx] = True
                    else:
                        dp[y_idx, x_idx] = stay + scores[y_idx, x_idx]

            x_idx = x_len - 1
            for y_idx in range(y_len - 1, -1, -1):
                path[batch_idx, y_idx, x_idx] = 1.0
                if y_idx > 0 and back[y_idx, x_idx]:
                    x_idx = max(0, x_idx - 1)

    return path * mask


def crop_to_match(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(a.size(-1), b.size(-1))
    return a[..., :n], b[..., :n]


def pad_1d(values: list[torch.Tensor], *, pad_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(v.numel()) for v in values)
    out = values[0].new_full((len(values), max_len), pad_value)
    lengths = torch.zeros(len(values), dtype=torch.long)
    for i, value in enumerate(values):
        n = int(value.numel())
        out[i, :n] = value
        lengths[i] = n
    return out, lengths


def pad_2d(values: list[torch.Tensor], *, pad_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    channels = int(values[0].shape[0])
    max_len = max(int(v.shape[1]) for v in values)
    out = values[0].new_full((len(values), channels, max_len), pad_value)
    lengths = torch.zeros(len(values), dtype=torch.long)
    for i, value in enumerate(values):
        n = int(value.shape[1])
        out[i, :, :n] = value
        lengths[i] = n
    return out, lengths


def pad_or_crop_last_dim(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.size(-1) == target_len:
        return x
    if x.size(-1) > target_len:
        return x[..., :target_len]
    return F.pad(x, (0, target_len - x.size(-1)))
