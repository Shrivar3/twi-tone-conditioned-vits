from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    *,
    step: int,
    epoch: int,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    config: dict[str, Any],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "epoch": int(epoch),
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "config": config,
        },
        path,
    )
    return path


def load_checkpoint(
    path: str | Path,
    *,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer | None = None,
    optimizer_d: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    generator.load_state_dict(payload["generator"])
    discriminator.load_state_dict(payload["discriminator"])

    if optimizer_g is not None and "optimizer_g" in payload:
        optimizer_g.load_state_dict(payload["optimizer_g"])
    if optimizer_d is not None and "optimizer_d" in payload:
        optimizer_d.load_state_dict(payload["optimizer_d"])

    return payload
