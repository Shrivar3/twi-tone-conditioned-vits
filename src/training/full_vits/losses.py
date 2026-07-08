from __future__ import annotations

# Adapted from the MIT-licensed VITS losses by Jaehyeon Kim:
# https://github.com/jaywalnut310/vits/blob/main/losses.py

import torch


def feature_loss(fmap_real, fmap_generated) -> torch.Tensor:
    loss = 0.0
    for disc_real, disc_generated in zip(fmap_real, fmap_generated):
        for real_layer, generated_layer in zip(disc_real, disc_generated):
            loss = loss + torch.mean(
                torch.abs(real_layer.float().detach() - generated_layer.float())
            )
    return loss * 2.0


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0.0
    real_losses = []
    generated_losses = []

    for disc_real, disc_generated in zip(disc_real_outputs, disc_generated_outputs):
        disc_real = disc_real.float()
        disc_generated = disc_generated.float()
        real_loss = torch.mean((1.0 - disc_real) ** 2)
        generated_loss = torch.mean(disc_generated**2)
        loss = loss + real_loss + generated_loss
        real_losses.append(real_loss)
        generated_losses.append(generated_loss)

    return loss, real_losses, generated_losses


def generator_loss(disc_outputs):
    loss = 0.0
    generated_losses = []

    for disc_generated in disc_outputs:
        disc_generated = disc_generated.float()
        item_loss = torch.mean((1.0 - disc_generated) ** 2)
        generated_losses.append(item_loss)
        loss = loss + item_loss

    return loss, generated_losses


def kl_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor,
) -> torch.Tensor:
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl = kl + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    return kl / torch.sum(z_mask).clamp_min(1.0)
