# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trained noise-conditioned behavior classifier ``C(x, t)``.

Drop-in replacement for :class:`HeuristicBehaviorClassifier`: same
``forward(x, t) -> (B, K)`` signature, differentiable with respect to ``x``,
so it plugs into :func:`classifier_gradient_guidance_fn` unchanged.

Architecture: small 1D-conv stack with FiLM-style injection of a sinusoidal
time embedding. ~200 k params at the default config — trains in minutes on
a single GPU from the cached ``(action_x1, label)`` dataset.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Transformer-style sinusoidal embedding on a scalar ``t`` per sample."""
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")
    t = t.reshape(-1).float()
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(0.0, math.log(10_000.0), half, device=t.device) * -1.0
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0) * (2 * math.pi)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class _FiLMConvBlock(nn.Module):
    """GroupNorm → FiLM(γ, β) → Conv1d → GELU → Conv1d, residual.

    Operates on tensors shaped ``(B, C, L)``.
    """

    def __init__(self, channels: int, time_dim: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        groups = 8 if channels % 8 == 0 else 1
        self.norm = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad)
        self.act = nn.GELU()
        self.film = nn.Linear(time_dim, 2 * channels)

    def forward(self, h: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        residual = h
        h = self.norm(h)
        gamma_beta = self.film(phi)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        h = gamma.unsqueeze(-1) * h + beta.unsqueeze(-1)
        h = self.conv1(h)
        h = self.act(h)
        h = self.conv2(h)
        return residual + h


class TrainedBehaviorClassifier(nn.Module):
    """Noise-conditioned classifier over normalized action tensors.

    Shape contract (same as :class:`HeuristicBehaviorClassifier`):

    * ``x``: ``(B, T, 2)`` normalized ``[accel, κ]``.
    * ``t``: ``(B, ...)`` noise level in ``[0, 1]`` — any broadcastable shape
      is reduced to one scalar per batch row. May also be ``None`` (treated
      as ``t = 1``).
    * Returns ``(B, num_classes)`` logits.
    """

    def __init__(
        self,
        num_classes: int = 5,
        *,
        d_model: int = 96,
        depth: int = 3,
        kernel: int = 3,
        time_dim: int = 96,
        n_waypoints: int = 64,
        input_channels: int = 2,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.d_model = int(d_model)
        self.depth = int(depth)
        self.kernel = int(kernel)
        self.time_dim = int(time_dim)
        self.n_waypoints = int(n_waypoints)
        self.input_channels = int(input_channels)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_proj = nn.Linear(input_channels, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_waypoints, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList(
            [_FiLMConvBlock(d_model, time_dim, kernel=kernel) for _ in range(depth)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.num_classes),
        )

    def _pool_t(self, t: torch.Tensor | None, batch: int, device: torch.device) -> torch.Tensor:
        if t is None:
            return torch.ones(batch, device=device, dtype=torch.float32)
        t_flat = t.reshape(batch, -1).float().mean(dim=-1)
        return t_flat.clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected x of shape (B, T, C), got {tuple(x.shape)}")
        b, tlen, c = x.shape
        if tlen != self.n_waypoints or c != self.input_channels:
            raise ValueError(
                f"expected x (B, {self.n_waypoints}, {self.input_channels}), got {tuple(x.shape)}"
            )
        t_scalar = self._pool_t(t, b, x.device)
        phi = _sinusoidal_time_embedding(t_scalar, self.time_dim)
        phi = self.time_mlp(phi)

        h = self.input_proj(x) + self.pos_emb.to(x.dtype)
        h = h.transpose(1, 2).contiguous()
        for block in self.blocks:
            h = block(h, phi)
        pooled = h.mean(dim=-1)
        logits = self.head(pooled)
        return logits

    def save(self, path: str | Path, *, extra: dict[str, Any] | None = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "state_dict": self.state_dict(),
            "config": {
                "num_classes": self.num_classes,
                "d_model": self.d_model,
                "depth": self.depth,
                "kernel": self.kernel,
                "time_dim": self.time_dim,
                "n_waypoints": self.n_waypoints,
                "input_channels": self.input_channels,
            },
        }
        if extra:
            payload["extra"] = extra
        torch.save(payload, p)

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        map_location: torch.device | str | None = "cpu",
        strict: bool = True,
    ) -> "TrainedBehaviorClassifier":
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        cfg = payload.get("config", {})
        model = cls(**cfg)
        model.load_state_dict(payload["state_dict"], strict=strict)
        return model
