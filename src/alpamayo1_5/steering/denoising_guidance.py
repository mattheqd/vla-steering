# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Denoising-time guidance for flow-matched trajectory sampling.

Implements the intervention from the AR1 midterm plan: at each Euler step augment the
velocity with a classifier gradient,

    v ← v + λ ∇_x log p_C(y | x, t),

where C is a (trainable or heuristic) behavior classifier over noisy actions x at noise
level t. See Section 5 of the project report.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import torch
from torch import nn

ScheduleName = Literal["all", "early", "late"]


class HeuristicBehaviorClassifier(nn.Module):
    """Softmax classifier over pooled normalized (accel, curvature).

    Uses fixed Gaussian prototypes in mean-action space (no learned weights). Suitable
    for debugging the guidance pipeline and for coarse yield / neutral / accelerate
    steering before a dataset-trained C is available.
    """

    num_classes: int = 3

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        """Compute class logits for each batch row.

        Args:
            x: Noisy actions ``(B, T, 2)`` (accel, curvature) in model normalization.
            t: Flow time ``(B, ...)`` broadcast like training; pooled to ``(B,)`` when given.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        _ = t  # API placeholder for noise-conditioned C; extend when training C(x, t).
        mean_a = x[..., 0].mean(dim=-1)
        mean_k = x[..., 1].mean(dim=-1)
        feat = torch.stack([mean_a, mean_k], dim=-1)
        mus = feat.new_tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        return -((feat[:, None, :] - mus[None, :, :]) ** 2).sum(dim=-1)


def classifier_gradient_guidance_fn(
    classifier: nn.Module,
    target_class: int,
    scale: float,
) -> Callable[..., torch.Tensor]:
    """Build ``denoising_guidance_fn`` for :meth:`FlowMatching.sample`.

    Returns a tensor ``delta_v`` with the same shape as ``v`` so the integrator can do
    ``v = v + delta_v`` (equivalently λ ∇_x log p(y|x) when ``scale`` absorbs λ).

    Args:
        classifier: Module ``forward(x, t) -> (B, K)`` logits (``t`` may be ignored).
        target_class: Integer class index ``y``.
        scale: Guidance strength λ (larger => stronger push toward ``y``).

    Returns:
        Callable with keyword args ``x, t, v, step_index`` returning ``delta_v``.
    """

    def denoising_guidance_fn(
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        v: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        del step_index
        with torch.enable_grad():
            xc = x.detach().float().clone().requires_grad_(True)
            logits = classifier(xc, t.detach().float())
            logp = torch.log_softmax(logits, dim=-1)[:, target_class].sum()
            grad = torch.autograd.grad(logp, xc)[0]
        return (scale * grad).to(device=v.device, dtype=v.dtype)

    return denoising_guidance_fn


def wrap_guidance_schedule(
    inner: Callable[..., torch.Tensor],
    *,
    schedule: ScheduleName | str,
    num_inference_steps: int,
) -> Callable[..., torch.Tensor]:
    """Mask ``inner`` so guidance applies only on a subset of Euler steps.

    * ``all``: same as ``inner`` (every step, indices ``0 .. num_inference_steps-1``).
    * ``early``: first 40% of steps only, i.e. ``step_index < ceil(0.4 * N)``.
    * ``late``: last 40% of steps only, i.e. ``step_index >= floor(0.6 * N)``.

    Here ``N == num_inference_steps`` (the number of Euler updates, not including t=1).
    """
    s = str(schedule).lower()
    if s == "all":
        return inner
    if s not in ("early", "late"):
        raise ValueError(f"schedule must be all|early|late, got {schedule!r}")
    n = int(num_inference_steps)
    if n <= 0:
        raise ValueError(f"num_inference_steps must be positive, got {n}")
    early_apply_until = max(0, math.ceil(0.4 * n))
    late_apply_from = min(n, math.floor(0.6 * n))

    def wrapped(
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        v: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        if s == "early" and step_index >= early_apply_until:
            return torch.zeros_like(v)
        if s == "late" and step_index < late_apply_from:
            return torch.zeros_like(v)
        return inner(x=x, t=t, v=v, step_index=step_index)

    return wrapped
