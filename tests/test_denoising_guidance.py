# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from alpamayo1_5.diffusion.flow_matching import FlowMatching
from alpamayo1_5.steering import HeuristicBehaviorClassifier, classifier_gradient_guidance_fn


def test_classifier_gradient_guidance_fn_shape() -> None:
    clf = HeuristicBehaviorClassifier()
    gfn = classifier_gradient_guidance_fn(clf, target_class=0, scale=0.25)
    x = torch.randn(2, 64, 2)
    t = torch.zeros(2, 1, 1)
    v = torch.randn(2, 64, 2)
    delta = gfn(x=x, t=t, v=v, step_index=0)
    assert delta.shape == v.shape
    assert torch.isfinite(delta).all()


def test_flow_matching_accepts_denoising_guidance_fn() -> None:
    fm = FlowMatching(x_dims=(64, 2), num_inference_steps=4)
    device = torch.device("cpu")

    def step_fn(*, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def guidance(*, x: torch.Tensor, t: torch.Tensor, v: torch.Tensor, step_index: int) -> torch.Tensor:
        return 0.01 * torch.randn_like(v)

    x = fm.sample(
        batch_size=3,
        step_fn=step_fn,
        device=device,
        inference_step=4,
        denoising_guidance_fn=guidance,
    )
    assert x.shape == (3, 64, 2)
