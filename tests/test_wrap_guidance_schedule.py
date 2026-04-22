# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from alpamayo1_5.steering.denoising_guidance import wrap_guidance_schedule


def _dummy(*, x: torch.Tensor, t: torch.Tensor, v: torch.Tensor, step_index: int) -> torch.Tensor:
    del x, t, step_index
    return torch.ones_like(v)


def test_wrap_all_matches_inner() -> None:
    v = torch.zeros(1, 2, 2)
    w = wrap_guidance_schedule(_dummy, schedule="all", num_inference_steps=5)
    out = w(x=v, t=v, v=v, step_index=3)
    assert torch.allclose(out, torch.ones_like(v))


def test_wrap_early_zeros_after_first_fraction() -> None:
    v = torch.zeros(1, 2, 2)
    w = wrap_guidance_schedule(_dummy, schedule="early", num_inference_steps=10)
    assert torch.allclose(w(x=v, t=v, v=v, step_index=0), torch.ones_like(v))
    assert torch.allclose(w(x=v, t=v, v=v, step_index=3), torch.ones_like(v))
    assert torch.allclose(w(x=v, t=v, v=v, step_index=4), torch.zeros_like(v))


def test_wrap_late_zeros_before_last_fraction() -> None:
    v = torch.zeros(1, 2, 2)
    w = wrap_guidance_schedule(_dummy, schedule="late", num_inference_steps=10)
    assert torch.allclose(w(x=v, t=v, v=v, step_index=0), torch.zeros_like(v))
    assert torch.allclose(w(x=v, t=v, v=v, step_index=5), torch.zeros_like(v))
    assert torch.allclose(w(x=v, t=v, v=v, step_index=6), torch.ones_like(v))
