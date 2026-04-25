# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import torch

from alpamayo1_5.steering import (
    HeuristicBehaviorClassifier,
    TrainedBehaviorClassifier,
    classifier_gradient_guidance_fn,
)


def test_output_shape_matches_heuristic_signature() -> None:
    trained = TrainedBehaviorClassifier(num_classes=5)
    heuristic = HeuristicBehaviorClassifier()
    x = torch.randn(3, 64, 2)
    t = torch.rand(3, 1, 1)
    out_trained = trained(x, t)
    out_heur = heuristic(x, t)
    assert out_trained.shape == (3, 5)
    assert out_heur.shape[0] == 3 and out_heur.ndim == 2


def test_t_none_is_accepted() -> None:
    trained = TrainedBehaviorClassifier(num_classes=5)
    out = trained(torch.randn(2, 64, 2), None)
    assert out.shape == (2, 5)
    assert torch.isfinite(out).all()


def test_gradient_flows_to_x() -> None:
    trained = TrainedBehaviorClassifier(num_classes=5)
    x = torch.randn(2, 64, 2, requires_grad=True)
    logits = trained(x, torch.full((2,), 0.3))
    logits[:, 0].sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0.0


def test_works_with_guidance_factory() -> None:
    trained = TrainedBehaviorClassifier(num_classes=5)
    gfn = classifier_gradient_guidance_fn(trained, target_class=2, scale=0.25)
    x = torch.randn(2, 64, 2)
    t = torch.full((2, 1, 1), 0.4)
    v = torch.randn(2, 64, 2)
    delta = gfn(x=x, t=t, v=v, step_index=1)
    assert delta.shape == v.shape
    assert torch.isfinite(delta).all()


def test_save_and_from_pretrained_round_trip() -> None:
    trained = TrainedBehaviorClassifier(num_classes=5, d_model=64, depth=2).eval()
    x = torch.randn(1, 64, 2)
    t = torch.full((1,), 0.5)
    with torch.no_grad():
        out_before = trained(x, t)
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "clf.pt"
        trained.save(ckpt, extra={"note": "test"})
        loaded = TrainedBehaviorClassifier.from_pretrained(ckpt, map_location="cpu").eval()
    with torch.no_grad():
        out_after = loaded(x, t)
    assert torch.allclose(out_before, out_after, atol=1e-6)


def test_forward_rejects_bad_shape() -> None:
    trained = TrainedBehaviorClassifier(num_classes=5)
    try:
        trained(torch.randn(2, 32, 2), torch.rand(2))
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong waypoint count")
