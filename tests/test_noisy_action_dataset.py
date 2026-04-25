# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from alpamayo1_5.data import NoisyActionDataset, split_indices


def _toy_cache(n: int = 50, K: int = 5) -> dict:
    torch.manual_seed(0)
    return {
        "actions": torch.randn(n, 64, 2),
        "labels": torch.arange(n) % K,
        "label_names": ["yield", "cruise", "accelerate", "turn_left", "turn_right"],
        "normalization": {
            "accel_mean": 0.0, "accel_std": 1.0,
            "curvature_mean": 0.0, "curvature_std": 1.0,
        },
    }


def test_split_indices_random_disjoint_and_covers_all() -> None:
    labels = (torch.arange(100) % 5)
    train, val = split_indices(labels, val_frac=0.2, seed=7, stratified=False)
    assert train.size + val.size == 100
    assert set(train.tolist()).isdisjoint(val.tolist())
    assert sorted(train.tolist() + val.tolist()) == list(range(100))


def test_split_indices_stratified_balances_per_class() -> None:
    labels = torch.tensor([0]*100 + [1]*20 + [2]*5)
    train, val = split_indices(labels, val_frac=0.2, seed=0, stratified=True)
    val_counts = np.bincount(labels[val].numpy(), minlength=3)
    # Each class is represented in val (with val_frac=0.2): 20, 4, 1 expected
    assert (val_counts > 0).all()
    assert val_counts[0] == 20
    assert val_counts[1] == 4
    assert val_counts[2] == 1


def test_train_mode_shapes_and_random_t() -> None:
    cache = _toy_cache()
    ds = NoisyActionDataset(cache, mode="train")
    x_t, t, y = ds[0]
    assert x_t.shape == (64, 2)
    assert t.shape == ()
    assert y.shape == ()
    ts = torch.stack([ds[i % len(ds)][1] for i in range(64)])
    assert float(ts.min()) >= 0.0 and float(ts.max()) <= 1.0
    assert float(ts.std()) > 0.05


def test_val_mode_deterministic() -> None:
    cache = _toy_cache()
    ds1 = NoisyActionDataset(cache, mode="val", fixed_t=0.3, seed=123)
    ds2 = NoisyActionDataset(cache, mode="val", fixed_t=0.3, seed=123)
    x1, t1, y1 = ds1[5]
    x2, t2, y2 = ds2[5]
    assert torch.allclose(x1, x2)
    assert float(t1) == pytest.approx(0.3)
    assert int(y1) == int(y2)


def test_val_mode_requires_fixed_t() -> None:
    with pytest.raises(ValueError):
        NoisyActionDataset(_toy_cache(), mode="val")


def test_val_noise_changes_with_seed() -> None:
    cache = _toy_cache()
    ds1 = NoisyActionDataset(cache, mode="val", fixed_t=0.3, seed=1)
    ds2 = NoisyActionDataset(cache, mode="val", fixed_t=0.3, seed=2)
    assert not torch.allclose(ds1[0][0], ds2[0][0])


def test_forward_flow_identity_at_t1() -> None:
    cache = _toy_cache()
    ds = NoisyActionDataset(cache, mode="val", fixed_t=1.0, seed=0)
    x_t, t, _ = ds[3]
    assert torch.allclose(x_t, cache["actions"][3], atol=1e-6)
    assert float(t) == pytest.approx(1.0)
