# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch ``Dataset`` that serves ``(x_t, t, y)`` triples for classifier training.

Matches the flow-matching forward process used by the expert:

    x_t = (1 − t) · x_0 + t · x_1,   x_0 ∼ N(0, I),   t ∼ U(0, 1).

``x_1`` is the clean, **normalized** action tensor cached from a ground-truth
rollout by :func:`build_labeled_cache`. Sampling ``t`` and ``x_0`` at
``__getitem__`` time means every epoch sees each ``x_1`` at a different
noise level and a different noise draw, which is what we want.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from alpamayo1_5.data.label_extraction import load_labeled_cache


def split_indices(
    labels: torch.Tensor | np.ndarray,
    *,
    val_frac: float = 0.1,
    seed: int = 0,
    stratified: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic ``(train_idx, val_idx)`` int64 arrays.

    With ``stratified=True`` (default), each class contributes the same
    fraction to val — important here because turn classes are <3 % of the
    cache and a plain random split routinely puts <10 turn samples in val.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    rng = np.random.default_rng(seed)

    if not stratified:
        perm = rng.permutation(labels_np.shape[0])
        n_val = max(1, int(round(labels_np.shape[0] * val_frac)))
        val = np.sort(perm[:n_val])
        train = np.sort(perm[n_val:])
        return train.astype(np.int64), val.astype(np.int64)

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for k in np.unique(labels_np):
        idx_k = np.where(labels_np == k)[0]
        rng.shuffle(idx_k)
        n_val_k = max(1, int(round(idx_k.shape[0] * val_frac)))
        val_parts.append(idx_k[:n_val_k])
        train_parts.append(idx_k[n_val_k:])
    train = np.sort(np.concatenate(train_parts)).astype(np.int64)
    val = np.sort(np.concatenate(val_parts)).astype(np.int64)
    return train, val


class NoisyActionDataset(Dataset):
    """Serve noisy action tensors plus labels, on demand.

    Each sample is drawn fresh at ``__getitem__``; the underlying ``x_1``
    tensor is reused across epochs. Validation uses a fixed-``t`` mode so
    the per-noise-level accuracy curve is reproducible.
    """

    def __init__(
        self,
        cache: dict[str, Any] | str | Path,
        *,
        indices: np.ndarray | None = None,
        mode: str = "train",
        fixed_t: float | None = None,
        seed: int | None = None,
    ) -> None:
        if isinstance(cache, (str, Path)):
            cache = load_labeled_cache(cache)
        self._actions: torch.Tensor = cache["actions"].float()
        self._labels: torch.Tensor = cache["labels"].long()
        self.label_names: list[str] = list(cache["label_names"])
        self.normalization: dict[str, float] = dict(cache.get("normalization", {}))

        n = self._actions.shape[0]
        if indices is None:
            self._indices = np.arange(n, dtype=np.int64)
        else:
            self._indices = np.asarray(indices, dtype=np.int64)

        if mode not in ("train", "val"):
            raise ValueError(f"mode must be 'train' or 'val', got {mode!r}")
        if mode == "val" and fixed_t is None:
            raise ValueError("val mode requires fixed_t in [0, 1]")
        self.mode = mode
        self.fixed_t = None if fixed_t is None else float(fixed_t)
        self._val_seed = 0 if seed is None else int(seed)

    def __len__(self) -> int:
        return self._indices.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = int(self._indices[i])
        x1 = self._actions[idx]
        y = self._labels[idx]

        if self.mode == "train":
            t = torch.rand(())
            x0 = torch.randn_like(x1)
        else:
            t = torch.tensor(float(self.fixed_t))
            g = torch.Generator().manual_seed(self._val_seed * 1_000_003 + idx)
            x0 = torch.randn(x1.shape, generator=g)

        x_t = (1.0 - t) * x0 + t * x1
        return x_t.float(), t.float(), y
