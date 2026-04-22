# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and load the ``(action_x1, label)`` cache of weakly-labelled clips.

We iterate clip IDs, sample a grid of ``t0_us`` per clip, load the
ego-future polyline, bucket it into a kinematic behavior class, and store
the **normalized** action tensor ``x1 = traj_to_action(...)`` alongside.

The cache is self-contained — normalization constants are included so that
downstream consumers (visualization, analysis, or a future training step)
do not need the 22 GB VLA model loaded.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from alpamayo1_5.action_space.action_space import ActionSpace
from alpamayo1_5.data.clip_enumeration import sample_t0_us_grid
from alpamayo1_5.labels import LABEL_NAMES, label_from_future

logger = logging.getLogger(__name__)


_NUM_HISTORY_STEPS = 16
_NUM_FUTURE_STEPS = 64
_TIME_STEP = 0.1
_STEP_US = int(_TIME_STEP * 1_000_000)


def _estimate_clip_duration_us(loader_probe: Callable[..., dict[str, Any]], clip_id: str) -> int | None:
    """Infer clip length (µs) by probing a small t0 and reading timestamp metadata.

    Falls back to ``None`` if the probe itself fails (clip unavailable,
    network error, etc.). The caller skips such clips.
    """
    t0 = _NUM_HISTORY_STEPS * _STEP_US + _STEP_US  # first valid t0
    try:
        data = loader_probe(clip_id=clip_id, t0_us=t0)
    except Exception as e:  # pragma: no cover — network/streaming-dependent
        logger.warning("probe failed for clip %s: %s", clip_id, e)
        return None
    ts = data.get("absolute_timestamps")
    if ts is None:
        return None
    return int(ts.max().item())


def _extract_action_x1(
    action_space: ActionSpace,
    data: dict[str, Any],
) -> torch.Tensor | None:
    """Run ``traj_to_action`` on a loaded clip dict; return ``(T, 2)`` or ``None``."""
    try:
        act = action_space.traj_to_action(
            data["ego_history_xyz"],
            data["ego_history_rot"],
            data["ego_future_xyz"],
            data["ego_future_rot"],
        )
        if isinstance(act, tuple):
            act = act[0]
        # act: (1, 1, T, 2)
        return act.detach().float().cpu().squeeze(0).squeeze(0).contiguous()
    except Exception as e:  # pragma: no cover — model-config-dependent
        logger.warning("traj_to_action failed: %s", e)
        return None


def build_labeled_cache(
    clip_ids: list[str],
    action_space: ActionSpace,
    *,
    out_path: str | Path,
    n_t0_per_clip: int = 5,
    min_gap_s: float = 2.0,
    loader: Callable[..., dict[str, Any]] | None = None,
    resume: bool = True,
    max_clips: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Stream clips, label them, and append rows to a ``.pt`` cache.

    Args:
        clip_ids: Strings to iterate in order.
        action_space: The model's action space (use the model's own
            instance to preserve trained normalization constants).
        out_path: Where to save the ``.pt`` cache (parent dirs created).
        n_t0_per_clip: Max ``t0_us`` samples per clip.
        min_gap_s: Minimum seconds between consecutive ``t0_us`` samples.
        loader: ``(clip_id, t0_us) -> dict`` loader. Defaults to
            :func:`load_physical_aiavdataset` (lazy-imported).
        resume: If ``True`` and ``out_path`` exists, skip clip IDs already
            present; new rows are appended.
        max_clips: Optional cap on clip count (useful for smoke tests).
        progress: Optional ``(done, total)`` callback called after each clip.

    Returns:
        The saved cache dict (also written to disk).
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if loader is None:  # pragma: no cover — network-dependent
        from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset as _loader

        loader = _loader

    actions: list[torch.Tensor] = []
    labels: list[int] = []
    cids: list[str] = []
    t0s: list[int] = []
    done: set[str] = set()

    if resume and out.is_file():
        prior = torch.load(out, map_location="cpu", weights_only=False)
        actions = [t for t in prior.get("actions", torch.empty(0, 64, 2))]
        labels = list(prior.get("labels", torch.empty(0, dtype=torch.long)).tolist())
        cids = list(prior.get("clip_ids", []))
        t0s = list(prior.get("t0s", torch.empty(0, dtype=torch.long)).tolist())
        done = set(cids)
        logger.info("resumed from %s with %d rows (%d unique clips)", out, len(labels), len(done))

    queue = [c for c in clip_ids if c not in done]
    if max_clips is not None:
        queue = queue[:max_clips]

    accel_mean = float(getattr(action_space, "accel_mean", torch.tensor(0.0)).item())
    accel_std = float(getattr(action_space, "accel_std", torch.tensor(1.0)).item())
    curv_mean = float(getattr(action_space, "curvature_mean", torch.tensor(0.0)).item())
    curv_std = float(getattr(action_space, "curvature_std", torch.tensor(1.0)).item())

    for i, cid in enumerate(queue):
        try:
            duration_us = _estimate_clip_duration_us(loader, cid)
        except Exception as e:  # pragma: no cover
            logger.warning("duration probe errored for %s: %s", cid, e)
            duration_us = None
        if duration_us is None:
            if progress:
                progress(i + 1, len(queue))
            continue
        t0_list = sample_t0_us_grid(
            duration_us,
            n=n_t0_per_clip,
            num_history_steps=_NUM_HISTORY_STEPS,
            num_future_steps=_NUM_FUTURE_STEPS,
            time_step=_TIME_STEP,
            min_gap_s=min_gap_s,
        )
        for t0 in t0_list:
            try:
                data = loader(clip_id=cid, t0_us=t0)
            except Exception as e:  # pragma: no cover
                logger.warning("load failed clip=%s t0=%d: %s", cid, t0, e)
                continue
            fut_xyz = data["ego_future_xyz"].detach().cpu().numpy()[0, 0]
            fut_rot = data["ego_future_rot"].detach().cpu().numpy()[0, 0]
            y = label_from_future(fut_xyz, fut_rot, dt=_TIME_STEP)
            if y is None:
                continue
            x1 = _extract_action_x1(action_space, data)
            if x1 is None or not torch.isfinite(x1).all():
                continue
            actions.append(x1)
            labels.append(int(y))
            cids.append(str(cid))
            t0s.append(int(t0))
        if progress:
            progress(i + 1, len(queue))
        if (i + 1) % 25 == 0:
            _save_cache(
                out, actions, labels, cids, t0s,
                accel_mean, accel_std, curv_mean, curv_std,
            )
    cache = _save_cache(
        out, actions, labels, cids, t0s,
        accel_mean, accel_std, curv_mean, curv_std,
    )
    _write_stats_sidecar(out, labels)
    return cache


def _save_cache(
    out: Path,
    actions: list[torch.Tensor],
    labels: list[int],
    cids: list[str],
    t0s: list[int],
    accel_mean: float,
    accel_std: float,
    curv_mean: float,
    curv_std: float,
) -> dict[str, Any]:
    acts = torch.stack(actions, dim=0) if actions else torch.empty(0, 64, 2)
    cache: dict[str, Any] = {
        "actions": acts.float().contiguous(),
        "labels": torch.tensor(labels, dtype=torch.long),
        "clip_ids": list(cids),
        "t0s": torch.tensor(t0s, dtype=torch.long),
        "label_names": list(LABEL_NAMES),
        "normalization": {
            "accel_mean": accel_mean,
            "accel_std": accel_std,
            "curvature_mean": curv_mean,
            "curvature_std": curv_std,
        },
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(cache, tmp)
    tmp.replace(out)
    return cache


def _write_stats_sidecar(out: Path, labels: list[int]) -> None:
    counts = Counter(labels)
    stats = {
        "num_rows": len(labels),
        "per_class_counts": {LABEL_NAMES[k]: counts.get(k, 0) for k in range(len(LABEL_NAMES))},
    }
    sidecar = out.with_suffix(".stats.json")
    sidecar.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_labeled_cache(path: str | Path) -> dict[str, Any]:
    """Read a cache produced by :func:`build_labeled_cache` from ``.pt``."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"labeled cache not found: {p}")
    cache = torch.load(p, map_location="cpu", weights_only=False)
    required = {"actions", "labels", "label_names", "normalization"}
    missing = required - set(cache)
    if missing:
        raise ValueError(f"cache {p} is missing keys: {sorted(missing)}")
    acts = cache["actions"]
    labels = cache["labels"]
    if acts.ndim != 3 or acts.shape[-2:] != (64, 2):
        raise ValueError(f"cache {p}: expected actions (N, 64, 2), got {tuple(acts.shape)}")
    if labels.shape[0] != acts.shape[0]:
        raise ValueError(
            f"cache {p}: labels ({labels.shape[0]}) and actions ({acts.shape[0]}) disagree"
        )
    return cache
