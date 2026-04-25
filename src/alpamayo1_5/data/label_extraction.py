# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and load the ``(action_x1, label)`` cache of weakly-labelled clips.

We iterate clip IDs, fetch the clip's **egomotion feature** directly (single
HuggingFace download per clip — camera frames are **not** touched, which
makes this ~5× faster than going through ``load_physical_aiavdataset``),
sample a grid of ``t0_us`` inside the clip's valid window, bucket each
future polyline into a kinematic behavior class, and store the normalized
action tensor ``x1 = traj_to_action(...)`` alongside.

The cache is self-contained — normalization constants are included so that
downstream consumers (visualization, analysis, or a future training step)
do not need the 22 GB VLA model loaded.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import scipy.spatial.transform as spt
import torch

from alpamayo1_5.action_space.action_space import ActionSpace
from alpamayo1_5.data.clip_enumeration import sample_t0_us_grid
from alpamayo1_5.labels import LABEL_NAMES, label_from_future

logger = logging.getLogger(__name__)


_NUM_HISTORY_STEPS = 16
_NUM_FUTURE_STEPS = 64
_TIME_STEP = 0.1
_STEP_US = int(_TIME_STEP * 1_000_000)


def _history_timestamps(t0_us: int) -> np.ndarray:
    return (
        t0_us
        + np.arange(
            -(_NUM_HISTORY_STEPS - 1) * _STEP_US,
            _STEP_US // 2,
            _STEP_US,
            dtype=np.int64,
        )
    )


def _future_timestamps(t0_us: int) -> np.ndarray:
    return (
        t0_us
        + np.arange(
            _STEP_US,
            int((_NUM_FUTURE_STEPS + 0.5) * _STEP_US),
            _STEP_US,
            dtype=np.int64,
        )
    )


def _poses_to_local_frame(
    hist_xyz: np.ndarray,
    hist_quat: np.ndarray,
    fut_xyz: np.ndarray,
    fut_quat: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Mirror of the local-frame transform in ``load_physical_aiavdataset``.

    All four tensors come back with batch dims ``(1, 1, T, ...)`` so they plug
    straight into :meth:`ActionSpace.traj_to_action`.
    """
    t0_xyz = hist_xyz[-1].copy()
    t0_quat = hist_quat[-1].copy()
    t0_rot = spt.Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()

    hist_xyz_local = t0_rot_inv.apply(hist_xyz - t0_xyz)
    fut_xyz_local = t0_rot_inv.apply(fut_xyz - t0_xyz)
    hist_rot_local = (t0_rot_inv * spt.Rotation.from_quat(hist_quat)).as_matrix()
    fut_rot_local = (t0_rot_inv * spt.Rotation.from_quat(fut_quat)).as_matrix()

    def _t(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(arr)).float().unsqueeze(0).unsqueeze(0)

    return {
        "ego_history_xyz": _t(hist_xyz_local),
        "ego_history_rot": _t(hist_rot_local),
        "ego_future_xyz": _t(fut_xyz_local),
        "ego_future_rot": _t(fut_rot_local),
    }


def _ego_sample(egomotion, t0_us: int) -> dict[str, torch.Tensor]:
    """Sample (history, future) poses around ``t0_us`` from an ``Interpolator``."""
    ego_hist = egomotion(_history_timestamps(t0_us))
    ego_fut = egomotion(_future_timestamps(t0_us))
    return _poses_to_local_frame(
        np.asarray(ego_hist.pose.translation),
        np.asarray(ego_hist.pose.rotation.as_quat()),
        np.asarray(ego_fut.pose.translation),
        np.asarray(ego_fut.pose.rotation.as_quat()),
    )


def _valid_t0_range_us(egomotion) -> tuple[int, int]:
    """Return ``(lo, hi)`` µs range of valid ``t0`` values for this clip.

    Needs ``num_history_steps`` samples before ``t0`` and ``num_future_steps``
    samples after, at ``_TIME_STEP`` resolution. Clip timestamps can start
    slightly negative (per Physical AI convention), so we use the interpolator's
    actual ``time_range`` rather than assuming zero-based.
    """
    lo_raw, hi_raw = egomotion.time_range
    lo = int(lo_raw) + _NUM_HISTORY_STEPS * _STEP_US + 1
    hi = int(hi_raw) - _NUM_FUTURE_STEPS * _STEP_US
    return lo, hi


def _evenly_spaced_t0(lo_us: int, hi_us: int, *, n: int, min_gap_s: float) -> list[int]:
    """Up to ``n`` t0 values in ``[lo_us, hi_us]`` with at least ``min_gap_s`` gap."""
    if hi_us <= lo_us or n <= 0:
        return []
    gap_us = max(1, int(min_gap_s * 1_000_000))
    max_by_gap = max(1, (hi_us - lo_us) // gap_us + 1)
    k = min(n, max_by_gap)
    if k == 1:
        return [(lo_us + hi_us) // 2]
    stride = (hi_us - lo_us) // (k - 1)
    return [int(lo_us + i * stride) for i in range(k)]


def build_labeled_cache(
    clip_ids: list[str],
    action_space: ActionSpace,
    *,
    out_path: str | Path,
    n_t0_per_clip: int = 5,
    min_gap_s: float = 2.0,
    avdi: Any | None = None,
    resume: bool = True,
    max_clips: int | None = None,
    progress=None,
) -> dict[str, Any]:
    """Stream clips, label them, and append rows to a ``.pt`` cache.

    Args:
        clip_ids: Strings to iterate in order.
        action_space: Model's action space (use the model's own instance to
            preserve trained normalization constants; identity is OK for a
            first labeling pass).
        out_path: ``.pt`` file to write (parent dirs created).
        n_t0_per_clip: Max ``t0_us`` samples per clip.
        min_gap_s: Minimum seconds between consecutive ``t0_us`` samples.
        avdi: Optional pre-initialized ``PhysicalAIAVDatasetInterface``. When
            ``None``, one is lazily created and shared across clips.
        resume: If ``True`` and ``out_path`` exists, skip clip IDs already
            present; new rows are appended.
        max_clips: Optional cap on clip count (useful for smoke tests).
        progress: Optional ``(done, total)`` callback called after each clip.

    Returns:
        The saved cache dict (also written to disk).
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if avdi is None:  # pragma: no cover — network-dependent
        import physical_ai_av

        avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

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
            egomotion = avdi.get_clip_feature(
                cid, avdi.features.LABELS.EGOMOTION, maybe_stream=True
            )
        except Exception as e:  # pragma: no cover
            logger.warning("egomotion fetch failed for %s: %s", cid, e)
            if progress:
                progress(i + 1, len(queue))
            continue
        try:
            lo, hi = _valid_t0_range_us(egomotion)
        except Exception as e:  # pragma: no cover
            logger.warning("time_range read failed for %s: %s", cid, e)
            if progress:
                progress(i + 1, len(queue))
            continue

        t0_list = _evenly_spaced_t0(lo, hi, n=n_t0_per_clip, min_gap_s=min_gap_s)
        for t0 in t0_list:
            try:
                sample = _ego_sample(egomotion, t0)
            except Exception as e:  # pragma: no cover
                logger.warning("ego sample failed clip=%s t0=%d: %s", cid, t0, e)
                continue
            fut_xyz = sample["ego_future_xyz"].detach().cpu().numpy()[0, 0]
            fut_rot = sample["ego_future_rot"].detach().cpu().numpy()[0, 0]
            y = label_from_future(fut_xyz, fut_rot, dt=_TIME_STEP)
            if y is None:
                continue
            try:
                act = action_space.traj_to_action(
                    sample["ego_history_xyz"],
                    sample["ego_history_rot"],
                    sample["ego_future_xyz"],
                    sample["ego_future_rot"],
                )
                if isinstance(act, tuple):
                    act = act[0]
                x1 = act.detach().float().cpu().squeeze(0).squeeze(0).contiguous()
            except Exception as e:  # pragma: no cover
                logger.warning("traj_to_action failed clip=%s t0=%d: %s", cid, t0, e)
                continue
            if not torch.isfinite(x1).all():
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
