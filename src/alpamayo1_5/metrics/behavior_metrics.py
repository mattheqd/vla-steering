# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight behavior-oriented summaries from predicted XY (and optional rotation).

All functions use **NumPy**, evenly spaced steps, and expect **ego-frame** polylines in meters
unless noted otherwise.
"""

from __future__ import annotations

import math

import numpy as np


def mean_speed(traj_xy: np.ndarray, dt: float) -> float:
    r"""Mean speed over finite-difference segments (m/s).

    \(v_i = \|p_{i+1}-p_i\| / \Delta t\) for \(i=0,\ldots,T-2\),
    \(\bar v = \frac{1}{T-1}\sum_i v_i\).
    """
    v = segment_speeds(traj_xy, dt)
    if v.size == 0:
        return float("nan")
    return float(np.mean(v))


def min_speed(traj_xy: np.ndarray, dt: float) -> float:
    """Minimum segment speed (m/s)."""
    v = segment_speeds(traj_xy, dt)
    if v.size == 0:
        return float("nan")
    return float(np.min(v))


def segment_speeds(traj_xy: np.ndarray, dt: float) -> np.ndarray:
    """Per-segment speeds ``(T-1,)``; empty if ``T < 2`` or ``dt <= 0``."""
    if traj_xy.ndim != 2 or traj_xy.shape[1] < 2 or traj_xy.shape[0] < 2 or dt <= 0:
        return np.array([], dtype=np.float64)
    d = np.linalg.norm(np.diff(traj_xy[:, :2], axis=0), axis=1)
    return d / dt


def max_abs_lateral_displacement(traj_xy: np.ndarray) -> float:
    r"""Max absolute lateral offset from the initial motion direction through ``p_0``.

    Let \(\hat u = (p_1-p_0)/\|p_1-p_0\|\) (2D), \(\hat n = (-\hat u_y, \hat u_x)\).
    Lateral offset of \(p_i\): \(d_i = (p_i-p_0)\cdot \hat n\).
    Returns \(\max_i |d_i|\). If \(\|p_1-p_0\|\) is tiny, falls back to \(\|p_2-p_0\|\) when
    available; otherwise NaN.
    """
    if traj_xy.ndim != 2 or traj_xy.shape[0] < 2 or traj_xy.shape[1] < 2:
        return float("nan")
    p0 = traj_xy[0, :2].astype(np.float64, copy=False)
    p1 = traj_xy[1, :2].astype(np.float64, copy=False)
    u = p1 - p0
    nu = np.linalg.norm(u)
    eps = 1e-9
    if nu < eps and traj_xy.shape[0] >= 3:
        u = traj_xy[2, :2] - p0
        nu = np.linalg.norm(u)
    if nu < eps:
        return float("nan")
    u = u / nu
    n = np.array([-u[1], u[0]], dtype=np.float64)
    rel = traj_xy[:, :2] - p0
    lat = rel @ n
    return float(np.max(np.abs(lat)))


def heading_change(*, traj_xy: np.ndarray | None = None, pred_rot: np.ndarray | None = None) -> float:
    r"""Sum of absolute yaw increments along the horizon (rad).

    If ``pred_rot`` is given (shape ``(T, 3, 3)``), yaw per step is
    \(\theta_t = \mathrm{atan2}(R_{1,0}, R_{0,0})\).

    Otherwise uses ``traj_xy`` ``(T,2)`` segment headings
    \(\phi_i = \mathrm{atan2}(\Delta y_i, \Delta x_i)\) and sums \(|\mathrm{wrap}(\phi_{i+1}-\phi_i)|\).

    ``pred_rot`` takes precedence when both are provided.
    """
    if pred_rot is not None:
        return heading_change_from_rot(pred_rot)
    if traj_xy is not None:
        return heading_change_from_xy(traj_xy)
    raise ValueError("heading_change requires traj_xy or pred_rot")


def heading_change_from_xy(traj_xy: np.ndarray) -> float:
    if traj_xy.ndim != 2 or traj_xy.shape[0] < 2 or traj_xy.shape[1] < 2:
        return float("nan")
    if traj_xy.shape[0] < 3:
        return 0.0
    d = np.diff(traj_xy[:, :2], axis=0)
    phi = np.arctan2(d[:, 1], d[:, 0])
    dphi = np.diff(phi)
    dphi = (dphi + math.pi) % (2 * math.pi) - math.pi
    return float(np.sum(np.abs(dphi)))


def heading_change_from_rot(pred_rot: np.ndarray) -> float:
    """Yaw from world XY of body x-axis: ``atan2(R_{1,0}, R_{0,0})`` per row."""
    if pred_rot.ndim != 3 or pred_rot.shape[-2:] != (3, 3) or pred_rot.shape[0] < 2:
        return float("nan")
    th = np.arctan2(pred_rot[:, 1, 0], pred_rot[:, 0, 0])
    dth = np.diff(th)
    dth = (dth + math.pi) % (2 * math.pi) - math.pi
    return float(np.sum(np.abs(dth)))


def min_clearance_to_point_obstacle(traj_xy: np.ndarray, obstacle_xy: np.ndarray | None) -> float:
    """Minimum Euclidean distance from any waypoint to ``obstacle_xy`` (m).

    If ``obstacle_xy`` is ``None`` or length != 2, returns NaN (logged as null in JSON / empty CSV).
    """
    if obstacle_xy is None or traj_xy.ndim != 2 or traj_xy.shape[0] == 0 or traj_xy.shape[1] < 2:
        return float("nan")
    o = np.asarray(obstacle_xy, dtype=np.float64).reshape(-1)
    if o.size != 2:
        return float("nan")
    d = np.linalg.norm(traj_xy[:, :2] - o[None, :2], axis=1)
    return float(np.min(d))


def time_to_stop(traj_xy: np.ndarray, dt: float, *, speed_threshold: float = 0.2) -> float:
    r"""First time (s) at end of a segment whose speed drops below ``speed_threshold`` (m/s).

    Segment speeds \(v_i=\|p_{i+1}-p_i\|/\Delta t\). Returns \((i+1)\Delta t\) for the smallest
    \(i\) with \(v_i < \text{threshold}\). If never, NaN. If ``T<2``, NaN.
    """
    v = segment_speeds(traj_xy, dt)
    if v.size == 0:
        return float("nan")
    idx = np.where(v < speed_threshold)[0]
    if idx.size == 0:
        return float("nan")
    return float((idx[0] + 1) * dt)


def summarize_pred_behavior(
    traj_xy: np.ndarray,
    dt: float,
    *,
    pred_rot: np.ndarray | None = None,
    obstacle_xy: np.ndarray | None = None,
    speed_threshold: float = 0.2,
) -> dict[str, float]:
    """Compute all behavior summaries for one predicted polyline (single dict for logging)."""
    hc = heading_change(traj_xy=traj_xy, pred_rot=pred_rot)
    return {
        "mean_speed_mps": mean_speed(traj_xy, dt),
        "min_speed_mps": min_speed(traj_xy, dt),
        "max_abs_lateral_disp_m": max_abs_lateral_displacement(traj_xy),
        "heading_change_sum_abs_rad": hc,
        "min_clearance_obstacle_m": min_clearance_to_point_obstacle(traj_xy, obstacle_xy),
        "time_to_stop_s": time_to_stop(traj_xy, dt, speed_threshold=speed_threshold),
    }
