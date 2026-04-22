# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rule-based weak behavior labels derived from ground-truth future polylines.

These labels are **coarse kinematic buckets**, not scenario-level intent. They
exist so that a noise-conditioned classifier ``C(x, t)`` can be trained to
slot into the denoising-guidance hook without relying on the heuristic
prototype classifier. See the project plan for the intended replacement path
(CoC-parsed labels, human-labeled subset).
"""

from __future__ import annotations

import math

import numpy as np

from alpamayo1_5.metrics.behavior_metrics import segment_speeds, time_to_stop

LABEL_NAMES: list[str] = [
    "yield",       # 0 — slowing meaningfully or stopping within horizon
    "cruise",      # 1 — steady speed, straight
    "accelerate",  # 2 — speeding up, straight
    "turn_left",   # 3 — signed yaw gain > threshold
    "turn_right",  # 4 — signed yaw loss < −threshold
]
NUM_CLASSES: int = len(LABEL_NAMES)

# Thresholds — tuned conservatively to keep labels clean at the cost of
# dropping borderline samples via the ambiguity rule.
_DELTA_V_LONG_MPS = 1.5      # longitudinal accel / decel floor
_DELTA_V_CRUISE_MPS = 1.0    # upper bound on |Δv| for cruise
_YAW_CRUISE_RAD = 0.25       # max |Δheading_sum| for a "straight" bucket
_YAW_TURN_RAD = 0.35         # min |Δheading_sum| for a turn bucket
_STOP_SPEED_MPS = 0.2        # speed threshold used by time_to_stop
_STOP_WITHIN_S = 4.0         # stop within N seconds → yield


def signed_heading_change_from_xy(traj_xy: np.ndarray) -> float:
    """Sum of wrapped yaw increments along the XY polyline, preserving sign.

    Uses segment headings ``φ_i = atan2(Δy_i, Δx_i)`` and sums
    ``wrap(φ_{i+1} - φ_i)`` into [−π, π] per step. Positive = left turn
    (counter-clockwise), negative = right turn.
    """
    if traj_xy.ndim != 2 or traj_xy.shape[0] < 3 or traj_xy.shape[1] < 2:
        return 0.0
    d = np.diff(traj_xy[:, :2], axis=0)
    phi = np.arctan2(d[:, 1], d[:, 0])
    dphi = np.diff(phi)
    dphi = (dphi + math.pi) % (2 * math.pi) - math.pi
    return float(np.sum(dphi))


def signed_heading_change_from_rot(pred_rot: np.ndarray) -> float:
    """Signed sum of yaw increments from rotation matrices ``(T, 3, 3)``.

    Yaw per step is ``atan2(R[1,0], R[0,0])``; increments wrapped to [−π, π].
    """
    if pred_rot.ndim != 3 or pred_rot.shape[-2:] != (3, 3) or pred_rot.shape[0] < 2:
        return 0.0
    th = np.arctan2(pred_rot[:, 1, 0], pred_rot[:, 0, 0])
    dth = np.diff(th)
    dth = (dth + math.pi) % (2 * math.pi) - math.pi
    return float(np.sum(dth))


def _delta_v(traj_xy: np.ndarray, dt: float) -> float:
    """Final segment speed minus first segment speed (m/s)."""
    v = segment_speeds(traj_xy, dt)
    if v.size < 2:
        return 0.0
    return float(v[-1] - v[0])


def label_from_future(
    fut_xyz: np.ndarray,
    fut_rot: np.ndarray | None = None,
    *,
    dt: float = 0.1,
) -> int | None:
    """Assign one of ``LABEL_NAMES`` from a ground-truth future polyline.

    Args:
        fut_xyz: ``(T, 3)`` or ``(T, 2)`` future positions in the ego frame.
        fut_rot: Optional ``(T, 3, 3)`` rotation matrices; when given, yaw is
            taken from the rotations rather than the XY polyline (more stable
            on near-stationary segments).
        dt: Seconds between consecutive waypoints (Physical AI default 0.1).

    Returns:
        Integer class index in ``[0, NUM_CLASSES)``, or ``None`` if the
        sample is ambiguous (matches multiple buckets) or degenerate
        (too short / no motion).

    Rules (ambiguous samples → ``None``):

    * **turn_left**  if signed yaw ≥ ``+_YAW_TURN_RAD`` and |Δv| ≤ longitudinal floor
    * **turn_right** if signed yaw ≤ ``−_YAW_TURN_RAD`` and |Δv| ≤ longitudinal floor
    * **yield**      if Δv ≤ ``−_DELTA_V_LONG_MPS`` or trajectory stops within
      ``_STOP_WITHIN_S`` seconds, AND |yaw| ≤ ``_YAW_CRUISE_RAD``
    * **accelerate** if Δv ≥ ``+_DELTA_V_LONG_MPS`` AND |yaw| ≤ ``_YAW_CRUISE_RAD``
    * **cruise**     if |Δv| ≤ ``_DELTA_V_CRUISE_MPS`` AND |yaw| ≤ ``_YAW_CRUISE_RAD``

    Samples satisfying a turn rule **and** a strong longitudinal rule
    (|Δv| above the longitudinal floor) are dropped as ambiguous —
    "accelerate into a turn" is a legitimate behavior but not one our
    5-class buckets distinguish.
    """
    if fut_xyz.ndim != 2 or fut_xyz.shape[0] < 3:
        return None
    xy = fut_xyz[:, :2].astype(np.float64, copy=False)

    dv = _delta_v(xy, dt)
    if fut_rot is not None and fut_rot.shape[0] == xy.shape[0]:
        yaw = signed_heading_change_from_rot(np.asarray(fut_rot, dtype=np.float64))
    else:
        yaw = signed_heading_change_from_xy(xy)
    abs_yaw = abs(yaw)
    t_stop = time_to_stop(xy, dt, speed_threshold=_STOP_SPEED_MPS)

    is_turn_left = yaw >= _YAW_TURN_RAD
    is_turn_right = yaw <= -_YAW_TURN_RAD
    is_decel = dv <= -_DELTA_V_LONG_MPS
    is_stopping = (not math.isnan(t_stop)) and t_stop <= _STOP_WITHIN_S
    is_yield = is_decel or is_stopping
    is_accel = dv >= _DELTA_V_LONG_MPS
    is_cruise = (abs(dv) <= _DELTA_V_CRUISE_MPS) and (abs_yaw <= _YAW_CRUISE_RAD)

    # Turn buckets require a non-aggressive longitudinal profile so we don't
    # confuse "turn + brake" and "turn + accelerate" with pure turns.
    if (is_turn_left or is_turn_right) and abs(dv) > _DELTA_V_LONG_MPS:
        return None
    if is_turn_left and is_turn_right:
        return None  # impossible by construction, but guard anyway
    if is_turn_left:
        return LABEL_NAMES.index("turn_left")
    if is_turn_right:
        return LABEL_NAMES.index("turn_right")

    # Longitudinal buckets require roughly straight motion.
    if is_yield and is_accel:
        return None
    if abs_yaw > _YAW_CRUISE_RAD:
        return None
    if is_yield:
        return LABEL_NAMES.index("yield")
    if is_accel:
        return LABEL_NAMES.index("accelerate")
    if is_cruise:
        return LABEL_NAMES.index("cruise")

    return None
