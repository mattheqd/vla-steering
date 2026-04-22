# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np

from alpamayo1_5.labels.kinematic_labels import (
    LABEL_NAMES,
    label_from_future,
    signed_heading_change_from_rot,
    signed_heading_change_from_xy,
)


_DT = 0.1
_T = 64


def _straight_constant_speed(v_mps: float) -> np.ndarray:
    x = np.arange(_T, dtype=np.float64) * v_mps * _DT
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=1)


def _straight_linear_speed(v_start: float, v_end: float) -> np.ndarray:
    vs = np.linspace(v_start, v_end, _T, dtype=np.float64)
    dx = vs * _DT
    x = np.cumsum(dx)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=1)


def _arc(v_mps: float, yaw_total: float) -> tuple[np.ndarray, np.ndarray]:
    """Constant-speed arc of total signed heading change ``yaw_total`` (rad)."""
    thetas = np.linspace(0.0, yaw_total, _T, dtype=np.float64)
    step = v_mps * _DT
    dx = step * np.cos(thetas)
    dy = step * np.sin(thetas)
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    z = np.zeros_like(x)
    xyz = np.stack([x, y, z], axis=1)
    rot = np.zeros((_T, 3, 3), dtype=np.float64)
    rot[:, 2, 2] = 1.0
    rot[:, 0, 0] = np.cos(thetas)
    rot[:, 0, 1] = -np.sin(thetas)
    rot[:, 1, 0] = np.sin(thetas)
    rot[:, 1, 1] = np.cos(thetas)
    return xyz, rot


def test_label_cruise_steady_speed() -> None:
    xyz = _straight_constant_speed(10.0)
    assert label_from_future(xyz, None, dt=_DT) == LABEL_NAMES.index("cruise")


def test_label_accelerate() -> None:
    xyz = _straight_linear_speed(5.0, 12.0)
    assert label_from_future(xyz, None, dt=_DT) == LABEL_NAMES.index("accelerate")


def test_label_yield_deceleration() -> None:
    xyz = _straight_linear_speed(12.0, 5.0)
    assert label_from_future(xyz, None, dt=_DT) == LABEL_NAMES.index("yield")


def test_label_yield_stopping() -> None:
    xyz = _straight_linear_speed(6.0, 0.0)
    assert label_from_future(xyz, None, dt=_DT) == LABEL_NAMES.index("yield")


def test_label_turn_left_then_right() -> None:
    xyz_l, rot_l = _arc(5.0, +0.8)
    assert label_from_future(xyz_l, rot_l, dt=_DT) == LABEL_NAMES.index("turn_left")
    xyz_r, rot_r = _arc(5.0, -0.8)
    assert label_from_future(xyz_r, rot_r, dt=_DT) == LABEL_NAMES.index("turn_right")


def test_label_none_when_turn_with_strong_accel() -> None:
    # Acceleration while turning → ambiguous under our 5-class rules.
    xyz, rot = _arc(5.0, +0.8)
    # Monkey-speed: scale x so final speed is much higher (cheap way to fake Δv).
    speeds = np.linspace(3.0, 15.0, _T)
    steps = speeds * _DT
    thetas = np.linspace(0.0, 0.8, _T)
    dx = steps * np.cos(thetas)
    dy = steps * np.sin(thetas)
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    xyz_fast = np.stack([x, y, np.zeros_like(x)], axis=1)
    assert label_from_future(xyz_fast, rot, dt=_DT) is None


def test_signed_heading_sign_matches_direction() -> None:
    _, rot_l = _arc(5.0, +0.8)
    _, rot_r = _arc(5.0, -0.8)
    assert signed_heading_change_from_rot(rot_l) > 0.0
    assert signed_heading_change_from_rot(rot_r) < 0.0

    xyz_l, _ = _arc(5.0, +0.8)
    xyz_r, _ = _arc(5.0, -0.8)
    assert signed_heading_change_from_xy(xyz_l[:, :2]) > 0.0
    assert signed_heading_change_from_xy(xyz_r[:, :2]) < 0.0


def test_degenerate_inputs_return_none_or_zero() -> None:
    assert label_from_future(np.zeros((2, 3)), None, dt=_DT) is None
    assert signed_heading_change_from_xy(np.zeros((2, 2))) == 0.0
    assert math.isfinite(signed_heading_change_from_rot(np.zeros((1, 3, 3))))
