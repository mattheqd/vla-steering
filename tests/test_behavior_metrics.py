# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pytest

from alpamayo1_5.metrics.behavior_metrics import (
    heading_change,
    max_abs_lateral_displacement,
    mean_speed,
    min_clearance_to_point_obstacle,
    min_speed,
    summarize_pred_behavior,
    time_to_stop,
)


def test_mean_speed_constant_motion() -> None:
    dt = 0.1
    t = np.arange(10, dtype=np.float64) * (3.0 * dt)
    traj = np.stack([t, np.zeros_like(t)], axis=1)
    assert abs(mean_speed(traj, dt) - 3.0) < 1e-6
    assert abs(min_speed(traj, dt) - 3.0) < 1e-6


def test_max_abs_lateral_straight_then_bend() -> None:
    traj = np.array([[0, 0], [1, 0], [1, 1], [1, 2]], dtype=np.float64)
    d = max_abs_lateral_displacement(traj)
    assert d >= 1.0 - 1e-6


def test_heading_change_from_rot_identity() -> None:
    T = 5
    r = np.tile(np.eye(3, dtype=np.float64), (T, 1, 1))
    assert heading_change(pred_rot=r) == 0.0


def test_min_clearance_none_is_nan() -> None:
    traj = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    assert math.isnan(min_clearance_to_point_obstacle(traj, None))


def test_time_to_stop_never() -> None:
    dt = 0.1
    t = np.arange(5, dtype=np.float64) * (5.0 * dt)
    traj = np.stack([t, np.zeros_like(t)], axis=1)
    assert math.isnan(time_to_stop(traj, dt, speed_threshold=0.2))


def test_summarize_pred_behavior_keys() -> None:
    traj = np.array([[0, 0], [1, 0], [2, 0.1]], dtype=np.float64)
    d = summarize_pred_behavior(traj, 0.1, obstacle_xy=None)
    assert set(d.keys()) == {
        "mean_speed_mps",
        "min_speed_mps",
        "max_abs_lateral_disp_m",
        "heading_change_sum_abs_rad",
        "min_clearance_obstacle_m",
        "time_to_stop_s",
    }


def test_heading_change_requires_arg() -> None:
    with pytest.raises(ValueError):
        heading_change()
