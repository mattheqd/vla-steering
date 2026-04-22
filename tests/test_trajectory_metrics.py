# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from alpamayo1_5.action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace
from alpamayo1_5.metrics import (
    mean_norm_accel_kappa_from_pred_traj,
    mean_xy_step_length,
    min_ade_fde,
    trajectory_shift_l2,
)


def test_min_ade_fde_zero_when_matching_gt() -> None:
    T = 64
    gt = torch.zeros(1, 1, T, 3)
    gt[..., 0] = torch.linspace(0, 1, T).view(1, 1, T)
    pred = gt.view(1, 1, 1, T, 3).clone()
    ade, fde = min_ade_fde(pred, gt)
    assert ade < 1e-5
    assert fde < 1e-5


def test_trajectory_shift_l2() -> None:
    T = 8
    a = torch.zeros(1, 1, 1, T, 3)
    b = torch.zeros(1, 1, 1, T, 3)
    b[..., 0] = 1.0
    d = trajectory_shift_l2(a, b)
    assert abs(d - 1.0) < 1e-5


def test_mean_xy_step_length() -> None:
    T = 4
    p = torch.zeros(1, 1, 1, T, 3)
    p[0, 0, 0, :, 0] = torch.arange(T, dtype=torch.float32)
    s = mean_xy_step_length(p)
    assert abs(s - 1.0) < 1e-5


def test_mean_norm_accel_kappa_from_pred_traj() -> None:
    sp = UnicycleAccelCurvatureActionSpace()
    B, T, Th = 1, 64, 2
    hist_xyz = torch.zeros(B, Th, 3)
    hist_rot = torch.eye(3).expand(B, Th, 3, 3).clone()
    fut_xyz = torch.zeros(B, T, 3)
    fut_rot = torch.eye(3).expand(B, T, 3, 3).clone()
    ma, mk = mean_norm_accel_kappa_from_pred_traj(sp, hist_xyz, hist_rot, fut_xyz, fut_rot)
    assert isinstance(ma, float) and isinstance(mk, float)
