# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trajectory-level metrics for denoising-guidance experiments (numpy + torch)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from alpamayo1_5.action_space.action_space import ActionSpace


def min_ade_fde(
    pred_xyz: torch.Tensor,
    gt_future_xyz: torch.Tensor,
    *,
    sample_index: int = 0,
) -> tuple[float, float]:
    """minADE and FDE (last waypoint) in meters, single clip with ``num_traj_samples`` trajectories.

    Args:
        pred_xyz: ``(B, n_sets, n_samples, T, 3)`` — uses ``B=0``, ``n_sets=0``.
        gt_future_xyz: ``(B, n_traj, T, 3)`` — uses ``[0, 0, :, :]`` for XY ground truth.
        sample_index: Which sample index along ``n_samples`` to use for FDE (ADE is min over all).

    Returns:
        ``(min_ade_m, fde_m)`` averaged over time for ADE; FDE is Euclidean XY at last step.
    """
    gt_xy = gt_future_xyz.detach().float().cpu().numpy()[0, 0, :, :2]
    pred_xy = pred_xyz.detach().float().cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy.T[np.newaxis, :, :], axis=1).mean(-1)
    min_ade = float(diff.min())
    pred_single = pred_xy[sample_index]
    fde = float(np.linalg.norm(pred_single[:, -1] - gt_xy[-1]))
    return min_ade, fde


def mean_xy_step_length(pred_xyz: torch.Tensor, *, sample_index: int = 0) -> float:
    """Mean L2 distance between consecutive XY waypoints (meters)."""
    p = pred_xyz.detach().float().cpu().numpy()[0, 0, sample_index, :, :2]
    if p.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).mean())


def trajectory_shift_l2(
    pred_xyz_a: torch.Tensor,
    pred_xyz_b: torch.Tensor,
    *,
    sample_index: int = 0,
) -> float:
    """RMS L2 distance between two predicted trajectories (full XYZ per waypoint).

    In the compare experiment runner, ``pred_xyz_a`` / ``pred_xyz_b`` are the baseline and
    guided predictions for the **same** pair so this value is **guided-vs-baseline** shift.
    """
    pa = pred_xyz_a.detach().float().cpu().numpy()[0, 0, sample_index, :, :]
    pb = pred_xyz_b.detach().float().cpu().numpy()[0, 0, sample_index, :, :]
    return float(np.sqrt(np.mean(np.sum((pa - pb) ** 2, axis=-1))))


def mean_norm_accel_kappa_from_pred_traj(
    action_space: ActionSpace,
    hist_xyz: torch.Tensor,
    hist_rot: torch.Tensor,
    fut_xyz: torch.Tensor,
    fut_rot: torch.Tensor,
) -> tuple[float, float]:
    """Means of **normalized** accel and curvature **inferred from predicted trajectory polyline**.

    This is **not** the flow-matching action tensor used inside the denoiser. Values are
    obtained by applying ``ActionSpace.traj_to_action`` to the **predicted** future
    ``fut_xyz`` / ``fut_rot`` (inverse fit), then averaging channel 0 (accel) and 1 (κ) over
    waypoints—the same normalized representation the expert uses after ``action_to_traj``.
    """
    fut_xyz_f = fut_xyz.detach().float()
    fut_rot_f = fut_rot.detach().float()
    hist_xyz_f = hist_xyz.detach().float()
    hist_rot_f = hist_rot.detach().float()
    act = action_space.traj_to_action(hist_xyz_f, hist_rot_f, fut_xyz_f, fut_rot_f)
    if isinstance(act, tuple):
        act = act[0]
    a = act[..., 0].mean().item()
    k = act[..., 1].mean().item()
    return float(a), float(k)
