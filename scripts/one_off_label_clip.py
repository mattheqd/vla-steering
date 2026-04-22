#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-off: pull a single Physical AI clip and label it.

Smoke test for the labeling pipeline on a single clip. No cache is built;
nothing is saved. Prints:

* clip metadata
* a kinematic summary of the ground-truth future polyline
* the **weak label** assigned by ``label_from_future`` (or ``ambiguous``)
* the **HeuristicBehaviorClassifier** prediction on the clean action tensor
  (for comparison — this is the existing guidance classifier, not a trained one)

Usage::

    # Default clip (the one used by test_inference.py); needs HF gated access
    # and the Alpamayo-1.5 config (~small download, not the 22 GB of weights).
    python scripts/one_off_label_clip.py

    # Specific clip / offset
    python scripts/one_off_label_clip.py \\
        --clip-id 030c760c-ae38-49aa-9ad8-f5650a545d26 \\
        --t0-us 5100000

    # Skip downloading the model config and use identity normalization.
    # Labels still work; action-space stats (``mean_norm_*``) won't match
    # the expert's distribution — debug / offline use only.
    python scripts/one_off_label_clip.py --identity-action-space
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch


_DT = 0.1


def _load_action_space(model_id: str, identity: bool):
    if identity:
        from alpamayo1_5.action_space.unicycle_accel_curvature import (
            UnicycleAccelCurvatureActionSpace,
        )

        print("[one_off] WARNING: using identity action-space normalization (--identity-action-space)")
        return UnicycleAccelCurvatureActionSpace()

    import hydra.utils as hyu
    from alpamayo1_5.config import Alpamayo1_5Config

    cfg = Alpamayo1_5Config.from_pretrained(model_id)
    action_space = hyu.instantiate(cfg.action_space_cfg)
    return action_space


def _print_kinematics(fut_xyz: np.ndarray, fut_rot: np.ndarray) -> None:
    from alpamayo1_5.labels.kinematic_labels import (
        signed_heading_change_from_rot,
        signed_heading_change_from_xy,
    )
    from alpamayo1_5.metrics.behavior_metrics import (
        mean_speed,
        min_speed,
        time_to_stop,
    )

    xy = fut_xyz[:, :2].astype(np.float64, copy=False)
    v_mean = mean_speed(xy, _DT)
    v_min = min_speed(xy, _DT)
    t_stop = time_to_stop(xy, _DT, speed_threshold=0.2)
    yaw_rot = signed_heading_change_from_rot(np.asarray(fut_rot, dtype=np.float64))
    yaw_xy = signed_heading_change_from_xy(xy)
    # Simple Δv = v_end_segment - v_start_segment
    diffs = np.linalg.norm(np.diff(xy, axis=0), axis=1) / _DT
    delta_v = float(diffs[-1] - diffs[0]) if diffs.size >= 2 else 0.0

    def _fmt(x: float) -> str:
        return f"{x:.3f}" if not (isinstance(x, float) and math.isnan(x)) else "  nan"

    print(f"  mean speed (m/s)           : {_fmt(v_mean)}")
    print(f"  min  speed (m/s)           : {_fmt(v_min)}")
    print(f"  Δv  (v_end − v_start, m/s) : {_fmt(delta_v)}")
    print(f"  signed yaw sum from rot    : {_fmt(yaw_rot)}  rad")
    print(f"  signed yaw sum from xy     : {_fmt(yaw_xy)}  rad")
    print(f"  time-to-stop (0.2 m/s)     : {_fmt(t_stop)}  s")


def _classify_with(model, x1_tt: torch.Tensor, t_value: float, label: str) -> None:
    model.eval()
    with torch.no_grad():
        logits = model(x1_tt, torch.full((x1_tt.shape[0],), t_value, dtype=torch.float32))
        probs = torch.softmax(logits, dim=-1)[0]
        pred = int(torch.argmax(probs).item())
    names = getattr(model, "label_names", None)
    if names is None:
        names = [f"class_{i}" for i in range(probs.numel())]
    print(f"  {label}: pred={pred} ({names[pred]})")
    print("    probs: " + ", ".join(f"{n}={p:.3f}" for n, p in zip(names, probs.tolist())))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip-id", type=str, default="030c760c-ae38-49aa-9ad8-f5650a545d26")
    p.add_argument("--t0-us", type=int, default=5_100_000)
    p.add_argument("--model", type=str, default="nvidia/Alpamayo-1.5-10B")
    p.add_argument("--identity-action-space", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # Local imports so the script can --help without pulling physical_ai_av.
    from alpamayo1_5.labels import LABEL_NAMES, label_from_future
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo1_5.steering import HeuristicBehaviorClassifier

    print(f"[one_off] loading clip {args.clip_id} @ t0={args.t0_us} µs ...")
    data = load_physical_aiavdataset(args.clip_id, t0_us=args.t0_us)
    print(f"[one_off] loaded. image_frames={tuple(data['image_frames'].shape)} "
          f"ego_future_xyz={tuple(data['ego_future_xyz'].shape)}")

    fut_xyz = data["ego_future_xyz"].detach().cpu().numpy()[0, 0]
    fut_rot = data["ego_future_rot"].detach().cpu().numpy()[0, 0]

    print("\n== kinematic summary of ground-truth future (6.4 s, 10 Hz) ==")
    _print_kinematics(fut_xyz, fut_rot)

    y = label_from_future(fut_xyz, fut_rot, dt=_DT)
    if y is None:
        print("\n== weak label ==\n  ambiguous → None  (dropped at cache-build time)")
    else:
        print(f"\n== weak label ==\n  class {y}  ({LABEL_NAMES[y]})")

    print("\n== loading action space ==")
    action_space = _load_action_space(args.model, args.identity_action_space)

    act = action_space.traj_to_action(
        data["ego_history_xyz"],
        data["ego_history_rot"],
        data["ego_future_xyz"],
        data["ego_future_rot"],
    )
    if isinstance(act, tuple):
        act = act[0]
    x1 = act.detach().float().cpu().squeeze(0).squeeze(0).contiguous()
    if not torch.isfinite(x1).all():
        raise SystemExit("[one_off] non-finite values in x1 — check the clip / action space config")
    print(f"  x1 shape={tuple(x1.shape)}  mean={x1.mean().item():+.4f}  std={x1.std().item():.4f}")
    print(f"  accel  channel mean={x1[..., 0].mean().item():+.4f}  std={x1[..., 0].std().item():.4f}")
    print(f"  kappa  channel mean={x1[..., 1].mean().item():+.4f}  std={x1[..., 1].std().item():.4f}")

    x1_tt = x1.unsqueeze(0).to(args.device)  # (1, 64, 2)

    print("\n== classifier predictions on clean x1 (t = 1.0) ==")
    heur = HeuristicBehaviorClassifier().to(args.device)
    # Heuristic's 3 classes are yield/neutral/accel — give them readable names.
    heur.label_names = ["yield-ish", "neutral", "accel-ish"]
    _classify_with(heur, x1_tt, t_value=1.0, label="heuristic (3-class, untrained)")

    print("\n== done ==")
    if y is not None:
        print(f"reminder: this clip is a '{LABEL_NAMES[y]}' sample under our weak-label rules.")


if __name__ == "__main__":
    main()
