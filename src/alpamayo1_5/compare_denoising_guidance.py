# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A/B baseline vs denoising-time classifier guidance on one dataset clip.

Resets RNG to the same seed before each full ``sample_trajectories_from_data_with_vlm_rollout``
so the VLM rollout and flow noise initialization match as closely as PyTorch / cuDNN allow;
only the guided run adds ``denoising_guidance_fn`` during Euler integration.

Usage (from repo root, with GPU and HF auth)::

    python src/alpamayo1_5/compare_denoising_guidance.py
    python src/alpamayo1_5/compare_denoising_guidance.py --guidance-scale 0.25 --target-class 2

Set ``ALPAMAYO_DEBUG=0`` for quieter logs (same as ``test_inference``).
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from typing import Any

import numpy as np
import torch

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.steering import HeuristicBehaviorClassifier, classifier_gradient_guidance_fn


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _metrics_from_data(pred_xyz: torch.Tensor, data: dict[str, Any]) -> dict[str, float]:
    """Ground truth from ``data`` (CPU tensors ok)."""
    gt_xy = data["ego_future_xyz"].cpu().numpy()[0, 0, :, :2]
    pred_xy = pred_xyz.detach().float().cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy.T[None, ...], axis=1).mean(-1)
    min_ade = float(diff.min())
    pred_single = pred_xyz.detach().float().cpu().numpy()[0, 0, 0, :, :2]
    fde = float(np.linalg.norm(pred_single[-1] - gt_xy[-1]))
    mean_step = float(np.linalg.norm(np.diff(pred_single, axis=0), axis=1).mean())
    return {"min_ade_m": min_ade, "fde_m": fde, "mean_step_xy_m": mean_step}


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B baseline vs denoising guidance")
    parser.add_argument("--clip-id", type=str, default="030c760c-ae38-49aa-9ad8-f5650a545d26")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=0.15)
    parser.add_argument("--target-class", type=int, default=0, help="0=yield-ish, 1=neutral, 2=accel-ish")
    parser.add_argument("--model", type=str, default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable cudnn deterministic mode (slower; improves A/B match).",
    )
    args = parser.parse_args()

    if os.environ.get("ALPAMAYO_DEBUG", "1").lower() not in ("0", "false", "no"):
        logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
        logging.getLogger("alpamayo1_5").setLevel(logging.DEBUG)

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Loading dataset clip_id={args.clip_id} ...")
    data = load_physical_aiavdataset(args.clip_id, t0_us=5_100_000)
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"]
    )

    print(f"Loading model {args.model} ...")
    model = Alpamayo1_5.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    clf = HeuristicBehaviorClassifier().to("cuda")
    guidance_fn = classifier_gradient_guidance_fn(
        clf, target_class=args.target_class, scale=args.guidance_scale
    )

    _set_global_seed(args.seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_a, _, extra_a = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
            diffusion_kwargs=None,
        )
    metrics_a = _metrics_from_data(pred_a, data)
    print("\n--- Baseline (no denoising guidance) ---")
    print("CoC snippet:", extra_a["cot"][0])
    print(
        f"minADE={metrics_a['min_ade_m']:.4f} m  FDE={metrics_a['fde_m']:.4f} m  "
        f"mean_xy_step={metrics_a['mean_step_xy_m']:.4f} m"
    )

    _set_global_seed(args.seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_b, _, extra_b = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
            diffusion_kwargs={"denoising_guidance_fn": guidance_fn},
        )
    metrics_b = _metrics_from_data(pred_b, data)
    print("\n--- Guided (denoising_guidance_fn) ---")
    print("CoC snippet:", extra_b["cot"][0])
    print(
        f"minADE={metrics_b['min_ade_m']:.4f} m  FDE={metrics_b['fde_m']:.4f} m  "
        f"mean_xy_step={metrics_b['mean_step_xy_m']:.4f} m"
    )

    print("\n--- Delta (guided - baseline) ---")
    print(
        f"ΔminADE={metrics_b['min_ade_m'] - metrics_a['min_ade_m']:+.4f} m  "
        f"ΔFDE={metrics_b['fde_m'] - metrics_a['fde_m']:+.4f} m  "
        f"Δmean_xy_step={metrics_b['mean_step_xy_m'] - metrics_a['mean_step_xy_m']:+.4f} m"
    )
    if str(extra_a["cot"]) != str(extra_b["cot"]):
        print(
            "\nNote: CoC strings differ between runs; VLM sampling is not fully deterministic "
            "unless you enable --deterministic and a fixed backend. Diff A/B is still valid "
            "when CoC matches; re-run or lower temperature if you need identical CoC."
        )


if __name__ == "__main__":
    main()
