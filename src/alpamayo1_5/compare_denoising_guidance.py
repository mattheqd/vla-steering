# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experiment runner: baseline vs denoising-time classifier guidance on one clip.

Default: one baseline + one guided pair (same RNG reset before each rollout).

Sweeps (``--lambda-sweep`` or ``--schedule-sweep``): repeat baseline+guided with shared
``sweep_id`` in log notes; optional summary table on stdout. Each pair re-seeds and runs a
**full** baseline then guided rollout, so CSVs contain **one baseline row per pair**—that
duplication is intentional (independent rerolls), not a logging bug.

Usage::

    python src/alpamayo1_5/compare_denoising_guidance.py
    ALPAMAYO_DEBUG=0 python src/alpamayo1_5/compare_denoising_guidance.py --log-results
    ALPAMAYO_DEBUG=0 python src/alpamayo1_5/compare_denoising_guidance.py --lambda-sweep 0.0 0.15 0.5 --log-results
    ALPAMAYO_DEBUG=0 python src/alpamayo1_5/compare_denoising_guidance.py --schedule-sweep --guidance-scale 0.3 --log-results

Set ``ALPAMAYO_DEBUG=0`` for quieter logs (same as ``test_inference``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

from alpamayo1_5 import helper
from alpamayo1_5.experiments.run_logging import RunRecord, new_run_id, save_run_record
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.metrics import (
    mean_norm_accel_kappa_from_pred_traj,
    mean_xy_step_length,
    min_ade_fde,
    trajectory_shift_l2,
)
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.steering import (
    HeuristicBehaviorClassifier,
    classifier_gradient_guidance_fn,
    wrap_guidance_schedule,
)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _coc_text(extra: dict[str, Any]) -> str:
    return str(extra.get("cot", ""))


def _inference_steps(model: Alpamayo1_5) -> int:
    return int(getattr(model.diffusion, "num_inference_steps", 10))


def _collect_row_metrics(
    model: Alpamayo1_5,
    data: dict[str, Any],
    model_inputs: dict[str, Any],
    pred_xyz: torch.Tensor,
    pred_rot: torch.Tensor,
) -> dict[str, float]:
    """ADE/FDE, XY step length, and traj→action inverse statistics (see metrics docstrings)."""
    gt = data["ego_future_xyz"]
    min_ade, fde = min_ade_fde(pred_xyz, gt)
    mstep = mean_xy_step_length(pred_xyz)
    hist_xyz = model_inputs["ego_history_xyz"][:1, -1]
    hist_rot = model_inputs["ego_history_rot"][:1, -1]
    fut_xyz = pred_xyz[0, 0, 0:1, :, :].contiguous()
    fut_rot = pred_rot[0, 0, 0:1, :, :, :].contiguous()
    try:
        ma, mk = mean_norm_accel_kappa_from_pred_traj(
            model.action_space, hist_xyz, hist_rot, fut_xyz, fut_rot
        )
    except Exception:
        ma, mk = float("nan"), float("nan")
    return {
        "min_ade_m": min_ade,
        "fde_m": fde,
        "mean_step_xy_m": mstep,
        "mean_norm_accel_from_pred_traj": ma,
        "mean_norm_kappa_from_pred_traj": mk,
    }


def _gt_xy_numpy(data: dict[str, Any]) -> np.ndarray:
    xy = data["ego_future_xyz"].detach().float().cpu().numpy()[0, 0, :, :2]
    return xy.astype(np.float32)


def _pred_xy_numpy(pred_xyz: torch.Tensor) -> np.ndarray:
    return pred_xyz.detach().float().cpu().numpy()[0, 0, 0, :, :2].astype(np.float32)


def _save_pair_artifacts(
    *,
    results_dir: str,
    pair_id: str,
    clip_id: str,
    seed: int,
    guidance_scale: float,
    guidance_schedule: str,
    coc_matched: bool,
    pair_label: str,
    timestamp_iso: str,
    sweep_id: str | None,
    sweep_mode: str | None,
    min_ade_baseline: float,
    min_ade_guided: float,
    fde_baseline: float,
    fde_guided: float,
    trajectory_shift_l2: float,
    pred_baseline: torch.Tensor,
    pred_guided: torch.Tensor,
    data: dict[str, Any],
) -> Path:
    """Write ``pair_<pair_id>.npz`` + ``pair_<pair_id>_meta.json`` under ``results_dir/artifacts/``."""
    art_dir = Path(results_dir) / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pair_{pair_id}"
    gt_xy = _gt_xy_numpy(data)
    baseline_xy = _pred_xy_numpy(pred_baseline)
    guided_xy = _pred_xy_numpy(pred_guided)
    n = min(gt_xy.shape[0], baseline_xy.shape[0], guided_xy.shape[0])
    gt_xy, baseline_xy, guided_xy = gt_xy[:n], baseline_xy[:n], guided_xy[:n]
    npz_path = art_dir / f"{stem}.npz"
    np.savez_compressed(npz_path, gt_xy=gt_xy, baseline_xy=baseline_xy, guided_xy=guided_xy)
    meta = {
        "clip_id": clip_id,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "guidance_schedule": guidance_schedule,
        "coc_matched_baseline": coc_matched,
        "pair_id": pair_id,
        "sweep_id": sweep_id,
        "sweep_mode": sweep_mode,
        "pair_label": pair_label,
        "timestamp_iso": timestamp_iso,
        "min_ade_baseline": float(min_ade_baseline),
        "min_ade_guided": float(min_ade_guided),
        "fde_baseline": float(fde_baseline),
        "fde_guided": float(fde_guided),
        "trajectory_shift_l2": float(trajectory_shift_l2),
        "arrays": {"gt_xy": "(T,2)", "baseline_xy": "(T,2)", "guided_xy": "(T,2)", "T": int(n)},
    }
    meta_path = art_dir / f"{stem}_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return npz_path


def _format_notes(
    user_notes: str,
    *,
    pair_id: str,
    sweep_id: str | None = None,
    sweep_mode: str | None = None,
) -> str:
    parts = [
        user_notes.strip(),
        f"pair_id={pair_id}",
    ]
    if sweep_id:
        parts.append(f"sweep_id={sweep_id}")
    if sweep_mode:
        parts.append(f"sweep_mode={sweep_mode}")
    return " ".join(p for p in parts if p)


def _build_guidance_fn(
    clf: HeuristicBehaviorClassifier,
    target_class: int,
    scale: float,
    schedule: str,
    model: Alpamayo1_5,
) -> Any:
    base = classifier_gradient_guidance_fn(clf, target_class=target_class, scale=scale)
    return wrap_guidance_schedule(
        base,
        schedule=schedule,
        num_inference_steps=_inference_steps(model),
    )


def _print_metrics_line(prefix: str, metrics: dict[str, float]) -> None:
    print(
        f"{prefix}minADE={metrics['min_ade_m']:.4f} m  FDE={metrics['fde_m']:.4f} m  "
        f"mean_xy_step={metrics['mean_step_xy_m']:.4f} m  "
        f"mean_norm_accel(traj→action)={metrics['mean_norm_accel_from_pred_traj']:.4f}  "
        f"mean_norm_kappa(traj→action)={metrics['mean_norm_kappa_from_pred_traj']:.4f}"
    )


@dataclass
class PairSummary:
    label: str
    guidance_scale: float | None
    guidance_schedule: str
    d_min_ade: float
    d_fde: float


def _run_baseline_guided_pair(
    *,
    model: Alpamayo1_5,
    data: dict[str, Any],
    model_inputs: dict[str, Any],
    clf: HeuristicBehaviorClassifier,
    target_class: int,
    guidance_scale: float,
    guidance_schedule: str,
    pair_label: str,
    log_results: bool,
    save_artifacts: bool,
    results_dir: str,
    csv_name: str,
    notes: str,
    pair_id: str,
    sweep_id: str | None,
    sweep_mode: str | None,
    deterministic: bool,
    clip_id: str,
    seed: int,
) -> PairSummary:
    """One baseline + one guided rollout (RNG reset before each). Logs two records when requested."""
    _set_global_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_a, pred_rot_a, extra_a = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
            diffusion_kwargs=None,
        )
    metrics_a = _collect_row_metrics(model, data, model_inputs, pred_a, pred_rot_a)
    print(f"\n--- {pair_label} | baseline ---")
    print("CoC snippet:", extra_a["cot"][0])
    _print_metrics_line("", metrics_a)

    gn = _build_guidance_fn(clf, target_class, guidance_scale, guidance_schedule, model)
    _set_global_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_b, pred_rot_b, extra_b = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
            diffusion_kwargs={"denoising_guidance_fn": gn},
        )
    metrics_b = _collect_row_metrics(model, data, model_inputs, pred_b, pred_rot_b)
    print(f"--- {pair_label} | guided (schedule={guidance_schedule}, λ={guidance_scale}) ---")
    print("CoC snippet:", extra_b["cot"][0])
    _print_metrics_line("", metrics_b)

    d_ade = metrics_b["min_ade_m"] - metrics_a["min_ade_m"]
    d_fde = metrics_b["fde_m"] - metrics_a["fde_m"]
    d_step = metrics_b["mean_step_xy_m"] - metrics_a["mean_step_xy_m"]
    print("--- delta (guided - baseline) ---")
    print(f"ΔminADE={d_ade:+.4f} m  ΔFDE={d_fde:+.4f} m  Δmean_xy_step={d_step:+.4f} m")
    coc_match = str(extra_a["cot"]) == str(extra_b["cot"])
    if not coc_match:
        print(
            "Note: CoC strings differ between arms; use --deterministic for stricter repeatability."
        )

    merged_notes = _format_notes(
        notes, pair_id=pair_id, sweep_id=sweep_id, sweep_mode=sweep_mode
    )
    ts = datetime.now(timezone.utc).isoformat()
    # RMS XYZ distance guided vs baseline for *this* pair (stored on guided RunRecord only).
    shift_l2 = trajectory_shift_l2(pred_a, pred_b)

    if log_results:
        base_rec = RunRecord(
            run_id=new_run_id(),
            timestamp_iso=ts,
            arm="baseline",
            clip_id=clip_id,
            seed=seed,
            guidance_scale=None,
            target_class=target_class,
            guidance_schedule="none",
            deterministic=deterministic,
            coc_text=_coc_text(extra_a),
            coc_matched_baseline=None,
            min_ade=metrics_a["min_ade_m"],
            fde=metrics_a["fde_m"],
            trajectory_shift_l2=None,
            mean_xy_step_length=metrics_a["mean_step_xy_m"],
            mean_norm_accel_from_pred_traj=metrics_a["mean_norm_accel_from_pred_traj"],
            mean_norm_kappa_from_pred_traj=metrics_a["mean_norm_kappa_from_pred_traj"],
            notes=merged_notes,
        )
        guided_rec = RunRecord(
            run_id=new_run_id(),
            timestamp_iso=ts,
            arm="guided",
            clip_id=clip_id,
            seed=seed,
            guidance_scale=guidance_scale,
            target_class=target_class,
            guidance_schedule=guidance_schedule,
            deterministic=deterministic,
            coc_text=_coc_text(extra_b),
            coc_matched_baseline=coc_match,
            min_ade=metrics_b["min_ade_m"],
            fde=metrics_b["fde_m"],
            trajectory_shift_l2=shift_l2,
            mean_xy_step_length=metrics_b["mean_step_xy_m"],
            mean_norm_accel_from_pred_traj=metrics_b["mean_norm_accel_from_pred_traj"],
            mean_norm_kappa_from_pred_traj=metrics_b["mean_norm_kappa_from_pred_traj"],
            notes=merged_notes,
        )
        save_run_record(base_rec, results_dir=results_dir, csv_name=csv_name)
        save_run_record(guided_rec, results_dir=results_dir, csv_name=csv_name)

    if save_artifacts:
        ap = _save_pair_artifacts(
            results_dir=results_dir,
            pair_id=pair_id,
            clip_id=clip_id,
            seed=seed,
            guidance_scale=guidance_scale,
            guidance_schedule=guidance_schedule,
            coc_matched=coc_match,
            pair_label=pair_label,
            timestamp_iso=ts,
            sweep_id=sweep_id,
            sweep_mode=sweep_mode,
            min_ade_baseline=metrics_a["min_ade_m"],
            min_ade_guided=metrics_b["min_ade_m"],
            fde_baseline=metrics_a["fde_m"],
            fde_guided=metrics_b["fde_m"],
            trajectory_shift_l2=shift_l2,
            pred_baseline=pred_a,
            pred_guided=pred_b,
            data=data,
        )
        print(f"Saved trajectory artifact {ap} (+ meta JSON)")

    return PairSummary(
        label=pair_label,
        guidance_scale=guidance_scale,
        guidance_schedule=guidance_schedule,
        d_min_ade=d_ade,
        d_fde=d_fde,
    )


def _print_sweep_table(rows: list[PairSummary]) -> None:
    if not rows:
        return
    print("\n=== sweep summary (Δ guided − baseline) ===")
    w = max(len(r.label) for r in rows)
    print(f"{'case':<{w}}  {'sched':<6}  {'λ':>6}  {'ΔminADE':>10}  {'ΔFDE':>10}")
    for r in rows:
        lam = "" if r.guidance_scale is None else f"{r.guidance_scale:g}"
        print(
            f"{r.label:<{w}}  {r.guidance_schedule:<6}  {lam:>6}  "
            f"{r.d_min_ade:>+10.4f}  {r.d_fde:>+10.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline vs denoising guidance experiment runner")
    parser.add_argument("--clip-id", type=str, default="030c760c-ae38-49aa-9ad8-f5650a545d26")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=0.15, help="λ when not using --lambda-sweep.")
    parser.add_argument(
        "--target-class",
        type=int,
        default=0,
        help="Classifier label y for guided runs (0=yield-ish, 1=neutral, 2=accel-ish). "
        "Also stored on baseline rows as the paired experiment target (not inferred from data).",
    )
    parser.add_argument(
        "--guidance-schedule",
        type=str,
        default="all",
        choices=["all", "early", "late"],
        help="Guidance mask: all steps, first 40%% of Euler steps, or last 40%% (baseline logs as none).",
    )
    parser.add_argument(
        "--lambda-sweep",
        nargs="+",
        type=float,
        metavar="LAMBDA",
        default=None,
        help="Run baseline+guided for each λ (mutually exclusive with --schedule-sweep). "
        "Each pair uses a fresh seed reset and logs its own baseline row.",
    )
    parser.add_argument(
        "--schedule-sweep",
        action="store_true",
        help="Run baseline+guided for schedules all, early, late at fixed --guidance-scale "
        "(mutually exclusive with --lambda-sweep). Each schedule re-seeds and logs its own baseline row.",
    )
    parser.add_argument("--model", type=str, default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable cudnn deterministic mode (slower; improves repeatability).",
    )
    parser.add_argument(
        "--log-results",
        action="store_true",
        help="Append rows to results/csv/guidance_runs_v1.csv (default) and results/json/<run_id>.json per arm.",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Per pair, save results/artifacts/pair_<pair_id>.npz (gt_xy, baseline_xy, guided_xy) "
        "and pair_<pair_id>_meta.json for qualitative plotting.",
    )
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--csv-name", type=str, default="guidance_runs_v1.csv")
    parser.add_argument("--notes", type=str, default="", help="Optional free-text notes stored in each record.")
    args = parser.parse_args()

    if args.lambda_sweep is not None and args.schedule_sweep:
        parser.error("--lambda-sweep and --schedule-sweep are mutually exclusive")

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

    sweep_id: str | None = None
    sweep_mode: str | None = None
    summaries: list[PairSummary] = []

    if args.schedule_sweep:
        sweep_id = new_run_id()
        sweep_mode = "schedule_sweep"
        for sched in ("all", "early", "late"):
            pid = new_run_id()
            summ = _run_baseline_guided_pair(
                model=model,
                data=data,
                model_inputs=model_inputs,
                clf=clf,
                target_class=args.target_class,
                guidance_scale=args.guidance_scale,
                guidance_schedule=sched,
                pair_label=f"schedule={sched}",
                log_results=args.log_results,
                save_artifacts=args.save_artifacts,
                results_dir=args.results_dir,
                csv_name=args.csv_name,
                notes=args.notes,
                pair_id=pid,
                sweep_id=sweep_id,
                sweep_mode=sweep_mode,
                deterministic=args.deterministic,
                clip_id=args.clip_id,
                seed=args.seed,
            )
            summaries.append(summ)
        _print_sweep_table(summaries)
    elif args.lambda_sweep is not None:
        sweep_id = new_run_id()
        sweep_mode = "lambda_sweep"
        for lam in args.lambda_sweep:
            pid = new_run_id()
            summ = _run_baseline_guided_pair(
                model=model,
                data=data,
                model_inputs=model_inputs,
                clf=clf,
                target_class=args.target_class,
                guidance_scale=float(lam),
                guidance_schedule=args.guidance_schedule,
                pair_label=f"λ={lam:g}",
                log_results=args.log_results,
                save_artifacts=args.save_artifacts,
                results_dir=args.results_dir,
                csv_name=args.csv_name,
                notes=args.notes,
                pair_id=pid,
                sweep_id=sweep_id,
                sweep_mode=sweep_mode,
                deterministic=args.deterministic,
                clip_id=args.clip_id,
                seed=args.seed,
            )
            summaries.append(summ)
        _print_sweep_table(summaries)
    else:
        pid = new_run_id()
        _run_baseline_guided_pair(
            model=model,
            data=data,
            model_inputs=model_inputs,
            clf=clf,
            target_class=args.target_class,
            guidance_scale=args.guidance_scale,
            guidance_schedule=args.guidance_schedule,
            pair_label="single pair",
            log_results=args.log_results,
            save_artifacts=args.save_artifacts,
            results_dir=args.results_dir,
            csv_name=args.csv_name,
            notes=args.notes,
            pair_id=pid,
            sweep_id=None,
            sweep_mode=None,
            deterministic=args.deterministic,
            clip_id=args.clip_id,
            seed=args.seed,
        )

    if args.log_results:
        print(
            f"\nLogged append to {args.results_dir}/csv/{args.csv_name} "
            f"(and per-run JSON under {args.results_dir}/json/)."
        )


if __name__ == "__main__":
    main()
