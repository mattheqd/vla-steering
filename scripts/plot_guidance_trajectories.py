#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a 3-panel XY trajectory figure from ``compare_denoising_guidance.py --save-artifacts`` output.

Loads ``pair_<id>.npz`` (``gt_xy``, ``baseline_xy``, ``guided_xy``) and optional sibling
``pair_<id>_meta.json`` for titles and quantitative captions.

Example::

    python scripts/plot_guidance_trajectories.py --npz results/artifacts/pair_<id>.npz
    python scripts/plot_guidance_trajectories.py --npz results/artifacts/pair_<id>.npz \\
        --output results/figures/pair_<id>.png
    python scripts/plot_guidance_trajectories.py --npz .../pair_<id>.npz --aspect-mode equal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def _load_meta(npz_path: Path) -> dict:
    meta_path = npz_path.with_name(npz_path.stem + "_meta.json")
    if meta_path.is_file():
        with meta_path.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


def _apply_pub_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "legend.fontsize": 9.5,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _coc_label(meta: dict) -> str:
    v = meta.get("coc_matched_baseline")
    if v is True:
        return "yes"
    if v is False:
        return "no"
    return "?"


def _fmt_title_line1(meta: dict) -> str:
    clip = meta.get("clip_id", "?")
    lam = meta.get("guidance_scale", "?")
    sched = meta.get("guidance_schedule", "?")
    return f"{clip} | λ={lam} | {sched} | CoC {_coc_label(meta)}"


def _fmt_title_line2(meta: dict) -> str | None:
    keys = (
        "min_ade_baseline",
        "min_ade_guided",
        "fde_baseline",
        "fde_guided",
        "trajectory_shift_l2",
    )
    if not all(k in meta for k in keys):
        return None
    return (
        f"minADE {meta['min_ade_baseline']:.3f}→{meta['min_ade_guided']:.3f} m | "
        f"FDE {meta['fde_baseline']:.3f}→{meta['fde_guided']:.3f} m | "
        f"shift L2 {meta['trajectory_shift_l2']:.3f} m"
    )


def _readable_xy_limits(*arrays: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis limits for *readable* mode (auto aspect, not 1:1).

    Let ``P`` be all points from ``gt_xy``, ``baseline_xy``, and ``guided_xy`` stacked
    ``(N, 2)``. Let ``xmin, xmax, ymin, ymax`` be component-wise min/max over ``P``,
    ``sx = xmax - xmin``, ``sy = ymax - ymin`` (floored at ``1e-9``).

    * **x-limits:** ``[xmin - 0.06·sx, xmax + 0.06·sx]``
    * **y-limits (tighter):** ``[ymin - 0.045·sy, ymax + 0.045·sy]``

    Margins are in **data units** (meters), proportional to each axis span only.
    """
    stacked = np.vstack(arrays)
    xmin, ymin = stacked.min(axis=0)
    xmax, ymax = stacked.max(axis=0)
    sx = max(float(xmax - xmin), 1e-9)
    sy = max(float(ymax - ymin), 1e-9)
    mx = 0.06 * sx
    my = 0.045 * sy
    return (xmin - mx, xmax + mx), (ymin - my, ymax + my)


def _equal_xy_limits(*arrays: np.ndarray, margin: float = 0.08) -> tuple[float, float, float, float]:
    """Symmetric padding on x and y from each span (for ``--aspect-mode equal``)."""
    stacked = np.vstack(arrays)
    xmin, ymin = stacked.min(axis=0)
    xmax, ymax = stacked.max(axis=0)
    sx = max(float(xmax - xmin), 1e-9)
    sy = max(float(ymax - ymin), 1e-9)
    mx = margin * sx
    my = margin * sy
    return xmin - mx, xmax + mx, ymin - my, ymax + my


def _plot_traj_marked(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    label: str | None,
    zorder: int = 2,
) -> None:
    ax.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
        zorder=zorder,
        solid_capstyle="round",
    )
    ax.scatter(
        x[0],
        y[0],
        s=70,
        marker="o",
        facecolors=color,
        edgecolors="white",
        linewidths=1.2,
        zorder=zorder + 2,
    )
    ax.scatter(
        x[-1],
        y[-1],
        s=70,
        marker="s",
        facecolors=color,
        edgecolors="white",
        linewidths=1.2,
        zorder=zorder + 2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz",
        type=Path,
        required=True,
        help="Path to results/artifacts/pair_<pair_id>.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG (default: results/figures/<npz stem>.png under cwd)",
    )
    parser.add_argument(
        "--aspect-mode",
        choices=("readable", "equal"),
        default="readable",
        help="readable (default): auto aspect, tighter y-limits, sparse y-ticks. "
        "equal: 1 data unit on x equals 1 on y (geometry-faithful).",
    )
    args = parser.parse_args()
    npz_path = args.npz.resolve()
    if not npz_path.is_file():
        raise SystemExit(f"not found: {npz_path}")

    z = np.load(npz_path)
    gt = np.asarray(z["gt_xy"], dtype=np.float64)
    base = np.asarray(z["baseline_xy"], dtype=np.float64)
    guided = np.asarray(z["guided_xy"], dtype=np.float64)
    meta = _load_meta(npz_path)

    out = args.output
    if out is None:
        out = Path.cwd() / "results" / "figures" / f"{npz_path.stem}.png"
    else:
        out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    _apply_pub_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 6.4), layout="constrained")
    assert isinstance(axes, np.ndarray)
    ax0, ax1, ax2 = axes[0], axes[1], axes[2]

    if args.aspect_mode == "readable":
        xlim, ylim = _readable_xy_limits(gt, base, guided)
    else:
        xl0, xl1, yl0, yl1 = _equal_xy_limits(gt, base, guided)
        xlim, ylim = (xl0, xl1), (yl0, yl1)

    c_gt = "#000000"
    c_base = "#0072B2"
    c_guided = "#D55E00"
    lw_gt = 2.8
    lw_pred = 2.5

    _plot_traj_marked(
        ax0, gt[:, 0], gt[:, 1], color=c_gt, linestyle="--", linewidth=lw_gt, label="Ground truth"
    )
    _plot_traj_marked(
        ax0, base[:, 0], base[:, 1], color=c_base, linestyle="-", linewidth=lw_pred, label="Baseline"
    )
    ax0.set_title("Baseline vs ground truth")
    ax0.set_xlabel("x (m)")
    ax0.set_ylabel("y (m)")
    ax0.legend(
        loc="upper left",
        fontsize=7.5,
        labelspacing=0.35,
        handlelength=1.6,
        frameon=True,
        fancybox=False,
        edgecolor="0.85",
        framealpha=0.92,
    )

    _plot_traj_marked(
        ax1, gt[:, 0], gt[:, 1], color=c_gt, linestyle="--", linewidth=lw_gt, label="Ground truth"
    )
    _plot_traj_marked(
        ax1,
        guided[:, 0],
        guided[:, 1],
        color=c_guided,
        linestyle="-",
        linewidth=lw_pred,
        label="Guided",
    )
    ax1.set_title("Guided vs ground truth")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.legend(
        loc="upper right",
        fontsize=7.5,
        labelspacing=0.35,
        handlelength=1.6,
        frameon=True,
        fancybox=False,
        edgecolor="0.85",
        framealpha=0.92,
    )

    _plot_traj_marked(
        ax2, gt[:, 0], gt[:, 1], color=c_gt, linestyle="--", linewidth=lw_gt, label="Ground truth"
    )
    _plot_traj_marked(
        ax2, base[:, 0], base[:, 1], color=c_base, linestyle="-", linewidth=lw_pred, label="Baseline"
    )
    _plot_traj_marked(
        ax2,
        guided[:, 0],
        guided[:, 1],
        color=c_guided,
        linestyle="-",
        linewidth=lw_pred,
        label="Guided",
    )
    ax2.set_title("Baseline vs guided (+ GT)")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.legend(
        loc="lower left",
        fontsize=7.5,
        labelspacing=0.35,
        handlelength=1.6,
        frameon=True,
        fancybox=False,
        edgecolor="0.85",
        framealpha=0.92,
    )

    for ax in (ax0, ax1, ax2):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if args.aspect_mode == "readable":
            ax.set_aspect("auto")
            # ~3–4 major y-ticks: cap interval count; Matplotlib picks “nice” step sizes.
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3, prune=None))
        else:
            ax.set_aspect("equal", adjustable="box")

    line1 = _fmt_title_line1(meta)
    line2 = _fmt_title_line2(meta)
    fig.suptitle(
        line1 + ("\n" + line2 if line2 else ""),
        fontsize=10.5,
        x=0.5,
        y=1.02,
        ha="center",
        va="bottom",
    )

    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
