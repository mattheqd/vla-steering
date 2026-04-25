#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot the per-``t`` accuracy curve and a confusion matrix for a trained classifier.

Usage::

    python scripts/eval_behavior_classifier.py \\
        --ckpt results/checkpoints/behavior_clf_v1_best.pt \\
        --cache results/labels/cache_v1.pt \\
        --out results/figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from alpamayo1_5.data import NoisyActionDataset, load_labeled_cache, split_indices
from alpamayo1_5.steering import TrainedBehaviorClassifier


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--cache", type=str, required=True)
    p.add_argument("--out", type=str, default="results/figures")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = load_labeled_cache(args.cache)
    n = cache["labels"].shape[0]
    _, val_idx = split_indices(cache["labels"], val_frac=args.val_frac, seed=args.seed, stratified=True)

    model = TrainedBehaviorClassifier.from_pretrained(args.ckpt, map_location=args.device)
    model.to(args.device).eval()

    fixed_ts = np.linspace(0.05, 0.95, 19).tolist()
    per_t_acc = []
    K = model.num_classes
    confusion_end = torch.zeros(K, K, dtype=torch.long)

    with torch.no_grad():
        for t_val in fixed_ts:
            ds = NoisyActionDataset(
                cache, indices=val_idx, mode="val", fixed_t=float(t_val), seed=args.seed
            )
            dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size)
            correct = total = 0
            for x_t, t, y in dl:
                x_t = x_t.to(args.device); t = t.to(args.device); y = y.to(args.device)
                logits = model(x_t, t)
                pred = logits.argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.numel()
                if abs(t_val - fixed_ts[-1]) < 1e-9:
                    for p_i, y_i in zip(pred.cpu().tolist(), y.cpu().tolist()):
                        confusion_end[y_i, p_i] += 1
            per_t_acc.append(correct / max(1, total))

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise SystemExit(f"matplotlib required for eval script: {e}")

    fig, ax = plt.subplots(figsize=(6.0, 4.0), layout="constrained")
    ax.plot(fixed_ts, per_t_acc, marker="o")
    ax.axhline(0.70, color="gray", linestyle="--", linewidth=0.8, label="0.70 gate")
    ax.set_xlabel("noise level t")
    ax.set_ylabel("val accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Behavior classifier accuracy vs t  (mean={np.mean(per_t_acc):.3f})")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    acc_path = out_dir / (Path(args.ckpt).stem + "_acc_vs_t.png")
    fig.savefig(acc_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {acc_path}")

    fig, ax = plt.subplots(figsize=(5.5, 5.0), layout="constrained")
    conf_np = confusion_end.float().numpy()
    row_sums = conf_np.sum(axis=1, keepdims=True)
    conf_norm = conf_np / np.maximum(row_sums, 1)
    im = ax.imshow(conf_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046)
    names = cache["label_names"]
    ax.set_xticks(range(K), labels=names, rotation=35, ha="right")
    ax.set_yticks(range(K), labels=names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(K):
        for j in range(K):
            ax.text(
                j, i, f"{conf_norm[i, j]:.2f}", ha="center", va="center",
                color="black" if conf_norm[i, j] < 0.5 else "white", fontsize=9,
            )
    ax.set_title(f"confusion @ t={fixed_ts[-1]:.2f}")
    conf_path = out_dir / (Path(args.ckpt).stem + "_confusion.png")
    fig.savefig(conf_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {conf_path}")

    print(f"mean acc over {len(fixed_ts)} noise levels: {np.mean(per_t_acc):.4f}")
    print(f"acc @ t={fixed_ts[-1]:.2f}: {per_t_acc[-1]:.4f}")


if __name__ == "__main__":
    main()
