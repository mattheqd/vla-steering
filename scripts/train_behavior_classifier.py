#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train a ``TrainedBehaviorClassifier`` on a labeled cache.

Consumes the ``.pt`` cache produced by ``scripts/build_labeled_cache.py``,
splits into stratified train/val, and fits a noise-conditioned classifier.
Writes a best + last checkpoint and a per-epoch training log CSV.

Usage::

    python scripts/train_behavior_classifier.py \\
        --cache results/labels/cache_v1.pt \\
        --epochs 50 --batch-size 256 --lr 3e-4 \\
        --downsample-majority \\
        --out results/checkpoints/behavior_clf_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from alpamayo1_5.data import NoisyActionDataset, load_labeled_cache, split_indices
from alpamayo1_5.steering import TrainedBehaviorClassifier

logger = logging.getLogger("train_behavior_classifier")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cosine_lr(epoch: int, total_epochs: int, lr_start: float, lr_end: float) -> float:
    if total_epochs <= 1:
        return lr_start
    t = epoch / max(1, total_epochs - 1)
    return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * t))


def _class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights normalized to mean 1 over present classes."""
    counts = torch.bincount(labels.long(), minlength=num_classes).float()
    w = torch.zeros(num_classes, dtype=torch.float32)
    nz = counts > 0
    w[nz] = 1.0 / counts[nz]
    if nz.any():
        w = w * (nz.sum().float() / w[nz].sum())
    return w


def _downsample_majority(
    cache: dict, *, cap_ratio: float = 2.0, seed: int = 0
) -> dict:
    """Cap the largest class at ``cap_ratio × second_largest`` count."""
    labels = cache["labels"]
    K = len(cache["label_names"])
    counts = torch.bincount(labels, minlength=K)
    if counts.numel() == 0 or int(counts.max().item()) == 0:
        return cache
    sorted_counts = torch.sort(counts, descending=True).values
    if sorted_counts.numel() < 2:
        return cache
    second = int(sorted_counts[1].item())
    cap = int(cap_ratio * max(second, 1))
    g = torch.Generator().manual_seed(seed)
    keep = torch.zeros(labels.shape[0], dtype=torch.bool)
    for k in range(K):
        idxs = (labels == k).nonzero(as_tuple=True)[0]
        n = idxs.numel()
        if n <= cap:
            keep[idxs] = True
            continue
        perm = torch.randperm(n, generator=g)[:cap]
        keep[idxs[perm]] = True
    out = {k: v for k, v in cache.items()}
    out["actions"] = cache["actions"][keep].contiguous()
    out["labels"] = labels[keep].contiguous()
    out["clip_ids"] = [cid for cid, k in zip(cache.get("clip_ids", []), keep.tolist()) if k]
    t0s = cache.get("t0s")
    if t0s is not None:
        out["t0s"] = t0s[keep].contiguous()
    return out


def _validate(
    model: TrainedBehaviorClassifier,
    cache: dict,
    val_idx: np.ndarray,
    *,
    fixed_ts: list[float],
    batch_size: int,
    device: torch.device,
    seed: int = 0,
) -> dict:
    model.eval()
    K = model.num_classes
    per_t: dict[str, float] = {}
    confusion = torch.zeros(K, K, dtype=torch.long)
    with torch.no_grad():
        for t_val in fixed_ts:
            ds = NoisyActionDataset(
                cache, indices=val_idx, mode="val", fixed_t=t_val, seed=seed
            )
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
            correct = total = 0
            for x_t, t, y in dl:
                x_t = x_t.to(device); t = t.to(device); y = y.to(device)
                logits = model(x_t, t)
                pred = logits.argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.numel()
                if abs(t_val - fixed_ts[-1]) < 1e-9:
                    for p_i, y_i in zip(pred.cpu().tolist(), y.cpu().tolist()):
                        confusion[y_i, p_i] += 1
            per_t[f"{t_val:.2f}"] = correct / max(1, total)
    mean_acc = float(np.mean(list(per_t.values())))
    return {"per_t_acc": per_t, "mean_acc": mean_acc, "confusion_t_last": confusion.tolist()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=str, required=True)
    p.add_argument("--out", type=str, default="results/checkpoints/behavior_clf_v1")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-end", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-classes", type=int, default=5)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--kernel", type=int, default=3)
    p.add_argument("--downsample-majority", action="store_true",
                   help="Cap largest class at cap-ratio × second-largest before training.")
    p.add_argument("--downsample-cap-ratio", type=float, default=2.0)
    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s [%(name)s] %(message)s",
    )
    _seed_everything(args.seed)

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    cache = load_labeled_cache(args.cache)
    print(f"[train] cache: {cache['labels'].shape[0]} rows, classes={cache['label_names']}")
    print(f"[train] per-class counts: {torch.bincount(cache['labels'], minlength=args.num_classes).tolist()}")

    if args.downsample_majority:
        before = cache["labels"].shape[0]
        cache = _downsample_majority(cache, cap_ratio=args.downsample_cap_ratio, seed=args.seed)
        after = cache["labels"].shape[0]
        print(f"[train] downsampled majority: {before} -> {after} rows")
        print(f"[train] post-downsample counts: {torch.bincount(cache['labels'], minlength=args.num_classes).tolist()}")

    n = cache["labels"].shape[0]
    if n < 50:
        raise RuntimeError(f"cache has only {n} rows after filtering — too small to train")

    train_idx, val_idx = split_indices(
        cache["labels"], val_frac=args.val_frac, seed=args.seed, stratified=True
    )
    print(f"[train] split (stratified): train={train_idx.size} val={val_idx.size}")
    val_class_counts = torch.bincount(cache["labels"][val_idx], minlength=args.num_classes).tolist()
    print(f"[train] val per-class:  {val_class_counts}")

    class_w = _class_weights(cache["labels"][train_idx], args.num_classes).to(args.device)
    print(f"[train] class weights: {[f'{w:.2f}' for w in class_w.tolist()]}")

    model = TrainedBehaviorClassifier(
        num_classes=args.num_classes,
        d_model=args.d_model,
        depth=args.depth,
        kernel=args.kernel,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_w)

    train_ds = NoisyActionDataset(cache, indices=train_idx, mode="train")
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )

    fixed_ts = [0.05, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 0.95]

    log_path = Path(str(out_stem) + ".train_log.csv")
    with log_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_mean_acc", "val_acc_at_t095", "lr"],
        ).writeheader()

    best_mean = -1.0
    best_epoch = -1
    since_improve = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        lr_now = _cosine_lr(epoch, args.epochs, args.lr, args.lr_end)
        for g in optim.param_groups:
            g["lr"] = lr_now

        total_loss = 0.0
        n_batches = 0
        for x_t, t, y in train_dl:
            x_t = x_t.to(args.device); t = t.to(args.device); y = y.to(args.device)
            logits = model(x_t, t)
            loss = loss_fn(logits, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_loss += float(loss.item())
            n_batches += 1
        train_loss = total_loss / max(1, n_batches)

        val = _validate(
            model, cache, val_idx, fixed_ts=fixed_ts,
            batch_size=args.batch_size, device=torch.device(args.device), seed=args.seed,
        )
        val_mean = val["mean_acc"]
        val_t095 = val["per_t_acc"][f"{0.95:.2f}"]
        print(
            f"epoch {epoch:03d}  lr={lr_now:.2e}  train_loss={train_loss:.4f}  "
            f"val_mean_acc={val_mean:.4f}  val@t=0.95={val_t095:.4f}"
        )
        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(
                f, fieldnames=["epoch", "train_loss", "val_mean_acc", "val_acc_at_t095", "lr"],
            ).writerow({
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_mean_acc": f"{val_mean:.6f}",
                "val_acc_at_t095": f"{val_t095:.6f}",
                "lr": f"{lr_now:.6e}",
            })
        history.append({"epoch": epoch, **val})

        extra = {
            "label_names": cache["label_names"],
            "normalization": cache.get("normalization", {}),
            "class_weights": class_w.detach().cpu().tolist(),
            "val_mean_acc": val_mean,
            "val_acc_at_t095": val_t095,
        }
        last_path = out_stem.parent / (out_stem.name + "_last.pt")
        model.save(last_path, extra=extra)

        if val_mean > best_mean:
            best_mean = val_mean
            best_epoch = epoch
            since_improve = 0
            best_path = out_stem.parent / (out_stem.name + "_best.pt")
            model.save(best_path, extra=extra)
        else:
            since_improve += 1
            if since_improve >= args.early_stop_patience:
                print(f"early stop at epoch {epoch} (no improvement in {since_improve})")
                break

    history_path = Path(str(out_stem) + ".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(
        f"\n[train] best val_mean_acc={best_mean:.4f} @ epoch {best_epoch}  "
        f"(history → {history_path})"
    )
    if best_mean < 0.70:
        print(
            "[train] WARNING: best val_mean_acc < 0.70 — the success gate is not met. "
            "Inspect class balance / cache size / augment with longer training."
        )


if __name__ == "__main__":
    main()
