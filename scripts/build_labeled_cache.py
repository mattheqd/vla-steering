#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream Physical AI clips, label them kinematically, cache normalized actions.

Needs the Physical AI AV dataset (gated on HuggingFace — run ``hf auth login``
first) and the Alpamayo-1.5 config, which supplies the action-space
normalization constants the model was trained on. The full 22 GB of weights
are NOT downloaded — only the config JSON.

Usage::

    python scripts/build_labeled_cache.py \\
        --clip-ids notebooks/clip_ids.parquet \\
        --n-t0-per-clip 5 \\
        --out results/labels/cache_v1.pt \\
        --resume
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import hydra.utils as hyu

from alpamayo1_5.config import Alpamayo1_5Config
from alpamayo1_5.data import build_labeled_cache, load_clip_ids


def _tqdm(total: int):
    try:
        from tqdm.auto import tqdm  # type: ignore[import-not-found]

        bar = tqdm(total=total, desc="clips")

        def _cb(done: int, n_total: int) -> None:
            bar.n = done
            bar.total = n_total
            bar.refresh()

        return _cb, bar
    except ImportError:  # pragma: no cover
        def _cb(done: int, n_total: int) -> None:
            if done % 10 == 0 or done == n_total:
                print(f"[build_labeled_cache] clip {done}/{n_total}")

        return _cb, None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip-ids", type=str, default="notebooks/clip_ids.parquet")
    p.add_argument("--out", type=str, default="results/labels/cache_v1.pt")
    p.add_argument("--n-t0-per-clip", type=int, default=5)
    p.add_argument("--min-gap-s", type=float, default=2.0)
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--model", type=str, default="nvidia/Alpamayo-1.5-10B")
    p.add_argument(
        "--identity-action-space",
        action="store_true",
        help="Skip model config download; use identity normalization (debug only).",
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
        logging.getLogger("alpamayo1_5").setLevel(logging.INFO)

    cids = load_clip_ids(args.clip_ids)
    print(f"[build_labeled_cache] {len(cids)} clip IDs from {args.clip_ids}")

    if args.identity_action_space:
        from alpamayo1_5.action_space.unicycle_accel_curvature import (
            UnicycleAccelCurvatureActionSpace,
        )

        action_space = UnicycleAccelCurvatureActionSpace()
        print(
            "[build_labeled_cache] WARNING: identity action-space normalization; "
            "cache will NOT match the trained expert's action distribution."
        )
    else:
        cfg = Alpamayo1_5Config.from_pretrained(args.model)
        action_space = hyu.instantiate(cfg.action_space_cfg)
        print(f"[build_labeled_cache] loaded action space from {args.model} config")

    progress, bar = _tqdm(min(len(cids), args.max_clips or len(cids)))
    try:
        build_labeled_cache(
            cids,
            action_space,
            out_path=args.out,
            n_t0_per_clip=args.n_t0_per_clip,
            min_gap_s=args.min_gap_s,
            resume=args.resume,
            max_clips=args.max_clips,
            progress=progress,
        )
    finally:
        if bar is not None:
            bar.close()

    sidecar = Path(args.out).with_suffix(".stats.json")
    if sidecar.is_file():
        print(f"[build_labeled_cache] stats → {sidecar}")
        print(sidecar.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
