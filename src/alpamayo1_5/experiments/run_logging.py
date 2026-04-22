# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Append experiment run records to CSV and per-run JSON under ``results/``."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


@dataclass
class RunRecord:
    """One arm of an experiment (e.g. baseline or guided).

    ``target_class`` is the **paired** classifier label index used for the guided arm
    (``HeuristicBehaviorClassifier``). The same value is copied onto baseline rows for
    experiment traceability only—it is **not** inferred from the baseline trajectory.

    ``mean_norm_accel_from_pred_traj`` and ``mean_norm_kappa_from_pred_traj`` are means
    over waypoints of the **normalized** accel/κ obtained by ``traj_to_action`` on the
    **predicted** future polyline (inverse fit), not the denoiser's internal action tensor.

    ``guidance_schedule``: ``"none"`` for baseline; guided rows use ``"all"``, ``"early"``,
    or ``"late"`` (see ``wrap_guidance_schedule`` in ``alpamayo1_5.steering``).

    ``trajectory_shift_l2``: on guided rows, RMS L2 distance between that guided trajectory
    and the **baseline** trajectory from the **same** baseline+guided pair (``None`` on
    baseline rows).
    """

    run_id: str
    timestamp_iso: str
    arm: str
    clip_id: str
    seed: int
    guidance_scale: float | None
    target_class: int | None
    guidance_schedule: str
    deterministic: bool
    coc_text: str
    coc_matched_baseline: bool | None
    min_ade: float
    fde: float
    trajectory_shift_l2: float | None
    mean_xy_step_length: float
    mean_norm_accel_from_pred_traj: float
    mean_norm_kappa_from_pred_traj: float
    notes: str = ""

    def to_csv_row(self) -> dict[str, Any]:
        """Flat dict for ``csv.DictWriter`` (no nested values)."""
        out: dict[str, Any] = {}
        for k, v in asdict(self).items():
            if v is None:
                out[k] = ""
            elif isinstance(v, float) and math.isnan(v):
                out[k] = ""
            elif isinstance(v, bool):
                out[k] = "true" if v else "false"
            else:
                out[k] = v
        return out


CSV_COLUMNS: list[str] = [
    "run_id",
    "timestamp_iso",
    "arm",
    "clip_id",
    "seed",
    "guidance_scale",
    "target_class",
    "guidance_schedule",
    "deterministic",
    "coc_text",
    "coc_matched_baseline",
    "min_ade",
    "fde",
    "trajectory_shift_l2",
    "mean_xy_step_length",
    "mean_norm_accel_from_pred_traj",
    "mean_norm_kappa_from_pred_traj",
    "notes",
]


def save_run_record(
    record: RunRecord,
    *,
    results_dir: str | Path = "results",
    csv_name: str = "guidance_runs_v1.csv",
) -> Path:
    """Append one row to ``results/csv/<csv_name>`` and write ``results/json/<run_id>.json``.

    Creates parent directories as needed.

    Returns:
        Path to the written JSON file.
    """
    root = Path(results_dir)
    csv_dir = root / "csv"
    json_dir = root / "json"
    csv_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)

    csv_path = csv_dir / csv_name
    row = record.to_csv_row()
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_COLUMNS})

    json_path = json_dir / f"{record.run_id}.json"
    payload = _sanitize_for_json(asdict(record))

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    return json_path


def new_run_id() -> str:
    """Unique id for file names and CSV keys."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:10]
