# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
from pathlib import Path

from alpamayo1_5.experiments.run_logging import RunRecord, save_run_record


def test_save_run_record_csv_and_json(tmp_path: Path) -> None:
    rec = RunRecord(
        run_id="test_run_001",
        timestamp_iso="2026-01-01T00:00:00+00:00",
        arm="baseline",
        clip_id="clip",
        seed=1,
        guidance_scale=None,
        target_class=None,
        guidance_schedule="none",
        deterministic=False,
        coc_text="hello",
        coc_matched_baseline=None,
        min_ade=0.1,
        fde=0.2,
        trajectory_shift_l2=None,
        mean_xy_step_length=0.5,
        mean_norm_accel_from_pred_traj=0.0,
        mean_norm_kappa_from_pred_traj=0.0,
        notes="unit",
    )
    json_path = save_run_record(rec, results_dir=tmp_path)
    assert json_path.exists()
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    assert data["arm"] == "baseline"
    assert data["trajectory_shift_l2"] is None

    csv_path = tmp_path / "csv" / "guidance_runs_v1.csv"
    assert csv_path.exists()
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["run_id"] == "test_run_001"
