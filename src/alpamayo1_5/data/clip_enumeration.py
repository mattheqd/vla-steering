# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clip-ID enumeration and ``t0_us`` grid sampling for the labeled cache.

The canonical clip list is ``notebooks/clip_ids.parquet`` shipped with the
model release; we read it once and produce a flat list of string IDs.
"""

from __future__ import annotations

from pathlib import Path


def load_clip_ids(parquet_path: str | Path = "notebooks/clip_ids.parquet") -> list[str]:
    """Return the list of clip IDs from a parquet file.

    The parquet is expected to contain at least one string column of clip IDs
    (first string column wins). Works with pyarrow or pandas installed.
    """
    p = Path(parquet_path)
    if not p.is_file():
        raise FileNotFoundError(f"Clip-ID parquet not found: {p}")
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]

        table = pq.read_table(str(p))
        columns = table.column_names
        for name in columns:
            col = table.column(name)
            if col.type in (None,):
                continue
            if str(col.type).startswith("string") or str(col.type) == "large_string":
                return [str(x) for x in col.to_pylist()]
        # Fallback: first column, cast to str
        first = table.column(columns[0]).to_pylist()
        return [str(x) for x in first]
    except ImportError:  # pragma: no cover — fallback path
        import pandas as pd  # type: ignore[import-not-found]

        df = pd.read_parquet(str(p))
        for name in df.columns:
            if df[name].dtype == object:
                return [str(x) for x in df[name].tolist()]
        return [str(x) for x in df.iloc[:, 0].tolist()]


def sample_t0_us_grid(
    clip_duration_us: int,
    *,
    n: int = 5,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
    min_gap_s: float = 2.0,
) -> list[int]:
    """Evenly-spaced ``t0_us`` samples within the valid window of a clip.

    The valid window mirrors the precondition in
    :func:`load_physical_aiavdataset`: history needs ``num_history_steps``
    samples *before* ``t0``, and future needs ``num_future_steps`` samples
    *after*, all at ``time_step`` seconds apart.

    Args:
        clip_duration_us: Total clip length in microseconds.
        n: Requested number of ``t0_us`` samples.
        num_history_steps, num_future_steps, time_step: Mirror the loader.
        min_gap_s: Minimum spacing between consecutive ``t0_us`` samples.

    Returns:
        Sorted list of up to ``n`` integer microsecond timestamps. Returns
        an empty list if the clip is too short for even one valid sample.
    """
    if clip_duration_us <= 0 or n <= 0:
        return []
    step_us = int(time_step * 1_000_000)
    hist_us = num_history_steps * step_us
    fut_us = num_future_steps * step_us
    # Strict > in the loader assert: t0 > num_history_steps * dt * 1e6
    lo = hist_us + 1
    hi = clip_duration_us - fut_us
    if hi <= lo:
        return []
    gap_us = int(min_gap_s * 1_000_000)
    max_by_gap = max(1, (hi - lo) // max(gap_us, 1) + 1)
    k = min(n, max_by_gap)
    if k == 1:
        return [(lo + hi) // 2]
    stride = (hi - lo) // (k - 1)
    return [int(lo + i * stride) for i in range(k)]
