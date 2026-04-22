# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weak behavior labels for training the denoising-time classifier."""

from alpamayo1_5.labels.kinematic_labels import (
    LABEL_NAMES,
    NUM_CLASSES,
    label_from_future,
    signed_heading_change_from_rot,
    signed_heading_change_from_xy,
)

__all__ = [
    "LABEL_NAMES",
    "NUM_CLASSES",
    "label_from_future",
    "signed_heading_change_from_rot",
    "signed_heading_change_from_xy",
]
