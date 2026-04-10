#!/usr/bin/env python

# Copyright 2025 VLASH team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import asdict, dataclass


@dataclass
class StageTimingSummary:
    """Normalized stage timing payload shared by runtime and benchmarks."""

    prepare_mean_ms: float = 0.0
    prefix_mean_ms: float = 0.0
    suffix_mean_ms: float = 0.0
    total_mean_ms: float = 0.0


def summarize_stage_timings(
    prepare_total_s: float,
    prefix_total_s: float,
    suffix_total_s: float,
    total_total_s: float,
    samples: int,
) -> StageTimingSummary:
    """Convert cumulative seconds into a stable mean-ms payload."""
    if samples <= 0:
        return StageTimingSummary()

    return StageTimingSummary(
        prepare_mean_ms=round((prepare_total_s / samples) * 1000, 2),
        prefix_mean_ms=round((prefix_total_s / samples) * 1000, 2),
        suffix_mean_ms=round((suffix_total_s / samples) * 1000, 2),
        total_mean_ms=round((total_total_s / samples) * 1000, 2),
    )


def stage_timings_to_dict(summary: StageTimingSummary) -> dict[str, float]:
    """Serialize stage timings in a single canonical shape."""
    return asdict(summary)
