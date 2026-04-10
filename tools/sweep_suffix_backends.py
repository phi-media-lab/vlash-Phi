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

"""Sweep PI05 suffix prototype backends and aggregate correctness/runtime stats."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


DEFAULT_BACKENDS = [
    "local",
    "serialized_local",
    "numpy_local",
    "dispatched_numpy_local",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to a vlash run config")
    parser.add_argument(
        "--output-dir",
        default="outputs/libero_runtime/p3_backend_sweep",
        help="Directory for per-backend JSON outputs and summaries",
    )
    parser.add_argument(
        "--backend",
        action="append",
        dest="backends",
        default=[],
        help="Suffix backend to include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--run-wrapper",
        default="tools/run_rocm_vlash.sh",
        help="Wrapper used to launch `vlash run` under the intended runtime environment.",
    )
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args


def run_command(cmd: list[str], *, env: dict[str, str]) -> None:
    subprocess.run(cmd, check=True, env=env)


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_summary(summary_rows: list[dict], output_dir: Path) -> None:
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    lines = [
        "# Suffix Backend Sweep",
        "",
        "| backend | max_abs_diff | mean_abs_diff | loop_avg_ms | stage_total_ms | suffix_mean_ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['backend']} | {row['max_abs_diff']:.6f} | {row['mean_abs_diff']:.6f} | "
            f"{row['loop_avg_ms']:.2f} | {row['stage_total_ms']:.2f} | {row['suffix_mean_ms']:.2f} |"
        )
    lines.append("")

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backends = args.backends or DEFAULT_BACKENDS
    env = os.environ.copy()
    rocm_root = env.get("ROCM_ROOT", "/opt/rocm-7.2.1")
    ld_library_path = env.get("LD_LIBRARY_PATH", "")
    rocm_ld_parts = [f"{rocm_root}/lib", "/opt/rocm/lib"]
    if ld_library_path:
        rocm_ld_parts.append(ld_library_path)
    env["LD_LIBRARY_PATH"] = ":".join(rocm_ld_parts)
    wrapper = Path(args.run_wrapper)

    summary_rows: list[dict] = []
    for backend in backends:
        comparison_path = output_dir / f"{backend}_comparison.json"
        runtime_path = output_dir / f"{backend}_runtime.json"

        check_cmd = [
            env.get("VENV_DIR", "/home/amd/.venvs/vlash-rocm") + "/bin/python",
            "tools/check_suffix_backend.py",
            args.config_path,
            "--output-json",
            str(comparison_path),
            f"--policy.suffix_backend={backend}",
            *args.overrides,
        ]
        run_command(check_cmd, env=env)

        run_cmd = [
            "bash",
            str(wrapper),
            "run",
            args.config_path,
            f"--policy.suffix_backend={backend}",
            "--suffix_backend_check=true",
            f"--runtime_stats_output={runtime_path}",
            *args.overrides,
        ]
        run_command(run_cmd, env=env)

        comparison = load_json(comparison_path)
        runtime = load_json(runtime_path)
        summary_rows.append(
            {
                "backend": backend,
                "max_abs_diff": float(comparison["comparison"]["max_abs_diff"]),
                "mean_abs_diff": float(comparison["comparison"]["mean_abs_diff"]),
                "loop_avg_ms": float(runtime["timings_ms"]["loop_avg"]),
                "stage_total_ms": float(runtime["stage_timings_ms"]["total_mean_ms"]),
                "suffix_mean_ms": float(runtime["stage_timings_ms"]["suffix_mean_ms"]),
            }
        )

    write_summary(summary_rows, output_dir)


if __name__ == "__main__":
    main()
