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

"""Compare the active PI05 suffix backend against the local fallback backend.

This is a prototype-focused correctness check. It reuses the same observation
construction path as `vlash run`, then compares the current suffix backend
output against the built-in local suffix backend on a single observation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import draccus
import torch

from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features

from vlash.configs import RunConfig
from vlash.policies.factory import get_policy_class
from vlash.run import VLASHAsyncManager, make_runtime_robot
from vlash.utils import prepare_observation_for_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to a vlash run config")
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write comparison payload as JSON",
    )
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args


def main() -> None:
    args = parse_args()
    cfg = draccus.parse(RunConfig, config_path=args.config_path, args=args.overrides)

    policy_cls = get_policy_class(cfg.policy.type)
    policy = policy_cls.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)

    robot = make_runtime_robot(cfg.robot)
    robot.connect()
    try:
        raw_obs = robot.get_observation()
        dataset_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
        frame = build_dataset_frame(dataset_features, raw_obs, prefix="observation")

        manager = VLASHAsyncManager(
            policy=policy,
            robot=robot,
            single_task=cfg.single_task,
            overlap_steps=cfg.inference_overlap_steps * cfg.action_quant_ratio,
            image_feature_map=cfg.camera_feature_map,
        )

        observation, _ = manager.prepare_inference_observation(frame)
        observation = prepare_observation_for_inference(
            observation,
            torch.device(cfg.policy.device),
            cfg.single_task,
            robot.robot_type,
        )

        prefix_context = policy.build_prefix_context(observation)
        comparison = policy.compare_suffix_backend(prefix_context)

        payload = {
            "backend": policy.get_suffix_backend_name(),
            "comparison": asdict(comparison),
        }
        print(json.dumps(payload, indent=2))

        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
