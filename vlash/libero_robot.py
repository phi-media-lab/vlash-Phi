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

from dataclasses import dataclass, field

import numpy as np
from lerobot.cameras import CameraConfig
from lerobot.robots import Robot, RobotConfig


@CameraConfig.register_subclass("libero_camera")
@dataclass
class LiberoCameraConfig(CameraConfig):
    """Camera metadata for a LIBERO-backed simulator robot."""


@RobotConfig.register_subclass("libero_robot")
@dataclass
class LiberoRobotConfig(RobotConfig):
    """Robot config that wraps a single LIBERO environment as a Robot."""

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "image": LiberoCameraConfig(width=256, height=256, fps=30),
            "wrist_image": LiberoCameraConfig(width=256, height=256, fps=30),
        }
    )
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    episode_index: int = 0
    num_steps_wait: int = 10
    seed: int = 0


class LiberoRobot(Robot):
    """Expose a LIBERO simulator task through the LeRobot Robot interface."""

    config_class = LiberoRobotConfig
    name = "libero_robot"

    def __init__(self, config: LiberoRobotConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._env = None
        self._latest_observation = None
        self._state = np.zeros(8, dtype=np.float32)
        self._last_action = np.zeros(7, dtype=np.float32)
        self.cameras = config.cameras

    @property
    def observation_features(self) -> dict:
        features = {f"state_{index}": float for index in range(8)}
        for name, camera in self.cameras.items():
            features[name] = (camera.height, camera.width, 3)
        return features

    @property
    def action_features(self) -> dict:
        return {f"action_{index}": float for index in range(7)}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        from libero.libero import benchmark
        from lerobot.envs.libero import LiberoEnv

        task_suite = benchmark.get_benchmark_dict()[self.config.task_suite_name]()
        camera_names = ["agentview_image", "robot0_eye_in_hand_image"]
        camera_name_mapping = {
            "agentview_image": "image",
            "robot0_eye_in_hand_image": "wrist_image",
        }
        first_camera = next(iter(self.cameras.values()))

        self._env = LiberoEnv(
            task_suite=task_suite,
            task_id=self.config.task_id,
            task_suite_name=self.config.task_suite_name,
            camera_name=camera_names,
            camera_name_mapping=camera_name_mapping,
            obs_type="pixels_agent_pos",
            observation_width=first_camera.width,
            observation_height=first_camera.height,
            episode_index=self.config.episode_index,
            num_steps_wait=self.config.num_steps_wait,
        )
        self._latest_observation, _ = self._env.reset(seed=self.config.seed)
        self._sync_state_from_latest_observation()
        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def _sync_state_from_latest_observation(self) -> None:
        if self._latest_observation is None:
            return
        agent_pos = self._latest_observation.get("agent_pos")
        if agent_pos is not None:
            self._state = np.asarray(agent_pos, dtype=np.float32).copy()

    def get_observation(self) -> dict[str, float | np.ndarray]:
        if not self._connected or self._env is None or self._latest_observation is None:
            raise RuntimeError("LiberoRobot is not connected")

        observation: dict[str, float | np.ndarray] = {
            f"state_{index}": float(self._state[index]) for index in range(self._state.shape[0])
        }
        for name in self.cameras:
            observation[name] = np.ascontiguousarray(self._latest_observation["pixels"][name])
        return observation

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self._connected or self._env is None:
            raise RuntimeError("LiberoRobot is not connected")

        action_vector = np.array(
            [float(action[name]) for name in self.action_features],
            dtype=np.float32,
        )
        self._last_action = action_vector
        self._latest_observation, _, terminated, truncated, _ = self._env.step(action_vector)
        if terminated or truncated:
            self._latest_observation, _ = self._env.reset(seed=self.config.seed)
        self._sync_state_from_latest_observation()
        return action

    def disconnect(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
        self._connected = False
