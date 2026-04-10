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


@CameraConfig.register_subclass("mock")
@dataclass
class MockCameraConfig(CameraConfig):
    """Camera stub for local dry-run inference."""


@RobotConfig.register_subclass("mock_robot")
@dataclass
class MockRobotConfig(RobotConfig):
    """Robot config for local `vlash run` validation without hardware."""

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    observation_state_dim: int = 8
    action_dim: int = 7


class MockRobot(Robot):
    """Minimal Robot implementation for validating the `vlash run` path."""

    config_class = MockRobotConfig
    name = "mock_robot"

    def __init__(self, config: MockRobotConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._state = np.zeros(self.config.observation_state_dim, dtype=np.float32)
        self._last_action = np.zeros(self.config.action_dim, dtype=np.float32)
        self.cameras = config.cameras

    @property
    def observation_features(self) -> dict:
        features = {
            f"state_{index}": float for index in range(self.config.observation_state_dim)
        }
        for name, camera in self.cameras.items():
            features[name] = (camera.height, camera.width, 3)
        return features

    @property
    def action_features(self) -> dict:
        return {f"action_{index}": float for index in range(self.config.action_dim)}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def get_observation(self) -> dict[str, float | np.ndarray]:
        if not self._connected:
            raise RuntimeError("MockRobot is not connected")

        observation: dict[str, float | np.ndarray] = {
            name: float(self._state[index])
            for index, name in enumerate(self.observation_features)
            if not isinstance(self.observation_features[name], tuple)
        }
        for name, camera in self.cameras.items():
            frame = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
            frame[..., 0] = 32
            frame[..., 1] = 64
            frame[..., 2] = 96
            observation[name] = frame

        return observation

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self._connected:
            raise RuntimeError("MockRobot is not connected")

        action_vector = np.array(
            [float(action[name]) for name in self.action_features],
            dtype=np.float32,
        )
        self._last_action = action_vector
        self._state[: self.config.action_dim] = action_vector
        return action

    def disconnect(self) -> None:
        self._connected = False
