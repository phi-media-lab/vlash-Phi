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
"""VLASH Robot Inference Module.

This module implements real-time robot control using trained VLA policies
with VLASH's asynchronous inference strategy. The key innovation is
future-state-aware prediction that overlaps inference with execution.

Key components:
- VLASHAsyncManager: Manages asynchronous action chunk execution
- run_loop: Core control loop for real-time robot operation
- run: Main entry point for inference

Usage:
    vlash run examples/inference/async.yaml
"""

import logging
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from copy import copy
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.robots import Robot, make_robot_from_config
from lerobot.utils.constants import OBS_IMAGES
from lerobot.utils.control_utils import (
    init_keyboard_listener,
)

from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from vlash.configs import RunConfig
from vlash.libero_robot import LiberoRobot, LiberoRobotConfig
from vlash.mock_robot import MockRobot, MockRobotConfig
from vlash.policies.factory import get_policy_class
from vlash.runtime_stats import stage_timings_to_dict, summarize_stage_timings
from vlash.utils import prepare_observation_for_inference


@dataclass
class ObservationSlot:
    """Runtime container for the latest captured observation frame."""

    frame: dict | None = None
    captured_at: float = 0.0


@dataclass
class ChunkSlot:
    """Runtime container for action chunks handed between inference and execution."""

    actions_cpu: np.ndarray | None = None
    actions_gpu: torch.Tensor | None = None
    produced_at: float = 0.0


@dataclass
class InferenceTimings:
    """Per-phase timings for one inference launch."""

    observation_prepare_s: float = 0.0
    prefix_build_s: float = 0.0
    suffix_rollout_s: float = 0.0
    total_s: float = 0.0


@dataclass
class InferenceArtifacts:
    """Runtime container for staged inference state."""

    prefix_context: object | None = None
    timings: InferenceTimings | None = None
    future_state_mode: str = "none"


@dataclass
class RuntimeStats:
    """Aggregate runtime profiling across the control loop."""

    observation_fetches: int = 0
    observation_fetch_s: float = 0.0
    inference_launches: int = 0
    chunk_switches: int = 0
    chunk_handoff_s: float = 0.0
    loop_iterations: int = 0
    loop_s: float = 0.0
    last_future_state_mode: str = "none"
    observation_prepare_s: float = 0.0
    prefix_build_s: float = 0.0
    suffix_rollout_s: float = 0.0
    inference_total_s: float = 0.0


def normalize_image_feature_name(name: str) -> str:
    """Normalize short camera names into observation image feature names."""
    return name if name.startswith(f"{OBS_IMAGES}.") else f"{OBS_IMAGES}.{name}"


def apply_policy_image_mapping(
    observation: dict[str, np.ndarray],
    image_feature_map: dict[str, str],
) -> dict[str, np.ndarray]:
    """Copy robot image observations into the keys expected by the policy."""
    if not image_feature_map:
        return observation

    mapped = copy(observation)
    for target_key, source_key in image_feature_map.items():
        if source_key not in mapped:
            raise ValueError(
                f"Mapped source image feature {source_key!r} is missing from observation "
                f"(available keys: {sorted(mapped.keys())})"
            )
        mapped[target_key] = mapped[source_key]

    return mapped


class VLASHAsyncManager:
    """Manages asynchronous action chunk execution for VLASH inference.
    
    This class implements the core VLASH async inference strategy:
    1. Execute actions from the current chunk while preparing the next
    2. Use future state awareness by conditioning on predicted end state
    3. Overlap inference with execution to hide latency
    
    The execution timeline looks like:
    
        Chunk N:     [action_0, action_1, ..., action_{n-overlap}, ..., action_{n-1}]
                                                    ^
                                                    |-- Start inference for Chunk N+1
                                                        (using predicted state at action_{n-1})
        Chunk N+1:   [action_0, action_1, ...]
                     ^
                     |-- Switch to new chunk when Chunk N completes
    
    Attributes:
        policy: The trained policy for action prediction.
        robot: The robot being controlled.
        single_task: task description for policies.
        n_action_steps: Number of actions per chunk.
        overlap_steps: Steps before chunk end to start next inference.
        current_chunk: Currently executing action chunk (numpy array).
        next_chunk: Pre-computed next chunk (torch tensor, pending transfer).
        chunk_index: Current position within the executing chunk.
        device: Torch device for inference.
    """
    
    def __init__(
        self,
        policy: PreTrainedPolicy,
        robot: Robot,
        single_task: str | None,
        overlap_steps: int,
        image_feature_map: dict[str, str] | None = None,
    ):
        """Initialize the async manager.
        
        Args:
            policy: Trained policy for action prediction.
            robot: Robot instance to control.
            single_task: Task description string.
            overlap_steps: Number of steps before chunk end to start next inference.
                          Higher values give more time for inference but reduce
                          the accuracy of the prediction.
        """
        self.policy = policy
        self.robot = robot
        self.single_task = single_task
        self.n_action_steps = policy.config.n_action_steps
        self.overlap_steps = overlap_steps
        self.image_feature_map = image_feature_map or {}
        
        # Chunk state management
        self.current_chunk_slot = ChunkSlot()  # Currently executing chunk on CPU
        self.next_chunk_slot = ChunkSlot()     # Pre-computed chunk on GPU
        self.current_inference_artifacts = InferenceArtifacts()
        self.next_inference_artifacts = InferenceArtifacts()
        self.runtime_stats = RuntimeStats()
        self.chunk_index = 0  # Position within current chunk
        
        self.device = get_safe_torch_device(policy.config.device)

        # Validate configuration
        assert (
            self.n_action_steps >= self.overlap_steps
        ), "n_action_steps must be greater than or equal to overlap_steps"
        assert self.overlap_steps >= 0, "overlap_steps must be non-negative"

    def is_running(self) -> bool:
        """Check if the manager has any chunks to execute.
        
        Returns:
            True if there's a current or pending chunk, False otherwise.
        """
        return (self.current_chunk_slot.actions_cpu is not None) or (self.next_chunk_slot.actions_gpu is not None)

    def should_switch_chunk(self) -> bool:
        """Check if it's time to switch to the next chunk.
        
        Returns:
            True if at the beginning of a new chunk cycle (index == 0).
        """
        return self.chunk_index == 0

    def should_launch_next_inference(self) -> bool:
        """Check if it's time to start computing the next chunk.
        
        The next inference is launched `overlap_steps` before the current
        chunk ends, allowing inference to happen in parallel with execution.
        
        Returns:
            True if at the trigger point for next inference.
        """
        return self.chunk_index == self.n_action_steps - self.overlap_steps

    def should_fetch_observation(self) -> bool:
        """Check if a fresh observation is needed.
        
        Observations are fetched:
        1. At startup (not running yet)
        2. When launching next inference (need current state)
        
        Returns:
            True if observation should be captured this step.
        """
        return (not self.is_running()) or self.should_launch_next_inference()

    def get_current_action(self) -> dict[str, float]:
        """Extract the current action from the executing chunk.
        
        Returns:
            Dictionary mapping action feature names to values.
            
        Raises:
            RuntimeError: If no chunk is currently executing.
        """
        if self.current_chunk_slot.actions_cpu is None:
            raise RuntimeError("No chunk is currently executing")

        # Get action values at current index and map to feature names
        action_values = self.current_chunk_slot.actions_cpu[self.chunk_index]
        action = {key: action_values[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def launch_next_inference(self, observation: dict[str, np.ndarray]) -> torch.Tensor:
        """Compute the next action chunk using the policy.
        
        Implements future state awareness: if we have a current chunk,
        use its final action as the observation state (predicting where
        the robot will be when this chunk finishes).
        
        Args:
            observation: Current observation dictionary.
            
        Returns:
            Predicted action chunk as a torch tensor [n_action_steps, action_dim].
        """
        observation, _ = self.prepare_inference_observation(observation)

        with torch.inference_mode():
            # Prepare observation: convert images to CHW format, normalize, add batch dim
            observation = prepare_observation_for_inference(
                observation,
                self.device,
                self.single_task,
                self.robot.robot_type,
            )

            # Run policy inference to get action chunk
            action_chunk = self.policy.predict_action_chunk(observation)

        # Remove batch dimension
        return action_chunk.squeeze(0)

    def launch_next_inference_staged(
        self,
        observation: dict[str, np.ndarray],
    ) -> tuple[torch.Tensor, InferenceArtifacts]:
        """Run staged inference when the policy exposes explicit runtime phases."""
        total_start = time.perf_counter()
        observation, future_state_mode = self.prepare_inference_observation(observation)

        with torch.inference_mode():
            prepare_start = time.perf_counter()
            observation = prepare_observation_for_inference(
                observation,
                self.device,
                self.single_task,
                self.robot.robot_type,
            )
            prepare_time = time.perf_counter() - prepare_start

            prefix_start = time.perf_counter()
            prefix_context = self.policy.build_prefix_context(observation)
            prefix_time = time.perf_counter() - prefix_start

            suffix_start = time.perf_counter()
            action_chunk = self.policy.rollout_action_chunk(prefix_context)
            suffix_time = time.perf_counter() - suffix_start

        timings = InferenceTimings(
            observation_prepare_s=prepare_time,
            prefix_build_s=prefix_time,
            suffix_rollout_s=suffix_time,
            total_s=time.perf_counter() - total_start,
        )

        return action_chunk.squeeze(0), InferenceArtifacts(
            prefix_context=prefix_context,
            timings=timings,
            future_state_mode=future_state_mode,
        )

    def prepare_inference_observation(
        self,
        observation: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], str]:
        """Apply image remapping and future-state substitution before inference."""
        prepared = copy(observation)
        prepared = apply_policy_image_mapping(prepared, self.image_feature_map)

        future_state, future_state_mode = self.rollforward_state(prepared)
        if future_state is not None:
            prepared["observation.state"] = future_state

        return prepared, future_state_mode

    def rollforward_state(
        self,
        observation: dict[str, np.ndarray],
    ) -> tuple[np.ndarray | None, str]:
        """Approximate the state at the end of the current chunk.

        First pass implementation:
        - if no chunk is active, keep the live observation
        - if state and action dimensions match, extrapolate from the remaining chunk
        - otherwise, fall back to the previous last-action surrogate
        """
        if self.current_chunk_slot.actions_cpu is None:
            return None, "none"

        current_state = observation.get("observation.state")
        remaining_actions = self.current_chunk_slot.actions_cpu[self.chunk_index :]
        if remaining_actions.size == 0:
            return self.project_action_to_state(
                self.current_chunk_slot.actions_cpu[-1],
                current_state,
            ), "last_action"

        if current_state is None:
            return remaining_actions[-1].copy(), "last_action"

        current_state = np.asarray(current_state)
        if current_state.ndim == 0:
            return self.project_action_to_state(remaining_actions[-1], current_state), "last_action"

        state_dim = current_state.shape[-1]
        action_dim = remaining_actions.shape[-1]
        if state_dim != action_dim:
            return self.project_action_to_state(remaining_actions[-1], current_state), "last_action_projected"

        start_action = remaining_actions[0]
        end_action = remaining_actions[-1]
        predicted_state = current_state.copy()
        predicted_state[...] = current_state + (end_action - start_action)
        return predicted_state, "delta_rollforward"

    def project_action_to_state(
        self,
        action: np.ndarray,
        current_state: np.ndarray | None,
    ) -> np.ndarray:
        """Project an action vector into the policy state shape.

        This is a conservative fallback for mixed state/action dimensions:
        preserve the current state shape when available, then overwrite the
        leading coordinates with the action surrogate.
        """
        action = np.asarray(action, dtype=np.float32)
        if current_state is None:
            return action.copy()

        projected = np.asarray(current_state, dtype=np.float32).copy()
        if projected.ndim == 0:
            return action.copy()

        overlap_dim = min(projected.shape[-1], action.shape[-1])
        projected[..., :overlap_dim] = action[..., :overlap_dim]
        return projected

    def log_inference_artifacts(self, stage: str, inference_artifacts: InferenceArtifacts) -> None:
        """Emit debug logging for staged runtime launches."""
        if not logging.getLogger().isEnabledFor(logging.DEBUG):
            return

        payload = {
            "stage": stage,
            "future_state_mode": inference_artifacts.future_state_mode,
        }
        if inference_artifacts.timings is not None:
            payload["timings_ms"] = {
                key: round(value * 1000, 2)
                for key, value in asdict(inference_artifacts.timings).items()
            }
        logging.debug("Staged inference artifacts: %s", payload)

    def record_inference_artifacts(self, inference_artifacts: InferenceArtifacts) -> None:
        """Accumulate staged inference timings into runtime stats."""
        self.runtime_stats.inference_launches += 1
        self.runtime_stats.last_future_state_mode = inference_artifacts.future_state_mode
        if inference_artifacts.timings is None:
            return

        self.runtime_stats.observation_prepare_s += inference_artifacts.timings.observation_prepare_s
        self.runtime_stats.prefix_build_s += inference_artifacts.timings.prefix_build_s
        self.runtime_stats.suffix_rollout_s += inference_artifacts.timings.suffix_rollout_s
        self.runtime_stats.inference_total_s += inference_artifacts.timings.total_s

    def get_runtime_stats(self) -> RuntimeStats:
        """Return a shallow copy of aggregate runtime stats."""
        return RuntimeStats(**asdict(self.runtime_stats))

    def get_action(self, observation_frame: dict) -> dict[str, float]:
        """Get the next action to execute.
        
        This is the main interface called each control loop iteration.
        It manages chunk transitions and triggers async inference.
        
        Args:
            observation_frame: Current observation in dataset format.
            
        Returns:
            Action dictionary for the robot to execute.
        """
        use_staged_runtime = all(
            hasattr(self.policy, attr)
            for attr in ("build_prefix_context", "rollout_action_chunk")
        )

        # Bootstrap: compute first chunk synchronously
        if not self.is_running():
            if use_staged_runtime:
                action_chunk, inference_artifacts = self.launch_next_inference_staged(observation_frame)
                self.current_chunk_slot = ChunkSlot(
                    actions_cpu=action_chunk.cpu().numpy(),
                    produced_at=time.perf_counter(),
                )
                self.current_inference_artifacts = inference_artifacts
                self.record_inference_artifacts(inference_artifacts)
                self.log_inference_artifacts("bootstrap", inference_artifacts)
            else:
                self.current_chunk_slot = ChunkSlot(
                    actions_cpu=self.launch_next_inference(observation_frame).cpu().numpy(),
                    produced_at=time.perf_counter(),
                )
        # Chunk transition: move pre-computed next chunk to current
        elif self.should_switch_chunk():
            handoff_start = time.perf_counter()
            self.current_chunk_slot = ChunkSlot(
                actions_cpu=self.next_chunk_slot.actions_gpu.cpu().numpy()
                if self.next_chunk_slot.actions_gpu is not None
                else None,
                produced_at=self.next_chunk_slot.produced_at,
            )
            self.runtime_stats.chunk_switches += 1
            self.runtime_stats.chunk_handoff_s += time.perf_counter() - handoff_start
            self.current_inference_artifacts = self.next_inference_artifacts
            self.next_chunk_slot = ChunkSlot()
            self.next_inference_artifacts = InferenceArtifacts()

        # Async inference: start computing next chunk in advance
        if self.should_launch_next_inference():
            if use_staged_runtime:
                action_chunk, inference_artifacts = self.launch_next_inference_staged(observation_frame)
                self.next_chunk_slot = ChunkSlot(
                    actions_gpu=action_chunk,
                    produced_at=time.perf_counter(),
                )
                self.next_inference_artifacts = inference_artifacts
                self.record_inference_artifacts(inference_artifacts)
                self.log_inference_artifacts("overlap", inference_artifacts)
            else:
                self.next_chunk_slot = ChunkSlot(
                    actions_gpu=self.launch_next_inference(observation_frame),
                    produced_at=time.perf_counter(),
                )

        # Get action at current index
        action = self.get_current_action()

        # Advance index and handle chunk completion
        self.chunk_index = (self.chunk_index + 1) % self.n_action_steps
        if self.chunk_index == 0:
            self.current_chunk_slot = ChunkSlot()
            self.current_inference_artifacts = InferenceArtifacts()

        return action


def validate_robot_cameras(
    robot: Robot,
    policy_config: PreTrainedConfig,
    camera_feature_map: dict[str, str] | None = None,
) -> dict[str, str]:
    """Validate that robot cameras match policy expectations.
    
    Ensures the robot's camera configuration exactly matches what the
    policy was trained with. Mismatches will cause inference failures.
    
    Args:
        robot: Connected robot instance.
        policy_config: Configuration of the pretrained policy.
        
    Returns:
        Mapping from policy image feature name to robot image feature name.

    Raises:
        ValueError: If policy image features cannot be satisfied by robot cameras
            after applying the optional remapping.
    """
    # Build set of robot camera feature names (with observation.images prefix)
    robot_camera_names = set(robot.cameras.keys())
    robot_image_features = {f"{OBS_IMAGES}.{name}" for name in robot_camera_names}

    # Get policy's expected image features
    policy_image_features = policy_config.image_features
    if not isinstance(policy_image_features, dict):
        raise ValueError(
            f"Policy image_features must be a dict, got {type(policy_image_features)}: {policy_image_features}"
        )

    policy_camera_features = set(policy_image_features.keys())
    normalized_camera_map = {
        normalize_image_feature_name(target): normalize_image_feature_name(source)
        for target, source in (camera_feature_map or {}).items()
    }

    unknown_targets = sorted(set(normalized_camera_map) - policy_camera_features)
    unknown_sources = sorted(set(normalized_camera_map.values()) - robot_image_features)
    if unknown_targets or unknown_sources:
        raise ValueError(
            "Invalid camera_feature_map.\n"
            f"Unknown policy image targets: {unknown_targets or '[]'}\n"
            f"Unknown robot image sources: {unknown_sources or '[]'}\n"
            f"Robot cameras (with prefix): {sorted(robot_image_features)}\n"
            f"Policy image features: {sorted(policy_camera_features)}"
        )

    resolved_image_map: dict[str, str] = {}
    missing_targets: list[str] = []
    for target_feature in sorted(policy_camera_features):
        if target_feature in normalized_camera_map:
            resolved_image_map[target_feature] = normalized_camera_map[target_feature]
        elif target_feature in robot_image_features:
            resolved_image_map[target_feature] = target_feature
        else:
            missing_targets.append(target_feature)

    if missing_targets:
        raise ValueError(
            "Robot cameras do not satisfy policy image features.\n"
            f"Robot cameras (with prefix): {sorted(robot_image_features)}\n"
            f"Policy image features: {sorted(policy_camera_features)}\n"
            f"Missing policy image features after remapping: {missing_targets}\n"
            "Add `camera_feature_map` in the run config to rename or duplicate robot cameras, for example:\n"
            "camera_feature_map:\n"
            "  image: wrist\n"
            "  wrist_image: wrist"
        )

    if resolved_image_map != {feature: feature for feature in sorted(policy_camera_features)}:
        logging.info("Using camera feature remapping: %s", resolved_image_map)

    unused_robot_features = sorted(robot_image_features - set(resolved_image_map.values()))
    if unused_robot_features:
        logging.info("Ignoring extra robot image features not used by the policy: %s", unused_robot_features)

    return resolved_image_map


@torch.inference_mode()
def run_loop(
    robot: Robot,
    events: dict,
    fps: int,
    dataset_features: dict[str, dict],
    policy: PreTrainedPolicy,
    single_task: str | None,
    image_feature_map: dict[str, str] | None = None,
    action_quant_ratio: int = 1,
    inference_overlap_steps: int = 0,
    display_data: bool = False,
    control_time_s: int | float = 60,
) -> dict:
    """Core control loop for real-time robot operation.
    
    Runs the policy on the robot at the specified frequency, managing
    observation capture, action inference, and command execution.
    
    Args:
        robot: Connected robot instance.
        events: Event dictionary for keyboard control (exit_early flag).
        fps: Target control frequency in Hz.
        dataset_features: Feature definitions for observation/action conversion.
        policy: Loaded policy for action prediction.
        single_task: Task description for policies.
        action_quant_ratio: Action quantization ratio.
        inference_overlap_steps: Steps of overlap between chunks.
        display_data: Whether to log data to Rerun for visualization.
        control_time_s: Total runtime in seconds.
    """
    # Reset policy state (clears any cached observations)
    if policy is not None:
        policy.reset()

    # Initialize async manager for VLASH inference
    # Scale overlap_steps by action_quant_ratio to match effective step count
    effective_overlap_steps = inference_overlap_steps * action_quant_ratio
    logging.info(f"Effective overlap_steps: {effective_overlap_steps} (inference_overlap_steps={inference_overlap_steps} * action_quant_ratio={action_quant_ratio})")
    async_manager = VLASHAsyncManager(
        policy=policy,
        robot=robot,
        single_task=single_task,
        overlap_steps=effective_overlap_steps,
        image_feature_map=image_feature_map,
    )

    step_count = 0
    observation_slot = ObservationSlot()
    start_time = time.perf_counter()

    # Main control loop
    while time.perf_counter() - start_time < control_time_s:
        loop_start = time.perf_counter()

        # Check for keyboard interrupt (Escape key)
        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Fetch observation only when needed (reduces camera latency)
        if async_manager.should_fetch_observation():
            observation_fetch_start = time.perf_counter()
            observation = robot.get_observation()
            observation_slot = ObservationSlot(
                frame=build_dataset_frame(dataset_features, observation, prefix="observation"),
                captured_at=time.perf_counter(),
            )
            async_manager.runtime_stats.observation_fetches += 1
            async_manager.runtime_stats.observation_fetch_s += time.perf_counter() - observation_fetch_start
        else:
            observation = None

        # Get action from async manager (handles chunk management internally)
        action = async_manager.get_action(observation_slot.frame)

        # Send action based on quantization ratio
        if (step_count + 1) % action_quant_ratio == 0:
            robot.send_action(action)

            # Optional: log to Rerun for debugging/visualization
            if display_data and observation is not None:
                log_rerun_data(observation, action)

            # Maintain target frequency
            elapsed = time.perf_counter() - loop_start
            busy_wait(1 / fps - elapsed)

        async_manager.runtime_stats.loop_iterations += 1
        async_manager.runtime_stats.loop_s += time.perf_counter() - loop_start
        step_count += 1

    runtime_stats = async_manager.get_runtime_stats()
    payload = {
        "loop_iterations": runtime_stats.loop_iterations,
        "observation_fetches": runtime_stats.observation_fetches,
        "inference_launches": runtime_stats.inference_launches,
        "chunk_switches": runtime_stats.chunk_switches,
        "last_future_state_mode": runtime_stats.last_future_state_mode,
        "timings_ms": {
            "loop_avg": round(
                (runtime_stats.loop_s / runtime_stats.loop_iterations) * 1000, 2
            )
            if runtime_stats.loop_iterations
            else 0.0,
            "observation_fetch_avg": round(
                (runtime_stats.observation_fetch_s / runtime_stats.observation_fetches) * 1000, 2
            )
            if runtime_stats.observation_fetches
            else 0.0,
            "chunk_handoff_avg": round(
                (runtime_stats.chunk_handoff_s / runtime_stats.chunk_switches) * 1000, 2
            )
            if runtime_stats.chunk_switches
            else 0.0,
        },
        "stage_timings_ms": stage_timings_to_dict(
            summarize_stage_timings(
                prepare_total_s=runtime_stats.observation_prepare_s,
                prefix_total_s=runtime_stats.prefix_build_s,
                suffix_total_s=runtime_stats.suffix_rollout_s,
                total_total_s=runtime_stats.inference_total_s,
                samples=runtime_stats.inference_launches,
            )
        ),
    }
    logging.info("VLASH runtime stats: %s", payload)
    return payload


def load_and_compile_policy(cfg: RunConfig) -> PreTrainedPolicy:
    """Load pretrained policy from checkpoint.
    
    Args:
        cfg: Run configuration with policy path and settings.
        
    Returns:
        Loaded policy ready for inference.
    """
    policy_cls = get_policy_class(cfg.policy.type)
    policy: PreTrainedPolicy = policy_cls.from_pretrained(
        pretrained_name_or_path=cfg.policy.pretrained_path,
        config=cfg.policy,
    )

    if cfg.policy.compile_model:
        warmup_compiled_policy(policy, cfg.single_task)

    return policy


def warmup_compiled_policy(
    policy: PreTrainedPolicy,
    single_task: str | None,
    warmup_steps: int = 3,
):
    """Warm up compiled policy to trigger torch.compile.
    
    Running a few inference passes before actual control ensures that
    torch.compile has finished optimizing the model, avoiding latency
    spikes during real operation.
    
    Args:
        policy: Compiled policy to warm up.
        robot: Robot instance (unused, kept for API compatibility).
        single_task: Optional task string.
        warmup_steps: Number of warmup iterations.
    """
    logging.info("Warming up compiled policy...")
    
    device = get_safe_torch_device(policy.config.device)
    
    # Create dummy observation matching policy's expected input shape
    # Format: [B, C, H, W] for images, [B, state_dim] for state
    dummy_obs = {}
    
    # Add dummy image observations with correct shape [B, C, H, W]
    for img_key, img_feature in policy.config.image_features.items():
        shape = tuple(img_feature.shape)
        if len(shape) != 3:
            raise ValueError(f"Expected 3D image feature shape for {img_key}, got {shape}")

        # Checkpoint configs may store image shapes as either CHW or HWC.
        if shape[0] in (1, 3) and shape[0] < shape[1] and shape[0] < shape[2]:
            channels, height, width = shape
        elif shape[-1] in (1, 3):
            height, width, channels = shape
        else:
            channels, height, width = shape

        dummy_obs[img_key] = torch.zeros(
            (1, channels, height, width),
            dtype=torch.float32,
            device=device,
        )
    
    # Add dummy state observation with correct shape [B, state_dim]
    # Get state dimension from policy config's input_features
    if "observation.state" in policy.config.input_features:
        state_dim = policy.config.input_features["observation.state"].shape[0]
        dummy_obs["observation.state"] = torch.zeros(
            (1, state_dim),
            dtype=torch.float32,
            device=device,
        )
    
    # Add task string
    dummy_obs["task"] = single_task if single_task is not None else ""
    
    # Run warmup iterations to complete compilation
    # Use predict_action_chunk which includes all necessary preprocessing
    warmup_start = time.perf_counter()
    for i in range(warmup_steps):
        with torch.inference_mode():
            _ = policy.predict_action_chunk(dummy_obs)
    
    warmup_time = time.perf_counter() - warmup_start
    logging.info(f"Warmup complete ({warmup_steps} steps in {warmup_time:.2f}s)")


def build_dataset_features(robot: Robot) -> dict[str, dict]:
    """Build dataset-style feature definitions from robot config.
    
    Converts robot's hardware feature definitions to the format expected
    by LeRobot's dataset utilities for observation/action frame building.
    
    Args:
        robot: Robot instance with observation and action features.
        
    Returns:
        Combined dictionary of action and observation feature definitions.
    """
    action_features = hw_to_dataset_features(robot.action_features, "action", use_video=True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
    return {**action_features, **obs_features}


def make_runtime_robot(config) -> Robot:
    """Instantiate either a real robot or the local mock robot."""
    if isinstance(config, MockRobotConfig) or getattr(config, "type", None) == "mock_robot":
        return MockRobot(config)
    if isinstance(config, LiberoRobotConfig) or getattr(config, "type", None) == "libero_robot":
        return LiberoRobot(config)
    return make_robot_from_config(config)


@parser.wrap()
def run(cfg: RunConfig):
    """Main entry point for VLASH robot inference.
    
    Loads a pretrained policy and runs it on a connected robot using
    VLASH's async inference strategy for real-time control.
    
    Args:
        cfg: Run configuration parsed from YAML and CLI arguments.
    """
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Validate task description is provided (not placeholder)
    if cfg.single_task is None or cfg.single_task == "<task description>":
        raise ValueError(
            "Please provide a language prompt (task description) in the config file.\n"
            "The 'single_task' field cannot be empty or use the placeholder '<task description>'.\n"
            "Example: single_task: 'pick up the cube and place it in the box'"
        )

    # Initialize Rerun visualization if requested
    if cfg.display_data:
        init_rerun(session_name="vlash_run")

    # Setup robot and validate camera configuration
    robot = make_runtime_robot(cfg.robot)
    original_policy_config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)
    image_feature_map = validate_robot_cameras(robot, original_policy_config, cfg.camera_feature_map)

    # Load policy and prepare feature definitions
    policy = load_and_compile_policy(cfg)
    dataset_features = build_dataset_features(robot)

    # Connect to robot and setup keyboard listener for manual control
    robot.connect()
    listener, events = init_keyboard_listener()

    log_say("Starting VLASH run", cfg.play_sounds, blocking=True)

    try:
        # Run the main control loop
        runtime_stats = run_loop(
            robot=robot,
            events=events,
            fps=cfg.fps,
            dataset_features=dataset_features,
            policy=policy,
            single_task=cfg.single_task,
            image_feature_map=image_feature_map,
            action_quant_ratio=cfg.action_quant_ratio,
            inference_overlap_steps=cfg.inference_overlap_steps,
            display_data=cfg.display_data,
            control_time_s=cfg.control_time_s,
        )
        if cfg.runtime_stats_output:
            output_path = Path(cfg.runtime_stats_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as f:
                json.dump(runtime_stats, f, indent=2)
            logging.info("Saved VLASH runtime stats to %s", output_path)
    finally:
        # Cleanup: disconnect robot and stop keyboard listener
        log_say("Stopping VLASH run", cfg.play_sounds, blocking=True)
        robot.disconnect()
        if listener is not None:
            listener.stop()


def main():
    """CLI entry point."""
    run()


if __name__ == "__main__":
    main()
