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
"""Inference Latency Benchmark.

"""

import json
import logging
import time
from copy import copy
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lerobot.configs import parser
from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging

from benchmarks.benchmark_config import BenchmarkConfig
from vlash.policies.factory import get_policy_class, make_policy
from vlash.runtime_stats import stage_timings_to_dict, summarize_stage_timings


def load_dataset(cfg: BenchmarkConfig) -> tuple[LeRobotDataset, LeRobotDatasetMetadata]:
    """Load dataset for benchmarking.
    
    Uses standard LeRobotDataset without temporal augmentation.
    
    Args:
        cfg: Benchmark configuration.
        
    Returns:
        Tuple of (dataset, metadata).
    """
    logging.info(f"Loading dataset: {cfg.dataset.repo_id}")
    
    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
    )
    
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    
    dataset = LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        delta_timestamps=delta_timestamps,
        revision=cfg.dataset.revision,
    )
    
    logging.info(f"Dataset loaded: {len(dataset)} samples, {dataset.num_episodes} episodes")
    
    return dataset, ds_meta


def load_policy(cfg: BenchmarkConfig, ds_meta: LeRobotDatasetMetadata) -> PreTrainedPolicy:
    """Load pretrained policy for benchmarking.
    
    Uses VLASH's policy factory which handles both pretrained and new models.
    
    Args:
        cfg: Benchmark configuration.
        ds_meta: Dataset metadata.
        
    Returns:
        Policy in eval mode.
    """
    logging.info(f"Loading policy type: {cfg.policy.type}")
    
    policy_cls = get_policy_class(cfg.policy.type)
    if cfg.policy.pretrained_path:
        policy = policy_cls.from_pretrained(
            pretrained_name_or_path=cfg.policy.pretrained_path,
            config=cfg.policy,
        )
    else:
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=ds_meta,
        )
    
    policy.eval()
    
    device = get_safe_torch_device(cfg.policy.device)
    logging.info(f"Policy loaded successfully on device: {device}")
    
    return policy


def apply_image_feature_map(batch: dict, image_feature_map: dict[str, str]) -> dict:
    """Copy dataset image tensors into the keys expected by the policy."""
    if not image_feature_map:
        return batch

    mapped = copy(batch)
    for target_key, source_key in image_feature_map.items():
        if source_key not in mapped:
            raise ValueError(
                f"Mapped source image feature {source_key!r} is missing from benchmark batch "
                f"(available keys: {sorted(mapped.keys())})"
            )
        mapped[target_key] = mapped[source_key]

    return mapped


def validate_image_feature_map(
    cfg: BenchmarkConfig,
    policy: PreTrainedPolicy,
    dataset: LeRobotDataset,
) -> None:
    """Validate that dataset image features satisfy the policy inputs."""
    policy_image_features = getattr(policy.config, "image_features", {})
    if not isinstance(policy_image_features, dict):
        raise ValueError(
            f"Policy image_features must be a dict, got {type(policy_image_features)}: {policy_image_features}"
        )

    dataset_features = set(dataset.features)
    image_feature_map = cfg.image_feature_map or {}

    unknown_targets = sorted(set(image_feature_map) - set(policy_image_features))
    unknown_sources = sorted(set(image_feature_map.values()) - dataset_features)
    if unknown_targets or unknown_sources:
        raise ValueError(
            "Invalid image_feature_map.\n"
            f"Unknown policy image targets: {unknown_targets or '[]'}\n"
            f"Unknown dataset image sources: {unknown_sources or '[]'}\n"
            f"Dataset features: {sorted(dataset_features)}\n"
            f"Policy image features: {sorted(policy_image_features)}"
        )

    missing_targets: list[str] = []
    resolved_image_map: dict[str, str] = {}
    for target_feature in sorted(policy_image_features):
        if target_feature in image_feature_map:
            resolved_image_map[target_feature] = image_feature_map[target_feature]
        elif target_feature in dataset_features:
            resolved_image_map[target_feature] = target_feature
        else:
            missing_targets.append(target_feature)

    if missing_targets:
        raise ValueError(
            "Dataset features do not satisfy policy image features.\n"
            f"Dataset features: {sorted(dataset_features)}\n"
            f"Policy image features: {sorted(policy_image_features)}\n"
            f"Missing policy image features after remapping: {missing_targets}\n"
            "Add `image_feature_map` in the benchmark config, for example:\n"
            "image_feature_map:\n"
            "  observation.images.image: observation.image\n"
            "  observation.images.wrist_image: observation.image"
        )

    if resolved_image_map != {feature: feature for feature in sorted(policy_image_features)}:
        logging.info("Using benchmark image feature remapping: %s", resolved_image_map)


def adapt_state_features(batch: dict, policy: PreTrainedPolicy) -> dict:
    """Pad or truncate state tensors to the dimensions expected by the policy.

    This keeps latency smoke tests running even when the benchmark dataset and
    checkpoint were trained with different state layouts.
    """
    adapted = dict(batch)
    input_features = getattr(policy.config, "input_features", {}) or {}

    for key, feature in input_features.items():
        if key not in adapted or feature.type is not FeatureType.STATE:
            continue

        tensor = adapted[key]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
            continue

        target_dim = feature.shape[0]
        current_dim = tensor.shape[-1]
        if current_dim == target_dim:
            continue

        if current_dim < target_dim:
            adapted[key] = F.pad(tensor, (0, target_dim - current_dim))
        else:
            adapted[key] = tensor[..., :target_dim]

    return adapted


def log_state_feature_adaptation(policy: PreTrainedPolicy, dataset: LeRobotDataset) -> None:
    """Log state-shape mismatches that will be adapted during benchmarking."""
    input_features = getattr(policy.config, "input_features", {}) or {}
    for key, feature in input_features.items():
        if feature.type is not FeatureType.STATE or key not in dataset.features:
            continue

        dataset_shape = tuple(dataset.features[key]["shape"])
        policy_shape = tuple(feature.shape)
        if dataset_shape != policy_shape:
            logging.info(
                "Adapting benchmark state feature %s from dataset shape %s to policy shape %s",
                key,
                dataset_shape,
                policy_shape,
            )


def prepare_batch(
    batch: dict,
    device: torch.device,
    policy: PreTrainedPolicy,
    image_feature_map: dict[str, str] | None = None,
) -> dict:
    """Move batch tensors to device.
    
    Also converts language_instruction to task field expected by policy.
    
    Args:
        batch: Input batch from dataloader.
        device: Target device.
        
    Returns:
        Prepared batch dictionary.
    """
    batch = apply_image_feature_map(batch, image_feature_map or {})
    batch = adapt_state_features(batch, policy)
    prepared = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            prepared[k] = v.to(device)
        else:
            prepared[k] = v
    
    if "language_instruction" in batch:
        prepared["task"] = batch["language_instruction"]
    
    return prepared


def warmup_model(
    policy: PreTrainedPolicy,
    dataloader: DataLoader,
    cfg: BenchmarkConfig,
):
    """Warm up model before benchmarking.
    """
    if cfg.warmup_steps <= 0:
        return
    
    logging.info(f"Warming up model for {cfg.warmup_steps} steps...")
    device = get_safe_torch_device(cfg.policy.device)
    
    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
            if i >= cfg.warmup_steps:
                break
            
            batch = prepare_batch(batch, device, policy, cfg.image_feature_map)
            _ = policy.predict_action_chunk(batch)
    
    # Ensure warmup is complete before starting benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    logging.info("Warmup complete")


def benchmark_inference_latency_impl(
    policy: PreTrainedPolicy,
    dataloader: DataLoader,
    cfg: BenchmarkConfig,
) -> dict:
    """Run inference latency measurement.
        """
    device = get_safe_torch_device(cfg.policy.device)
    latencies = []
    
    logging.info(f"Starting inference latency benchmarking with {cfg.num_samples} samples...")
    
    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
            if i >= cfg.num_samples:
                break
            
            batch = prepare_batch(batch, device, policy, cfg.image_feature_map)
            
            # Synchronize before timing for accurate GPU measurement
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            _ = policy.predict_action_chunk(batch)
            
            # Synchronize after inference
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000  # ms
            latencies.append(latency)
            
            if (i + 1) % 10 == 0:
                logging.info(f"Processed {i + 1}/{cfg.num_samples} samples...")
    
    # Compute statistics
    latencies = np.array(latencies)
    results = {
        "num_samples": len(latencies),
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p90_ms": float(np.percentile(latencies, 90)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "fps": float(1000.0 / np.mean(latencies)),
    }
    
    return results


def profile_stage_timings(
    policy: PreTrainedPolicy,
    dataloader: DataLoader,
    cfg: BenchmarkConfig,
) -> dict[str, float] | None:
    """Profile staged runtime phases without perturbing the main latency path."""
    if not all(hasattr(policy, attr) for attr in ("build_prefix_context", "rollout_action_chunk")):
        return None

    device = get_safe_torch_device(cfg.policy.device)
    stage_prepare_total_s = 0.0
    stage_prefix_total_s = 0.0
    stage_suffix_total_s = 0.0
    stage_total_total_s = 0.0
    stage_samples = 0

    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
            if i >= cfg.num_samples:
                break

            prepare_start = time.perf_counter()
            batch = prepare_batch(batch, device, policy, cfg.image_feature_map)
            if device.type == "cuda":
                torch.cuda.synchronize()
            prepare_s = time.perf_counter() - prepare_start

            total_start = time.perf_counter()

            prefix_start = time.perf_counter()
            prefix_context = policy.build_prefix_context(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            prefix_s = time.perf_counter() - prefix_start

            suffix_start = time.perf_counter()
            _ = policy.rollout_action_chunk(prefix_context)
            if device.type == "cuda":
                torch.cuda.synchronize()
            suffix_s = time.perf_counter() - suffix_start

            total_s = time.perf_counter() - total_start
            stage_prepare_total_s += prepare_s
            stage_prefix_total_s += prefix_s
            stage_suffix_total_s += suffix_s
            stage_total_total_s += total_s
            stage_samples += 1

    if stage_samples == 0:
        return None

    return stage_timings_to_dict(
        summarize_stage_timings(
            prepare_total_s=stage_prepare_total_s,
            prefix_total_s=stage_prefix_total_s,
            suffix_total_s=stage_suffix_total_s,
            total_total_s=stage_total_total_s,
            samples=stage_samples,
        )
    )


def print_results(results: dict, cfg: BenchmarkConfig):
    """Print formatted benchmark results to console."""
    pretrained_path = getattr(cfg.policy, 'pretrained_path', None) or "N/A (new model)"
    
    print("\n" + "=" * 80)
    print("INFERENCE LATENCY BENCHMARK RESULTS")
    print("=" * 80)
    print(f"\nPolicy Type: {cfg.policy.type}")
    print(f"Pretrained Path: {pretrained_path}")
    print(f"Dataset: {cfg.dataset.repo_id}")
    print(f"Device: {cfg.policy.device}")
    print(f"Batch Size: {cfg.batch_size}")
    print(f"Compile: {cfg.policy.compile_model}")
    print(f"\nNumber of samples: {results['num_samples']}")
    print(f"\nLatency Statistics (milliseconds):")
    print(f"  Mean:   {results['mean_ms']:.2f} ms")
    print(f"  Median: {results['median_ms']:.2f} ms")
    print(f"  Std:    {results['std_ms']:.2f} ms")
    print(f"  Min:    {results['min_ms']:.2f} ms")
    print(f"  Max:    {results['max_ms']:.2f} ms")
    print(f"\nPercentiles:")
    print(f"  P50: {results['p50_ms']:.2f} ms")
    print(f"  P90: {results['p90_ms']:.2f} ms")
    print(f"  P95: {results['p95_ms']:.2f} ms")
    print(f"  P99: {results['p99_ms']:.2f} ms")
    print(f"\nThroughput:")
    print(f"  FPS: {results['fps']:.2f}")
    if "stage_timings_ms" in results:
        stage_timings = results["stage_timings_ms"]
        print(f"\nStage Timing Means (milliseconds):")
        print(f"  Prepare: {stage_timings['prepare_mean_ms']:.2f} ms")
        print(f"  Prefix:  {stage_timings['prefix_mean_ms']:.2f} ms")
        print(f"  Suffix:  {stage_timings['suffix_mean_ms']:.2f} ms")
        print(f"  Total:   {stage_timings['total_mean_ms']:.2f} ms")
    print("=" * 80 + "\n")


def save_results(results: dict, cfg: BenchmarkConfig):
    """Save benchmark results to JSON file.
    
    Includes both configuration and results for reproducibility.
    
    Args:
        results: Benchmark results.
        cfg: Configuration used for benchmark.
    """
    if cfg.output_file is None:
        return
    
    output_path = Path(cfg.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    pretrained_path = getattr(cfg.policy, 'pretrained_path', None)
    
    output_data = {
        "config": {
            "benchmark_type": "inference_latency",
            "policy_type": cfg.policy.type,
            "policy_path": str(pretrained_path) if pretrained_path else None,
            "dataset_repo_id": cfg.dataset.repo_id,
            "device": cfg.policy.device,
            "batch_size": cfg.batch_size,
            "num_samples": cfg.num_samples,
            "warmup_steps": cfg.warmup_steps,
            "compile_model": cfg.policy.compile_model,
            "seed": cfg.seed,
        },
        "results": results,
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logging.info(f"Results saved to: {output_path}")


@parser.wrap()
def benchmark_inference_latency(cfg: BenchmarkConfig):
    """Main entry point for inference latency benchmark.
    """
    init_logging()
    logging.info("Starting inference latency benchmark")
        
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))
    
    set_seed(cfg.seed)
    
    # Load dataset and policy
    dataset, ds_meta = load_dataset(cfg)
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True if cfg.policy.device == "cuda" else False,
    )
    
    policy = load_policy(cfg, ds_meta)
    validate_image_feature_map(cfg, policy, dataset)
    log_state_feature_adaptation(policy, dataset)
    
    # Warmup
    warmup_model(policy, dataloader, cfg)
    
    # Reset dataloader for actual benchmarking
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True if cfg.policy.device == "cuda" else False,
    )
    
    # Benchmark
    results = benchmark_inference_latency_impl(policy, dataloader, cfg)

    # Stage profiling uses a separate pass so it does not perturb the main latency numbers.
    stage_dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True if cfg.policy.device == "cuda" else False,
    )
    stage_timings = profile_stage_timings(policy, stage_dataloader, cfg)
    if stage_timings is not None:
        results["stage_timings_ms"] = stage_timings
    
    # Output
    print_results(results, cfg)
    save_results(results, cfg)
    
    logging.info("Inference latency benchmark complete!")


def main():
    benchmark_inference_latency()


if __name__ == "__main__":
    main()
