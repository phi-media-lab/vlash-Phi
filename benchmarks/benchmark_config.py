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
"""Benchmark Configuration.

This module defines the configuration for VLASH benchmarks.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig


@dataclass
class BenchmarkConfig(TrainPipelineConfig):
    """Configuration for benchmarking a pretrained policy.
    
    Reuses TrainPipelineConfig's dataset and policy configuration.
    Training-specific fields are ignored during benchmarking.
    
    Use policy.compile_model=true to enable torch.compile optimization.
    """
    
    # Benchmark type
    type: str = "inference_latency"
    
    # Number of samples for benchmarking
    num_samples: int = 100
    
    # Warmup iterations before timing
    warmup_steps: int = 10
    
    # Optional output file for JSON results
    output_file: Union[str, None] = None

    # Optional mapping from policy image feature names to dataset image feature names.
    # Example:
    # image_feature_map:
    #   observation.images.image: observation.image
    #   observation.images.wrist_image: observation.image
    image_feature_map: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate benchmark-specific configuration.
        
        Overrides parent to skip training-specific validation.
        """
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = Path(policy_path)

        if self.type not in ["inference_latency"]:
            raise ValueError(f"Invalid benchmark type: {self.type}.")

        if self.policy is None:
            raise ValueError(
                "Policy is not configured. Please specify a pretrained policy with "
                "`--policy.path=...` or set `policy.pretrained_path` in the benchmark config."
            )
        
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
