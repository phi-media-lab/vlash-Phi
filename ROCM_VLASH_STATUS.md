# VLASH On AMD ROCm: Status Summary

Last updated: 2026-04-11

## Scope

This document summarizes the work completed to evaluate and enable VLASH inference on the local AMD machine in this workspace.

Machine context:

- CPU: AMD Ryzen AI 9 HX PRO 370
- GPU: AMD Radeon 890M iGPU
- ROCm arch: `gfx1150`
- ROCm runtime: `/opt/rocm-7.2.1`
- Repository: `/home/amd/vlash`

## Executive Summary

VLASH inference can run on this AMD GPU through ROCm, but it is not a zero-config path.

What is now working:

- ROCm-backed PyTorch detects and uses the AMD GPU.
- `PI05Policy.from_pretrained(...)` runs real GPU inference on this machine.
- CLI-level `vlash benchmark` runs end-to-end on AMD GPU with a reproducible ROCm environment.
- `compile_model=true` works and is the only optimization knob that clearly improves latency on this hardware.
- `vlash run` now works end to end on both a local mock robot and a real `LIBERO` simulator adapter on AMD ROCm.

What is not true:

- This machine does not match the paper's deployment-level latency.
- Existing `fuse_qkv` / `fuse_gate_up` inference fusions do not improve latency on this AMD ROCm setup.
- Current results are compatibility and system-path validation, not paper-level task-faithful reproduction.

## Hardware And Runtime Detection

System tools confirmed the following:

- `rocm-smi` detects one AMD GPU at PCI bus `0000:C4:00.0`
- GPU ID: `0x150e`
- ROCm marketing name: `AMD Radeon Graphics`
- GPU arch: `gfx1150`
- This corresponds to the integrated `Radeon 890M`

Representative commands:

```bash
rocm-smi --showproductname --showbus --showid --showvbios --json
rocminfo
```

## Initial State And Constraint

The original Python environment was not usable for AMD GPU inference:

- env: `/home/amd/.miniforge3/envs/vlash`
- `torch 2.7.1+cu126`
- `torch.version.hip == None`
- `torch.cuda.is_available() == False`

The repository dependency line is part of the problem:

- [pyproject.toml](/home/amd/vlash/pyproject.toml) depends on `lerobot==0.4.1`
- local metadata for `lerobot 0.4.1` constrains torch to `<2.8`
- that conflicts with the ROCm PyTorch stack that actually works on this hardware

## ROCm Stack Validation

Two probe environments were tested.

### Old-Compatible Stack: Failed For Real Compute

Environment:

- Python 3.10
- `torch 2.7.1`
- ROCm 6.3 wheels

Result:

- GPU enumeration succeeded
- actual GPU ops failed with `HIP error: invalid device function`

Conclusion:

- the older torch line that fits `lerobot 0.4.1` dependency bounds does not actually run kernels correctly on this `gfx1150` device

### Working Stack: Confirmed

Environment:

- Python 3.12
- `torch 2.9.1+rocm7.2.1`
- `torchvision 0.24.0+rocm7.2.1`
- `torchaudio 2.9.0+rocm7.2.1`
- `triton 3.5.1+rocm7.2.1`

Result:

- `torch.cuda.is_available() == True`
- device count = `1`
- device name = `AMD Radeon Graphics`
- `fp16` matmul works
- `bf16` matmul works
- `torch.compile(...)` works

Required runtime env:

```bash
export LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}
```

Without this, importing AMD ROCm torch can fail due to missing `libroctx64.so.4`.

## Public VLASH Checkpoint Availability

Official public checkpoint confirmed usable with the current loader format:

- `mit-han-lab/vlash-pi05-libero-async5`

Local copy:

- `/home/amd/.cache/vlash/models/vlash-pi05-libero-async5`

Local files verified:

- `config.json`
- `model.safetensors`

Loading it locally with VLASH succeeded:

- policy class: `PI05Policy`
- parameters: `3,616,757,594`

Another official public artifact exists but is not directly usable by the current loader:

- `mit-han-lab/vlash-kinetix-policy-async5`
- format is `*.pkl`, not `config.json + model.safetensors`

## Code Changes Made

### Staged Runtime And Mock Run Validation

Added or updated:

- [modeling_pi05.py](/home/amd/vlash/vlash/policies/pi05/modeling_pi05.py)
- [run.py](/home/amd/vlash/vlash/run.py)
- [runtime_stats.py](/home/amd/vlash/vlash/runtime_stats.py)
- [mock_robot.py](/home/amd/vlash/vlash/mock_robot.py)
- [mock_rocm.yaml](/home/amd/vlash/examples/inference/mock_rocm.yaml)
- [mock_rocm_async.yaml](/home/amd/vlash/examples/inference/mock_rocm_async.yaml)

These changes introduced:

- explicit staged `PI05` runtime entry points
- runtime phase timing for `prepare / prefix / suffix / total`
- mock `vlash run` validation without robot hardware
- structured JSON output for runtime stats

This means the branch now validates not only CLI benchmark inference, but also the main `vlash run` path in a reproducible local mock environment.

Mock runtime sweep results are summarized in:

- [MOCK_RUNTIME_SWEEP.md](/home/amd/vlash/MOCK_RUNTIME_SWEEP.md)
- [MOCK_RUNTIME_EXTENDED_SWEEP.md](/home/amd/vlash/MOCK_RUNTIME_EXTENDED_SWEEP.md)
- [MOCK_RUNTIME_N4_GRID.md](/home/amd/vlash/MOCK_RUNTIME_N4_GRID.md)

### LIBERO Simulator Validation

Added:

- [vlash/libero_robot.py](/home/amd/vlash/vlash/libero_robot.py)
- [examples/inference/libero_rocm.yaml](/home/amd/vlash/examples/inference/libero_rocm.yaml)

Updated:

- [vlash/run.py](/home/amd/vlash/vlash/run.py)
- [vlash/configs/run_config.py](/home/amd/vlash/vlash/configs/run_config.py)

These changes add a `LIBERO`-backed `Robot` adapter so the existing `vlash run` main loop can operate against a simulator without rewriting the runtime around environment-specific code.

Validated states:

- `LIBERO` imports and assets initialize successfully
- `LiberoEnv` works locally with offscreen MuJoCo rendering
- `vlash run` works end to end on CPU in the original `vlash` env
- `vlash run` works end to end on AMD ROCm in `/home/amd/.venvs/vlash-rocm`

Representative AMD ROCm simulator results:

- `n_action_steps=32`, sync:
  - `loop_avg = 356.02 ms`
  - `stage_total = 1083.77 ms`
- `n_action_steps=32`, async `q2/o2`:
  - `loop_avg = 212.93 ms`
  - `stage_total = 1108.55 ms`
  - `chunk_switches = 0`
- `n_action_steps=8`, sync:
  - `loop_avg = 503.11 ms`
  - `stage_total = 1506.57 ms`
- `n_action_steps=8`, async `q2/o2`:
  - `loop_avg = 419.30 ms`
  - `stage_total = 1571.80 ms`
  - `chunk_switches = 1`
  - `last_future_state_mode = last_action_projected`

This is the first simulator-backed evidence that overlap is not only valid in mock runtime, but also functionally active in a task environment with real image observations and environment stepping.

Follow-up `LIBERO` sweep results are summarized in:

- [outputs/libero_runtime/sweep/summary.md](/home/amd/vlash/outputs/libero_runtime/sweep/summary.md)
- [outputs/libero_runtime/confirm/summary.md](/home/amd/vlash/outputs/libero_runtime/confirm/summary.md)

The first systematic simulator sweep used:

- `policy.n_action_steps=8`
- `control_time_s=6`
- `fps=5`

Key findings:

- `action_quant_ratio=2` remains the strongest loop-cadence lever
- async overlap is now stable enough to trigger `chunk_switches=2` across all async cases
- async cases consistently exercise `last_action_projected`
- staged inference total time remains nearly flat, around `1034-1051 ms`
- overlap benefit is not monotonic

Best current simulator-backed candidate:

- `n_action_steps=8`
- `action_quant_ratio=2`
- `inference_overlap_steps=1`

This candidate preserves async chunk switching while avoiding the small regression seen at `q2/o2`.

The longer confirmation sweep supports the same recommendation:

- `sync q2/o0`, `t=10`: `loop_avg = 256.03 ms`
- `async q2/o1`, `t=10`: `loop_avg = 260.45 ms`, `chunk_switches = 4`
- `async q2/o2`, `t=10`: `loop_avg = 265.96 ms`, `chunk_switches = 4`

So the current branch recommendation remains:

- keep `q=2` as the main loop-cadence lever
- prefer `o=1` over `o=2`
- treat async value primarily as a scheduling change, not a compute-speed improvement

### Real-Robot Runtime Compatibility

To make runtime image feature expectations more flexible:

- [vlash/run.py](/home/amd/vlash/vlash/run.py)
- [vlash/configs/run_config.py](/home/amd/vlash/vlash/configs/run_config.py)

Added support for `camera_feature_map` so a policy can consume renamed or duplicated camera features.

Example:

```yaml
camera_feature_map:
  image: wrist
  wrist_image: wrist
```

This makes public checkpoints with `observation.images.image` plus `observation.images.wrist_image` compatible with a single-`wrist` camera setup.

### Checkpoint Compatibility Tool

Added:

- [tools/check_checkpoint_compat.py](/home/amd/vlash/tools/check_checkpoint_compat.py)

Purpose:

- statically validate whether a run config can satisfy a checkpoint's expected image features

### Benchmark Improvements

Updated:

- [benchmarks/benchmark_config.py](/home/amd/vlash/benchmarks/benchmark_config.py)
- [benchmarks/benchmark_inference_latency.py](/home/amd/vlash/benchmarks/benchmark_inference_latency.py)

Changes:

- benchmark now correctly loads `--policy.path`
- benchmark supports `image_feature_map`
- benchmark adapts mismatched state dimensions through pad/truncate
- benchmark loads pretrained policy in inference semantics instead of rebuilding from dataset stats

Added:

- [examples/benchmarks/inference_latency_rocm_smoke.yaml](/home/amd/vlash/examples/benchmarks/inference_latency_rocm_smoke.yaml)

This smoke config maps `lerobot/pusht` inputs into the public LIBERO checkpoint's expected image/state layout well enough to exercise the full CLI path.

### ROCm Helper Scripts

Added:

- [tools/run_rocm_vlash.sh](/home/amd/vlash/tools/run_rocm_vlash.sh)
- [tools/check_rocm_runtime.sh](/home/amd/vlash/tools/check_rocm_runtime.sh)
- [tools/setup_rocm_env.sh](/home/amd/vlash/tools/setup_rocm_env.sh)

Purpose:

- run VLASH with the correct ROCm runtime variables
- validate the ROCm torch runtime
- rebuild the working ROCm environment reproducibly

### README Updates

Updated:

- [README.md](/home/amd/vlash/README.md)

Added AMD ROCm setup and usage notes.

## Reproducible ROCm Environment

Two usable environments now exist:

- probe env: `/home/amd/.venvs/rocm-probe-312`
- rebuilt env: `/home/amd/.venvs/vlash-rocm`

The rebuilt environment was created successfully with:

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/setup_rocm_env.sh
```

Validation:

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/check_rocm_runtime.sh
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/run_rocm_vlash.sh --help
```

## Direct Policy Inference On AMD GPU

Before CLI benchmarking, the following was confirmed directly:

- load `PI05Policy` onto AMD GPU
- run `predict_action_chunk()` successfully on GPU
- run `compile_model=true` successfully on GPU

Representative direct results:

- uncompiled GPU inference: about `1532 ms`
- compiled steady-state call: about `836 ms`

These were direct Python-path checks, not full CLI benchmark runs.

## Mock `vlash run` Runtime Validation

The main `vlash run` path now works in a local mock environment on AMD ROCm.

What is validated:

- policy load
- compiled warmup
- camera feature remapping
- staged runtime timing
- structured runtime stats output

Example paths:

- [mock_rocm.yaml](/home/amd/vlash/examples/inference/mock_rocm.yaml)
- [mock_rocm_async.yaml](/home/amd/vlash/examples/inference/mock_rocm_async.yaml)

Representative outputs:

- [mock_rocm_sync_stats.json](/home/amd/vlash/outputs/mock_runtime/mock_rocm_sync_stats.json)
- [mock_rocm_async_stats.json](/home/amd/vlash/outputs/mock_runtime/mock_rocm_async_stats.json)

The first mock sweep shows:

- system-level loop cadence improves clearly with `action_quant_ratio`
- staged inference time stays roughly constant around `~1.0s`
- overlap alone does not yet show a strong benefit in this mock setup

The extended sweep adds a more important result:

- shorter chunks and longer runs do activate async chunk switching
- `future_state_mode` transitions from `none` to `last_action_projected`
- overlap is now functionally validated in the mock runtime, even though it still does not produce a large standalone loop-time improvement

The `n_action_steps=4` grid adds the clearest mock-runtime result so far:

- async overlap repeatedly triggers chunk switching
- overlap benefit is not monotonic
- the effective tuning constraint is:
  `effective_overlap_steps = inference_overlap_steps * action_quant_ratio <= n_action_steps`

For details, see:

- [MOCK_RUNTIME_SWEEP.md](/home/amd/vlash/MOCK_RUNTIME_SWEEP.md)
- [MOCK_RUNTIME_EXTENDED_SWEEP.md](/home/amd/vlash/MOCK_RUNTIME_EXTENDED_SWEEP.md)
- [MOCK_RUNTIME_N4_GRID.md](/home/amd/vlash/MOCK_RUNTIME_N4_GRID.md)

## End-To-End CLI Benchmark Results

All results below use:

- checkpoint: `/home/amd/.cache/vlash/models/vlash-pi05-libero-async5`
- config: [inference_latency_rocm_smoke.yaml](/home/amd/vlash/examples/benchmarks/inference_latency_rocm_smoke.yaml)
- environment: `/home/amd/.venvs/vlash-rocm`

### Non-Compiled Path

Single-sample end-to-end results were noisy but functional:

- one run: `2070.65 ms`
- earlier probe-env run: `1426.67 ms`

Conclusion:

- non-compiled path works, but is substantially slower

### Compiled Path

Single-sample runs showed around `780 ms`, but multi-sample benchmarking is more reliable.

Multi-sample result with `compile_model=true`, `num_samples=5`:

- mean: `711.18 ms`
- median: `694.79 ms`
- std: `35.50 ms`
- min: `687.52 ms`
- max: `781.31 ms`
- throughput: `1.41 FPS`

Practical summary:

- compiled steady-state latency on this hardware is about `690-710 ms`

## Profiling Result

A compiled single-step profiler run showed:

- CPU total: about `769.5 ms`
- CUDA total: about `687.8 ms`
- almost all time is inside the compiled model region

Main conclusion:

- the current bottleneck is model compute on GPU
- not Python wrapper overhead
- not batch remapping overhead
- not host-side control logic

The top hot kernels were large fused Triton/GEMM kernels typical of transformer and MLP compute.

## Optimization Knobs Tested

### `compile_model`

This is the only optimization that clearly helps on this machine.

- non-compiled: clearly slower
- compiled: about `700 ms` steady state

Conclusion:

- keep `compile_model=true`

### `ROCM_EXPERIMENTAL_ATTENTION=1`

Tested by exporting:

```bash
ROCM_EXPERIMENTAL_ATTENTION=1
```

Result:

- compiled latency changed from `781.07 ms` to `780.93 ms` in the single-sample comparison

Conclusion:

- no meaningful gain observed

### `fuse_qkv` And `fuse_gate_up`

These are real model-path optimizations, enabled in:

- [vlash/policies/pi05/modeling_pi05.py](/home/amd/vlash/vlash/policies/pi05/modeling_pi05.py)
- [vlash/policies/pi0/modeling_pi0.py](/home/amd/vlash/vlash/policies/pi0/modeling_pi0.py)

Measured on AMD ROCm with `compile_model=true`, `num_samples=5`:

- `fuse_qkv=false`, `fuse_gate_up=false`
  - mean `713.79 ms`
  - median `705.44 ms`
- `fuse_qkv=true`, `fuse_gate_up=false`
  - mean `744.30 ms`
  - median `728.36 ms`
- `fuse_qkv=false`, `fuse_gate_up=true`
  - mean `755.19 ms`
  - median `744.71 ms`
- `fuse_qkv=true`, `fuse_gate_up=true`
  - mean `713.24 ms`
  - median `705.47 ms`

Conclusion:

- single enabling either fusion makes things worse
- enabling both together is effectively tied with disabling both
- on this AMD ROCm setup there is no compelling latency win from these fusions

Practical setting:

- `compile_model=true`
- `fuse_qkv=false`
- `fuse_gate_up=false`

### `action_quant_ratio`

This was analyzed in code, not with real robot hardware.

Relevant path:

- [vlash/run.py](/home/amd/vlash/vlash/run.py)

Important distinction:

- `action_quant_ratio` changes action send cadence in the control loop
- it does not reduce raw `predict_action_chunk()` model latency

Conclusion:

- it is a system-level control/execution knob, not a pure model inference latency knob
- without real robot hardware, its practical benefit cannot be quantified correctly here

## Comparison To Paper-Level Results

Current local results do not match the paper's best deployment figures.

Paper highlights:

- up to `17.4x` reaction speedup in asynchronous setting
- `RTX 5090` table reports `30.4 ms` async reaction latency for `pi0` in the cited setup
- paper also discusses high-frequency real-world control on much stronger discrete NVIDIA GPUs

Source:

- https://arxiv.org/html/2512.01031v1

Current local best:

- compiled `pi05` steady-state latency about `690-710 ms`

Therefore:

- this work proves VLASH can run on AMD ROCm on the local machine
- it does not reproduce the paper's latency level

This gap is expected because:

- different hardware class: integrated Radeon 890M vs high-end NVIDIA GPUs
- different model/configuration: local work used public `pi05` checkpoint
- current benchmark is compatibility-oriented, not a task-faithful reproduction of the paper setup

## Current Best Known Settings On This Machine

For AMD ROCm inference on this machine, the best known settings are:

```yaml
policy:
  device: cuda
  dtype: bfloat16
  compile_model: true
  fuse_qkv: false
  fuse_gate_up: false
```

Runtime:

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm
export LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}
```

## Recommended Next Steps

Most valuable next directions:

1. Evaluate a smaller policy/checkpoint on AMD ROCm to measure the latency-vs-model-size slope.
2. Validate the real `vlash run` runtime path on AMD GPU with actual robot hardware or a strong robot stub.
3. If system-level async behavior matters, evaluate `action_quant_ratio=2/3` only in a real control-loop setting rather than benchmark-only mode.
