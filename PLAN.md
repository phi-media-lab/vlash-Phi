# pi05-rocm-vlash Plan

Last updated: 2026-04-11

## Purpose

This document defines the next implementation plan for the `pi05-rocm-vlash` branch.

The branch has already completed AMD ROCm bring-up:

- ROCm torch can use the local AMD GPU
- `PI05Policy.from_pretrained(...)` runs on GPU
- `vlash benchmark` runs end-to-end
- `compile_model=true` is confirmed useful

The next stage is not more bring-up. The next stage is runtime restructuring.

## Current Position

What is already true:

- AMD ROCm compatibility is established
- the main latency bottleneck is model compute, not Python wrapper overhead
- `compile_model=true` is the only optimization knob with clear benefit on this machine
- `fuse_qkv` and `fuse_gate_up` do not provide meaningful gain on this AMD setup
- the current codebase already exposes natural `prefix / suffix` boundaries inside `PI05Model`

What is not yet true:

- runtime is still organized around whole-chunk black-box calls
- future-state-awareness still uses a simplified surrogate
- `vlash run` has not yet been validated on real robot hardware
- NPU-oriented work would be premature before runtime boundaries are made explicit

## Guiding Decision

The immediate goal is to build a **no-NPU baseline v1**.

That baseline should:

- preserve current behavior where possible
- make runtime phases explicit
- make prefix/suffix execution measurable
- support later UMA-aware optimization
- support later suffix-only NPU offload

This means:

- do not prioritize more ROCm micro-tuning now
- do not prioritize NPU offload now
- do not treat benchmark-only latency as the final deployment metric

## Progress Update

The branch has now partially completed `P1` and `P2`:

- staged `PI05` runtime interfaces exist
- `vlash run` emits structured stage timings
- mock robot runtime exists for local `vlash run` validation
- benchmark and runtime now share the same `stage_timings_ms` structure
- first mock runtime sweep has been completed
- extended mock runtime sweep has exercised async chunk switching
- `n_action_steps=4` grid has shown that overlap benefit is not monotonic
- a `LIBERO` simulator adapter now exists for the `vlash run` main loop
- `vlash run` now works on AMD ROCm with a simulator, not only a mock runtime
- simulator runs show overlap becomes active once chunk length is short enough

The next gap is no longer "can `vlash run` start on AMD ROCm?"

The next gap is:

- validating overlap/chunk handoff under a longer or more demanding simulator runtime
- then validating the same path with real hardware rather than simulator or mock runtime

More specifically:

- overlap is now functionally active in mock runtime
- overlap is now functionally active in `LIBERO` simulation when `n_action_steps` is reduced
- the remaining unknown is whether it produces a meaningful system-level gain under more realistic timing pressure
- the current mock evidence suggests the feasible region must obey:
  `inference_overlap_steps * action_quant_ratio <= n_action_steps`

Current best simulator-backed evidence:

- with `n_action_steps=32`, async improves loop cadence but does not yet switch chunks
- with `n_action_steps=8`, async `q2/o2` produces `chunk_switches=1`
- `last_action_projected` is now exercised in both mock and simulator paths

## Phase Overview

The work will proceed in four phases:

1. `P0`: Freeze the current ROCm baseline
2. `P1`: Restructure the no-NPU runtime baseline
3. `P2`: Validate the real `vlash run` path on AMD GPU
4. `P3`: Prepare suffix-only NPU artifact after baseline v1 is stable

## P0: Freeze The Current Baseline

### Goal

Turn the current ROCm work into a stable baseline reference so every future change has a known comparison point.

### Required Outputs

- preserve current helper scripts and docs
- preserve current smoke benchmark path
- preserve current best-known config and measured results
- explicitly label the state as `rocm-baseline-v0`

### Effective Baseline Settings

Current best-known inference settings on this machine:

```yaml
policy:
  device: cuda
  dtype: bfloat16
  compile_model: true
  fuse_qkv: false
  fuse_gate_up: false
```

### Files Already Serving P0

- [README.md](/home/amd/vlash/README.md)
- [README.zh-CN.md](/home/amd/vlash/README.zh-CN.md)
- [ROCM_VLASH_STATUS.md](/home/amd/vlash/ROCM_VLASH_STATUS.md)
- [tools/setup_rocm_env.sh](/home/amd/vlash/tools/setup_rocm_env.sh)
- [tools/check_rocm_runtime.sh](/home/amd/vlash/tools/check_rocm_runtime.sh)
- [tools/run_rocm_vlash.sh](/home/amd/vlash/tools/run_rocm_vlash.sh)
- [examples/benchmarks/inference_latency_rocm_smoke.yaml](/home/amd/vlash/examples/benchmarks/inference_latency_rocm_smoke.yaml)

### Exit Criteria

- a new contributor can rebuild the ROCm environment and rerun the smoke benchmark
- baseline settings and expected performance are documented in one place

## P1: Restructure The No-NPU Baseline

### Goal

Convert the current black-box chunk runtime into a staged runtime with explicit boundaries:

- observation ingress
- prefix prefill
- suffix rollout
- chunk handoff

### Why This Is The Highest-Priority Work

Current AMD results already show:

- most time is spent inside compiled model execution
- wrapper overhead is not the dominant bottleneck
- benchmark compatibility issues are largely under control

So the highest-value next work is to make the runtime analyzable and separable, not to keep tweaking low-level fusion flags.

### P1.1: Introduce Runtime Data Objects

Add four explicit runtime objects:

- `ObservationSlot`
- `PrefixContext`
- `ChunkSlot`
- `SuffixRolloutEngine`

#### Intended Responsibilities

`ObservationSlot`

- owns the latest observation payload
- separates ingress state from inference state
- records timestamps and source metadata

`PrefixContext`

- owns prefix KV cache and related masks
- makes prefix data explicit instead of leaving it only inside module-local cache
- records dtype, device, valid lengths, and shape metadata

`ChunkSlot`

- owns one predicted action chunk and associated metadata
- can be produced by inference and consumed by execution
- becomes the handoff contract between workers

`SuffixRolloutEngine`

- owns suffix rollout execution over a built prefix context
- exposes timing and step-level profiling
- becomes the later insertion point for suffix-only backend variants

### P1.2: Refactor `PI05` Model Entry Points

Primary file:

- [vlash/policies/pi05/modeling_pi05.py](/home/amd/vlash/vlash/policies/pi05/modeling_pi05.py)

#### Current Situation

The internal model already contains usable boundaries:

- prefix embedding path
- suffix embedding path
- prefix KV cache behavior
- `denoise_step()`
- whole-chunk sampling entry point

#### Required Refactor

Keep `predict_action_chunk()` for backward compatibility, but add explicit staged methods:

- `build_prefix_context(...)`
- `rollout_suffix(...)`
- optional helper for preparing state/noise per suffix step

#### Rules

- do not change model semantics in the first pass
- do not remove existing public inference methods
- staged methods should be thin extractions of already existing logic

### P1.3: Make Attention Cache Externally Legible

Primary file:

- [vlash/layers/attention.py](/home/amd/vlash/vlash/layers/attention.py)

#### Problem

The cache exists today, but it is held as implicit module state:

- `self.k_cache`
- `self.v_cache`

That is acceptable for single-path bring-up, but not enough for:

- explicit prefix contexts
- per-phase profiling
- future cross-device suffix execution

#### Required Refactor

Expose cache lifecycle more explicitly:

- reset
- initialize
- read/export
- reuse

This does not need a full redesign in the first pass. It only needs enough structure for `PrefixContext` to become real.

### P1.4: Upgrade `VLASHAsyncManager`

Primary file:

- [vlash/run.py](/home/amd/vlash/vlash/run.py)

#### Current Situation

The current manager is still effectively:

- current chunk
- next chunk
- overlap-triggered whole-chunk inference

#### Required Refactor

Separate:

- execution loop
- inference worker
- observation ingress
- prefix prefill timing
- suffix rollout timing

The first version does not need threads or processes. Logical separation is enough.

#### Target Runtime Shape

- execution consumes `ChunkSlot`
- inference produces `ChunkSlot`
- observation updates `ObservationSlot`
- inference uses `PrefixContext`

### P1.5: Replace Simplified Future-State Surrogate

Primary file:

- [vlash/run.py](/home/amd/vlash/vlash/run.py)

#### Current Situation

Current future-state-awareness uses a surrogate:

- replace `observation.state` with the last action of the current chunk

This is directionally correct, but still a simplification.

#### Required Change

Introduce `rollforward_state(...)`:

- input: current state
- input: relevant actions during delay window
- input: robot/control metadata
- output: estimated execution-time state

#### Implementation Strategy

Phase 1 version can be approximate:

- start with a position-control approximation
- keep current surrogate as fallback/debug path

### P1.6: Add Per-Stage Profiling

Primary files:

- [vlash/run.py](/home/amd/vlash/vlash/run.py)
- [benchmarks/benchmark_inference_latency.py](/home/amd/vlash/benchmarks/benchmark_inference_latency.py)

#### Goal

Measure:

- observation preprocess time
- prefix prefill time
- suffix rollout time
- chunk handoff time

This is the minimum instrumentation required before considering NPU suffix offload.

### P1 Exit Criteria

P1 is complete when all of the following are true:

- runtime phases are explicit in code
- prefix context is a real runtime object
- suffix rollout can be timed independently
- `rollforward_state()` exists and can replace the current surrogate
- old benchmark and inference entry points still work

## P2: Validate The Real `vlash run` Path

### Goal

Promote `vlash run` from "not yet validated on AMD ROCm" to "measured, usable runtime path."

### Why This Matters

Only the real `vlash run` path can validate:

- async overlap behavior
- `action_quant_ratio`
- control-loop timing
- camera ingress overhead
- chunk handoff behavior in deployment-style flow

### Primary Files

- [vlash/run.py](/home/amd/vlash/vlash/run.py)
- [vlash/configs/run_config.py](/home/amd/vlash/vlash/configs/run_config.py)
- [examples/inference/async.yaml](/home/amd/vlash/examples/inference/async.yaml)
- [examples/inference/sync.yaml](/home/amd/vlash/examples/inference/sync.yaml)

### Required Work

- ensure `compile_model=true` path is stable in `vlash run`
- keep `camera_feature_map` behavior stable
- add stage-aware profiling to runtime logs
- validate async overlap logic on AMD GPU

### Exit Criteria

- `vlash run` launches successfully in the AMD ROCm environment
- stage timings can be observed in a real runtime loop
- async overlap and chunk scheduling do not regress relative to the current baseline
- simulator-backed runs demonstrate the runtime behavior outside the mock-only path

Current status:

- mock `vlash run` launch: done
- structured stage timings: done
- first mock sweep: done
- `LIBERO` simulator adapter and ROCm run path: done
- real hardware validation: not done

## P3: Prepare Suffix-Only NPU Artifact

### Goal

Prepare the codebase for suffix-only offload after the no-NPU baseline is clean and measurable.

### Important Non-Goals

Do not:

- offload the whole model
- attempt automatic partitioning first
- treat NPU as the next step before runtime restructuring

### Intended Partition

Keep on iGPU:

- prefix prefill
- prefix context construction
- cache ownership

Move later to suffix-only backend:

- suffix embedder path
- shared suffix rollout layers
- `action_out_proj`

### Why This Ordering Matters

NPU work is only meaningful after:

- prefix context is explicit
- suffix rollout is standalone
- per-stage timing exists
- chunk handoff is explicit

### Exit Criteria

- suffix rollout boundary is explicit enough that a different backend could consume it
- there is no need to inspect hidden module cache state to run suffix rollout

## Validation Matrix

Every phase should be validated against the current AMD baseline.

### Required Comparisons

- P0 baseline vs P1 runtime refactor
- P1 benchmark timings vs `vlash run` timings
- future-state surrogate vs `rollforward_state()`
- later suffix-only backend vs P1 no-NPU baseline

### Required Stability Checks

- checkpoint loading still works
- benchmark still works
- runtime image remapping still works
- ROCm environment scripts still work

## Recommended Implementation Order

The practical implementation order should be:

1. extract staged `PI05` runtime interfaces
2. make attention cache externally legible
3. add `PrefixContext`, `ChunkSlot`, and `ObservationSlot`
4. refactor `VLASHAsyncManager`
5. implement `rollforward_state()`
6. add per-stage profiling
7. validate `vlash run`
8. only then start suffix-only NPU work

## Immediate Next Action

The next concrete work item should be:

**Implement the first pass of P1 in [vlash/policies/pi05/modeling_pi05.py](/home/amd/vlash/vlash/policies/pi05/modeling_pi05.py) and [vlash/run.py](/home/amd/vlash/vlash/run.py), without changing external behavior.**

That means:

- keep existing CLI and benchmark commands working
- add staged runtime interfaces alongside current methods
- introduce explicit runtime objects without breaking current flow
