# LIBERO System Comparison Plan

Last updated: 2026-04-22

## Purpose

This document defines the next evaluation stage after the shared `pi05_base` core-model bridge work.

The current branch has already answered the model-level question:

- `openpi pi05_base -> vlash` can be bridged successfully
- both implementations can produce finite outputs on the same machine
- retraining is not the first step

The next question is different:

- which **complete system** is better suited for actual `LIBERO` task execution on this machine

This plan is for that system-level comparison.

## What This Plan Is Comparing

This is **not** the shared-checkpoint core benchmark.

This plan compares two complete task-facing systems:

1. `openpi` system
   - task checkpoint: `pi05_libero`
   - full `openpi` preprocessing, transforms, and task-facing action path
2. `vlash` system
   - task checkpoint: `mit-han-lab/vlash-pi05-libero-async5`
   - full `vlash` preprocessing, runtime, and task-facing action path

The result will answer:

- which system is easier to bring up for `LIBERO`
- which system is faster in realistic control-loop conditions
- which system is more stable over repeated episodes
- which system is better suited for future work on this hardware

## Non-Goals

This plan does **not** try to preserve strict checkpoint equivalence.

That work already belongs to the shared `pi05_base` bridge path and should stay separate.

This plan does **not** claim:

- paper-faithful reproduction
- final deployment readiness
- conclusions that automatically transfer to `MI300X`

## Fixed Evaluation Scope

To keep the comparison meaningful, the following must be held fixed:

- machine: local `gfx1150 / Radeon 890M`
- simulator backend: `LIBERO`
- task subset: same task list for both systems
- episode seeds: same seed list for both systems
- observation cadence: same control frequency and simulator step cadence
- evaluation budget: same number of warmup episodes and measured episodes
- output logging: same JSON schema for metrics and failure modes

## Comparison Units

### A. Runtime Comparison

This is for latency and stability, not success rate.

Metrics:

- model load time
- first actionable output latency
- warm-cache first-call latency
- steady-state loop latency
- per-step loop jitter
- GPU memory footprint
- crash / timeout / numerical failure count

### B. Task Comparison

This is for actual `LIBERO` usefulness.

Metrics:

- task success rate
- average episode return where available
- episode length to success
- timeout rate
- action validity failures
- total wall-clock evaluation time

## Required Preconditions

Before running the full comparison, each system must pass all of:

1. checkpoint load succeeds on the target machine
2. first action output is finite
3. one short `LIBERO` episode runs without crash
4. metrics can be emitted in the shared output schema

If any system fails these gates, the comparison stops and the blocker is recorded first.

## System Definitions

### `openpi`

Expected task-facing configuration:

- checkpoint family: `pi05_libero`
- task adapter: `openpi` `libero_policy` path
- execution path: whichever `openpi` route is currently the most stable on this machine

Record explicitly for every run:

- backend (`PyTorch/ROCm`, `JAX/ROCm Docker`, or other)
- dtype
- compile settings
- startup/warmup strategy

### `vlash`

Expected task-facing configuration:

- checkpoint: `mit-han-lab/vlash-pi05-libero-async5`
- runtime config family: `examples/inference/libero_rocm*.yaml`
- current baseline settings:
  - `compile_model: true`
  - `compile_mode: max-autotune`
  - `staged_compile_mode: max-autotune-no-cudagraphs`
  - `fuse_qkv: false`
  - `fuse_gate_up: false`

## Evaluation Phases

### Phase 0: Freeze Current Baselines

Record the exact branch, commit, checkpoint path, and environment for:

- `openpi`
- `vlash`

Output:

- one markdown snapshot of branches / commits / checkpoints / envs

### Phase 1: Build A Common `LIBERO` Harness

The comparison should not rely on two unrelated ad-hoc scripts.

Needed:

- one shared evaluation driver
- one shared metrics JSON schema
- one adapter per system:
  - `OpenPIPolicyAdapter`
  - `VLASHPPolicyAdapter`

Each adapter should expose the same minimal interface:

- `load()`
- `reset_episode()`
- `act(observation) -> action`
- `close()`

Output:

- a single harness that can swap systems without changing task logic

### Phase 2: Runtime Validation

Run a short `LIBERO` subset with low episode count only to validate runtime behavior.

Target output:

- startup latency
- first-step latency
- steady-state step latency
- failure counts
- memory behavior

This phase should answer:

- can both systems finish repeated short episodes
- which system has the better runtime profile on this machine

### Phase 3: Task Evaluation

Run a fixed task set with a fixed seed set.

Recommended first pass:

- 3 tasks
- 5 seeds each
- same max episode length

Target output:

- per-task success rate
- per-seed outcome table
- aggregate success rate
- failure mode summary

### Phase 4: Decision Summary

Produce one comparison summary that separates:

- runtime winner
- task-success winner
- operational simplicity winner
- recommended next branch for continued work

## Task Set Recommendation

Start with a small, stable subset rather than the full benchmark.

Recommended first pass:

- 3 representative `LIBERO` tasks
- 5 seeds each
- fixed camera and reset configuration

Only expand to a larger sweep after both systems complete the first pass cleanly.

## Output Layout

Store results under one common directory tree, for example:

```text
outputs/system_compare/
  openpi/
  vlash/
  summaries/
```

Each measured run should write:

- config snapshot
- checkpoint reference
- metrics JSON
- stderr/stdout log
- final summary row

## Exit Criteria

This plan is complete when we have:

1. both systems running through the same `LIBERO` harness
2. the same task set and seed set completed for both systems
3. one summary table covering runtime and task outcomes
4. one recommendation stating which system is better for actual `LIBERO` work on `gfx1150`

## Current Recommendation

Based on current evidence, the next engineering move should be:

1. keep the shared `pi05_base` bridge results as a separate model-core reference
2. switch full-system comparison to task-level `LIBERO` checkpoints
3. compare `openpi` and `vlash` through one common `LIBERO` evaluation harness

That is the shortest path to answering the real question:

- not "which model core is faster"
- but "which complete system is better suited for actual tasks on this machine"
