# P3 Entry Criteria

Last updated: 2026-04-11

## Purpose

This document defines the minimum bar for moving from `P2.5` into `P3`.

`P3` is the point where the branch stops validating the no-NPU runtime baseline and starts preparing a suffix-only acceleration artifact. The criteria below are meant to prevent that transition from happening too early.

## Required Conditions

All of the following should be true before `P3` begins.

### 1. Runtime Boundary Stability

The staged runtime must already be stable and externally legible.

Required:

- prefix/suffix boundaries exist as explicit runtime concepts
- stage timings are observable in both benchmark and runtime paths
- chunk handoff behavior is measurable
- fallback future-state handling is implemented and exercised

Current status:

- satisfied

### 2. ROCm Baseline Stability

The AMD ROCm baseline must already be reproducible and documented.

Required:

- ROCm environment can be rebuilt from scripts and docs
- public `PI05` checkpoint loads and runs on AMD GPU
- `compile_model=true` path is stable
- known-good inference settings are documented

Current status:

- satisfied

### 3. Simulator-Backed Runtime Recommendation

There must be a simulator-backed no-NPU recommendation, not just isolated ad hoc runs.

Required:

- at least one simulator path runs end to end on AMD ROCm
- sync/async comparisons exist for the same simulator task
- a recommended setting for `n_action_steps`, `action_quant_ratio`, and `inference_overlap_steps` is documented
- the recommendation is supported by at least one follow-up confirmation sweep

Current status:

- satisfied

Current recommended simulator baseline:

```yaml
policy:
  n_action_steps: 8

action_quant_ratio: 2
inference_overlap_steps: 1
```

### 4. Overlap Semantics Must Be Understood

`P3` should not start while overlap behavior is still a black box.

Required:

- overlap has been observed to trigger real chunk switching
- valid and invalid overlap regions are understood
- the team understands whether a gain is due to scheduling or due to lower inference time

Current status:

- satisfied at simulator level
- not yet satisfied at real-hardware level

### 5. Remaining Unknowns Must Be Mostly Sim-to-Real

The main open questions before `P3` should be deployment questions, not basic runtime questions.

Required:

- remaining uncertainty is mostly about transfer to real hardware
- there are no known correctness or stability regressions in the current no-NPU path
- moving to `P3` would add new capability rather than compensate for an unstable baseline

Current status:

- mostly satisfied

## Conditions That Do Not Block P3

These are still incomplete, but they do not have to block `P3` once the required conditions above are met.

- real robot hardware validation
- final task-success evaluation
- polished user-facing docs beyond current engineering status docs
- final NPU backend implementation details

## Conditions That Still Argue For Caution

Even though the branch is close to `P3`, these points still justify restraint:

- simulator evidence is stronger than before, but still not equivalent to real robot behavior
- async improves scheduling behavior more clearly than it improves end-to-end loop time
- the current recommendation is strong enough for backend planning, but not yet a deployment claim

## Current Decision

The branch is now close enough to `P3` to begin design work, but not yet compelled to switch the main line of work fully into `P3`.

Recommended interpretation:

- `P3` design and interface planning: allowed
- `P3` implementation as the main branch focus: wait until the team explicitly decides to leave `P2.5`
