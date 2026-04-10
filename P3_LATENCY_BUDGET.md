# P3 Latency Budget

Last updated: 2026-04-11

## Purpose

This document turns the current `P3-prototype` evidence into a practical
latency-budget framework for future suffix-only NPU work.

The question is not "can we offload suffix-only execution?" The current branch
has already shown that we can preserve correctness across progressively stricter
backend boundaries.

The question is:

- how much extra dispatch cost can suffix-only execution absorb,
- before the async/runtime value of that split starts to erode?

## Current Baseline Context

The current simulator-backed baseline recommendation is:

```yaml
policy:
  n_action_steps: 8

action_quant_ratio: 2
inference_overlap_steps: 1
```

On this branch, that means:

- `effective_overlap_steps = inference_overlap_steps * action_quant_ratio = 2`
- async value mainly comes from chunk scheduling rather than a faster single inference
- stage timing is dominated by model execution, especially the suffix phase

Representative no-delay local-like runtime numbers under this baseline:

| backend | loop_avg_ms | prefix_mean_ms | suffix_mean_ms | total_mean_ms |
|---|---:|---:|---:|---:|
| `local` | `1054.43` | `186.42` | `859.32` | `1907.20` |
| `numpy_local` | `1022.90` | `183.87` | `844.64` | `1844.49` |

These numbers show the current structure clearly:

- suffix is the largest single stage
- prefix is substantial but smaller
- wrapper/runtime overhead is not the main bottleneck

## What Determines The Dispatch Budget

The practical dispatch budget is set by six things.

### 1. Control-loop time budget

The runtime only benefits from suffix-only offload if the result returns before
chunk scheduling loses value.

The effective control budget is shaped by:

- `fps`
- `n_action_steps`
- `action_quant_ratio`
- `inference_overlap_steps`

In other words, the budget is not a generic number. It is configuration-specific.

### 2. Suffix share of total inference

If suffix dominates total inference time, there is more room to justify moving it.

Current evidence on this branch says:

- suffix is the largest stage
- prefix is meaningful but smaller
- transport overhead must stay small enough that suffix-only split still preserves system value

### 3. Payload shape and transport format

Dispatch cost depends on:

- how much data is moved
- whether dtype conversion is required
- whether CPU round-trips are required
- whether the transport format is tensor, numpy, bytes, or queue-based

This is exactly why the branch now has a backend ladder:

- `serialized_local`
- `numpy_local`
- `dispatched_numpy_local`
- `queued_dispatched_numpy_local`
- `delayed_queued_dispatched_numpy_local`

### 4. Scheduling semantics

Dispatch cost is not only about raw transfer time.

It also depends on:

- whether dispatch is synchronous
- whether queueing is involved
- whether result return can overlap with other work
- whether request/response introduces new synchronization points

### 5. Fixed backend overhead

A real NPU path will usually add fixed costs such as:

- graph load / compile
- buffer preparation
- submission overhead
- result synchronization

Even if transport is cheap, fixed per-call cost can still make suffix-only offload unattractive.

### 6. Chunk handoff sensitivity

The final cost is determined by what happens after the result returns:

- does it still arrive in time for chunk switch?
- does it still preserve overlap value?
- does it start to inflate loop cadence?

That is why `loop_avg_ms` and `chunk_switches` matter alongside stage timings.

## Measured Delay Sensitivity

The branch now includes a delayed queued backend:

- `delayed_queued_dispatched_numpy_local`

This backend keeps the queued request/response structure but injects an
artificial dispatch delay:

- `policy.suffix_backend_dispatch_delay_ms`

Measured results under the current simulator baseline:

| dispatch_delay_ms | max_abs_diff | loop_avg_ms | suffix_mean_ms | total_mean_ms |
|---|---:|---:|---:|---:|
| `0` | `0.0` | `1038.33` | `856.27` | `1875.08` |
| `5` | `0.0` | `1038.05` | `853.91` | `1874.66` |
| `20` | `0.0` | `1056.47` | `872.54` | `1911.26` |

Source files:

- [delayed_sweep_0ms.json](/home/amd/vlash/outputs/libero_runtime/delayed_sweep_0ms.json)
- [delayed_sweep_5ms.json](/home/amd/vlash/outputs/libero_runtime/delayed_sweep_5ms.json)
- [delayed_sweep_20ms.json](/home/amd/vlash/outputs/libero_runtime/delayed_sweep_20ms.json)

## Current Interpretation

The current evidence suggests:

- `0-5 ms` extra dispatch cost is mostly hidden at the present simulator setting
- `20 ms` extra dispatch cost starts to move both loop and stage timings upward
- correctness remains intact across all tested delays so far

This does **not** yet mean "20 ms is the hard limit."

It means:

- the current runtime has some slack for small dispatch overhead
- the branch has now reached a point where transport delay can be observed as a real system cost
- larger remote-backend costs will need more careful justification

## Working Budget Bands

Based on the current simulator evidence, the branch can use these working bands:

### Green band

Approximate range:

- `0-5 ms` added dispatch latency

Interpretation:

- likely acceptable for further NPU-oriented prototyping
- currently does not visibly degrade loop cadence under the tested configuration

### Yellow band

Approximate range:

- `5-20 ms`

Interpretation:

- probably still usable for experimentation
- should be treated as a budget that needs explicit measurement rather than assumption

### Red band

Approximate range:

- `>20 ms`

Interpretation:

- likely to start eroding suffix-only value unless other runtime assumptions change
- should not be assumed acceptable without stronger evidence

These bands are not final deployment claims. They are prototype guidance bands.

## When NPU Work Becomes Worth It

It becomes worth starting **NPU-oriented implementation work** when all of the following are true:

1. The suffix contract is stable enough.
   This branch is close to satisfying that already.

2. The expected NPU-side dispatch cost appears to land in the green or low-yellow band.

3. The target NPU integration scope is narrow.
   For example:
   - suffix-only rollout worker
   - not full-model migration
   - not speculative broad offload

4. The first NPU milestone is framed as an integration spike, not as an optimization claim.

## Immediate Practical Rule

For this branch, the current practical rule is:

- NPU design work is justified now
- full NPU implementation work is justified only if the expected dispatch path can plausibly stay near the current small-delay band

In short:

**the branch is ready for NPU-oriented spikes, but real NPU development should stay narrow until transport overhead is shown to fit inside the current prototype budget.**
