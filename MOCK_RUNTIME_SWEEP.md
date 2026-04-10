# Mock Runtime Sweep

Last updated: 2026-04-10

## Purpose

This document summarizes the first system-level sweep executed on the local mock
`vlash run` path after staged runtime profiling and structured runtime stats were added.

Scope:

- environment: `/home/amd/.venvs/vlash-rocm`
- checkpoint: `/home/amd/.cache/vlash/models/vlash-pi05-libero-async5`
- runtime: local mock robot configs in [mock_rocm.yaml](/home/amd/vlash/examples/inference/mock_rocm.yaml) and [mock_rocm_async.yaml](/home/amd/vlash/examples/inference/mock_rocm_async.yaml)
- output directory: [outputs/mock_runtime/sweep](/home/amd/vlash/outputs/mock_runtime/sweep)

## Configurations Swept

The following combinations were run:

| case | action_quant_ratio | inference_overlap_steps |
|---|---:|---:|
| `sync_q1_o0` | 1 | 0 |
| `sync_q2_o0` | 2 | 0 |
| `async_q1_o4` | 1 | 4 |
| `async_q2_o4` | 2 | 4 |
| `async_q4_o4` | 4 | 4 |

## Results

| case | loop iters | obs fetches | inference launches | future state | loop avg ms | stage total ms | prefix ms | suffix ms |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| `async_q1_o4` | 6 | 1 | 1 | `none` | 342.21 | 1021.87 | 177.94 | 826.19 |
| `async_q2_o4` | 10 | 1 | 1 | `none` | 205.40 | 1022.57 | 178.18 | 827.98 |
| `async_q4_o4` | 17 | 2 | 2 | `last_action_projected` | 170.31 | 1031.79 | 178.65 | 843.37 |
| `sync_q1_o0` | 6 | 1 | 1 | `none` | 345.21 | 1039.75 | 180.75 | 838.29 |
| `sync_q2_o0` | 10 | 1 | 1 | `none` | 205.53 | 1022.65 | 180.25 | 823.83 |

Raw artifacts:

- [summary.md](/home/amd/vlash/outputs/mock_runtime/sweep/summary.md)
- [sync_q1_o0.json](/home/amd/vlash/outputs/mock_runtime/sweep/sync_q1_o0.json)
- [sync_q2_o0.json](/home/amd/vlash/outputs/mock_runtime/sweep/sync_q2_o0.json)
- [async_q1_o4.json](/home/amd/vlash/outputs/mock_runtime/sweep/async_q1_o4.json)
- [async_q2_o4.json](/home/amd/vlash/outputs/mock_runtime/sweep/async_q2_o4.json)
- [async_q4_o4.json](/home/amd/vlash/outputs/mock_runtime/sweep/async_q4_o4.json)

## Main Findings

1. `action_quant_ratio` changes loop cadence much more than overlap in the current mock setup.
2. `sync_q1_o0` and `async_q1_o4` are effectively tied at the system level.
3. `sync_q2_o0` and `async_q2_o4` are also effectively tied at the system level.
4. The per-inference staged compute remains roughly constant across all cases:
   - prefix: about `178-181 ms`
   - suffix: about `824-843 ms`
   - total staged inference: about `1022-1040 ms`
5. `async_q4_o4` is the first case that triggered more than one inference launch inside the 2-second mock run window.

## Interpretation

The current mock runtime is useful for validating:

- chunk scheduling behavior
- structured runtime stats
- future-state surrogate logic
- interaction between `action_quant_ratio` and runtime loop cadence

It is not yet a strong proxy for real overlap benefit, because:

- there is no real camera latency
- there is no robot actuation jitter
- only a small number of inference launches occur in the short mock window
- chunk handoff remains mostly unexercised in the current short-duration runs

So the current sweep supports the following conclusion:

- mock runtime confirms that `action_quant_ratio` can improve system-level loop cadence
- mock runtime does not yet show a clear standalone benefit from `inference_overlap_steps`

## Bug Found During Sweep

The original `async_q4_o4` run exposed a real bug:

- `rollforward_state()` could replace `observation.state` with a raw action vector even when state and action dimensions differed
- on this checkpoint, state is 8D while action is 7D
- this caused the second async inference launch to fail in normalization

This was fixed in [run.py](/home/amd/vlash/vlash/run.py) by introducing a conservative action-to-state projection fallback:

- preserve the current state shape
- overwrite only the shared leading dimensions with the action surrogate

The repaired run now reports:

- `future_state_mode = last_action_projected`

for the `async_q4_o4` case, confirming the fallback is active.

## Recommended Next Step

The next useful step is a deeper mock sweep that increases the number of inference launches and chunk transitions, for example:

- longer `control_time_s`
- smaller `n_action_steps`
- more combinations of `action_quant_ratio` and `inference_overlap_steps`

After that, the next real milestone is to validate the same stats path on a hardware-backed `vlash run`.
