# Mock Runtime Extended Sweep

Last updated: 2026-04-10

## Purpose

This document summarizes the second mock `vlash run` sweep, designed to stress
chunk scheduling more than the initial short-window sweep.

Compared with the first sweep:

- `control_time_s` was increased from `2` to `6`
- `policy.n_action_steps` was reduced from `32` to `8`

The goal was to trigger more:

- inference launches
- chunk transitions
- future-state rollforward usage

## Configurations Swept

| case | action_quant_ratio | inference_overlap_steps | n_action_steps | control_time_s |
|---|---:|---:|---:|---:|
| `sync_q1_o0_n8_t6` | 1 | 0 | 8 | 6 |
| `sync_q2_o0_n8_t6` | 2 | 0 | 8 | 6 |
| `async_q1_o2_n8_t6` | 1 | 2 | 8 | 6 |
| `async_q2_o2_n8_t6` | 2 | 2 | 8 | 6 |

## Results

| case | loop iters | obs fetches | inference launches | chunk switches | future state | loop avg ms | chunk handoff avg ms | stage total ms |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| `async_q1_o2_n8_t6` | 18 | 3 | 3 | 2 | `last_action_projected` | 340.44 | 2.27 | 1030.68 |
| `async_q2_o2_n8_t6` | 21 | 4 | 4 | 2 | `last_action_projected` | 292.71 | 2.03 | 1028.62 |
| `sync_q1_o0_n8_t6` | 17 | 3 | 3 | 0 | `none` | 353.19 | 0.00 | 1034.82 |
| `sync_q2_o0_n8_t6` | 25 | 4 | 4 | 0 | `none` | 268.08 | 0.00 | 1042.29 |

Raw artifacts:

- [summary.md](/home/amd/vlash/outputs/mock_runtime/extended/summary.md)
- [sync_q1_o0_n8_t6.json](/home/amd/vlash/outputs/mock_runtime/extended/sync_q1_o0_n8_t6.json)
- [sync_q2_o0_n8_t6.json](/home/amd/vlash/outputs/mock_runtime/extended/sync_q2_o0_n8_t6.json)
- [async_q1_o2_n8_t6.json](/home/amd/vlash/outputs/mock_runtime/extended/async_q1_o2_n8_t6.json)
- [async_q2_o2_n8_t6.json](/home/amd/vlash/outputs/mock_runtime/extended/async_q2_o2_n8_t6.json)

## Main Findings

1. This extended sweep finally exercised chunk switching in the async path.
2. `async_q1_o2_n8_t6` and `async_q2_o2_n8_t6` both produced:
   - `chunk_switches = 2`
   - `future_state_mode = last_action_projected`
3. The sync runs produced the same number of inference launches as their async counterparts in the same window, but no chunk switching.
4. Stage-level inference time remained effectively unchanged:
   - total staged inference stayed around `~1030-1042 ms`
5. `action_quant_ratio=2` still improved loop cadence more than overlap alone.

## Interpretation

This sweep adds a more nuanced conclusion than the first short-window sweep.

Short-window sweep conclusion:

- `action_quant_ratio` matters
- overlap alone does not obviously help in the mock setup

Extended sweep conclusion:

- overlap does start to change runtime behavior when chunk cycles are shorter
- this shows up in `chunk_switches` and `future_state_mode`
- but the current mock runtime still does not show a large standalone loop-time win from overlap

So the current state is:

- overlap is now confirmed to be functionally active in the mock runtime
- overlap is not yet confirmed to produce a strong system-level latency win in the mock runtime

## Why The Result Still Has Limits

Even this extended sweep is still a mock environment:

- no real camera ingestion delay
- no robot execution uncertainty
- no control bus or actuator jitter
- no realistic sensor update skew

This means the extended sweep is good for:

- scheduling validation
- bug discovery
- stage-timing stability

It is still not a substitute for real hardware validation.

## Recommended Next Step

The next highest-value experiment is a third sweep that increases transition pressure further, for example:

- `n_action_steps = 4`
- longer `control_time_s`
- a grid over `inference_overlap_steps = 1, 2, 3`

That should reveal whether overlap starts to produce clearer system-level wins before moving to real hardware.
