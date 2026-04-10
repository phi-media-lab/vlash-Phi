# Mock Runtime N=4 Grid

Last updated: 2026-04-10

## Purpose

This document summarizes the third mock runtime sweep, which pushed the system into
shorter chunk cycles by setting:

- `policy.n_action_steps = 4`
- `control_time_s = 8`

The goal was to find when async overlap becomes clearly active at the scheduling level,
and whether larger overlap values produce better system-level results.

## Configurations Run

Completed runs:

| case | action_quant_ratio | inference_overlap_steps | n_action_steps | control_time_s |
|---|---:|---:|---:|---:|
| `sync_q1_o0_n4_t8` | 1 | 0 | 4 | 8 |
| `sync_q2_o0_n4_t8` | 2 | 0 | 4 | 8 |
| `async_q1_o1_n4_t8` | 1 | 1 | 4 | 8 |
| `async_q1_o2_n4_t8` | 1 | 2 | 4 | 8 |
| `async_q1_o3_n4_t8` | 1 | 3 | 4 | 8 |
| `async_q2_o1_n4_t8` | 2 | 1 | 4 | 8 |
| `async_q2_o2_n4_t8` | 2 | 2 | 4 | 8 |

Invalid configuration discovered during sweep:

| case | action_quant_ratio | inference_overlap_steps | effective_overlap_steps | n_action_steps | outcome |
|---|---:|---:|---:|---:|---|
| `async_q2_o3_n4_t8` | 2 | 3 | 6 | 4 | rejected by runtime assertion |

## Results

| case | loop iters | obs fetches | inference launches | chunk switches | future state | loop avg ms | chunk handoff avg ms | stage total ms |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| `async_q1_o1_n4_t8` | 19 | 5 | 5 | 4 | `last_action_projected` | 424.36 | 28.59 | 1044.70 |
| `async_q1_o2_n4_t8` | 19 | 6 | 6 | 4 | `last_action_projected` | 465.32 | 1.82 | 1034.83 |
| `async_q1_o3_n4_t8` | 18 | 6 | 6 | 4 | `last_action_projected` | 478.70 | 1.92 | 1030.65 |
| `async_q2_o1_n4_t8` | 19 | 6 | 6 | 4 | `last_action_projected` | 422.70 | 1.88 | 1031.90 |
| `async_q2_o2_n4_t8` | 18 | 5 | 6 | 4 | `last_action_projected` | 446.74 | 1.58 | 1034.06 |
| `sync_q1_o0_n4_t8` | 19 | 5 | 5 | 0 | `none` | 431.50 | 0.00 | 1046.31 |
| `sync_q2_o0_n4_t8` | 21 | 6 | 6 | 0 | `none` | 403.74 | 0.00 | 1046.69 |

## Main Findings

1. With `n_action_steps=4`, async overlap is now unquestionably active at the scheduling level.
2. Every async run produced:
   - `chunk_switches = 4`
   - `future_state_mode = last_action_projected`
3. Every sync run still produced:
   - `chunk_switches = 0`
   - `future_state_mode = none`
4. This confirms that async overlap is not merely enabled in configuration; it is changing runtime behavior under this shorter-chunk regime.
5. However, overlap benefit is not monotonic:
   - `o1` is better than `o2`/`o3` in some cases
   - larger overlap does not automatically improve loop cadence
6. Stage-level inference time remains almost constant across the grid:
   - roughly `1031-1047 ms`

## Interpretation

This sweep is the strongest mock-runtime evidence so far.

What it proves:

- overlap changes scheduling behavior
- overlap influences chunk handoff
- future-state rollforward is exercised repeatedly

What it does not prove:

- that larger overlap always helps
- that overlap alone dominates system-level performance

The current evidence instead suggests:

- `action_quant_ratio` is still the stronger control knob for loop cadence
- overlap has a useful operating region, but can overshoot
- the operating region depends on the effective relation:

```text
effective_overlap_steps = inference_overlap_steps * action_quant_ratio
```

## Important Constraint

The sweep discovered an operational boundary that should be treated as a design rule:

```text
effective_overlap_steps <= n_action_steps
```

Otherwise runtime initialization fails.

In practical terms, this means the overlap knob cannot be tuned independently.
It is coupled to:

- `action_quant_ratio`
- `n_action_steps`

## Current Best Reading

Under this mock `n_action_steps=4` regime:

- async overlap is real and repeated
- modest overlap is preferable to aggressive overlap
- `o1` looks safer than `o2/o3`
- `q2/o0` is still very competitive as a system-level baseline

## Recommended Next Step

The most useful next experiment is no longer "make overlap larger."

It is:

- search the feasible region more deliberately
- keep `effective_overlap_steps` bounded tightly
- compare `q2/o0`, `q2/o1`, and maybe `q1/o1` over longer windows

After that, the next major step remains real hardware validation.
