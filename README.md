<!-- markdownlint-disable MD001 MD041 -->

# suffix-dispatch-budget

中文说明见：[README.zh-CN.md](README.zh-CN.md)

This branch is a focused prototype branch for suffix-backend separation work in VLASH.

It is not the generic upstream README, and it is no longer just the AMD ROCm bring-up branch. The point of this branch is to keep the AMD ROCm baseline stable while turning `PI05` suffix rollout into an explicit runtime boundary that can be compared, dispatched, queued, delayed, and eventually replaced by a future heterogeneous backend.

## Branch Goal

This branch exists to answer a specific question:

- how far can the `PI05` suffix path be separated from the current local runtime
- while preserving correctness
- and before dispatch overhead starts to erase the system value of that split

In practice, this branch is about three things:

1. making the suffix rollout boundary explicit
2. validating progressively stricter suffix backend prototypes
3. estimating a practical dispatch-latency budget for future NPU-oriented work

## What This Branch Is For

This branch is the right place for:

- suffix backend interface changes
- dispatch / queue / delayed transport prototypes
- runtime-integrated suffix backend correctness checks
- latency-budget and P3-prototype design work

This branch is not meant to be the plain ROCm bring-up homepage. That baseline is now a prerequisite, not the main story.

## Current Baseline Assumption

The current prototype work still depends on a stable AMD ROCm baseline.

Validated local baseline:

- CPU: `AMD Ryzen AI 9 HX PRO 370`
- GPU: `AMD Radeon 890M`
- ROCm arch: `gfx1150`
- ROCm runtime: `/opt/rocm-7.2.1`
- PyTorch: `2.9.1+rocm7.2.1`

The public checkpoint used for the prototype path is:

- `mit-han-lab/vlash-pi05-libero-async5`

Expected local path:

```bash
/home/amd/.cache/vlash/models/vlash-pi05-libero-async5
```

## What This Branch Adds

Relative to a plain ROCm inference baseline, this branch now adds:

- explicit `prefix -> suffix` runtime staging in `PI05Policy`
- configurable suffix backend selection through config
- backend comparison hooks against the built-in local fallback
- runtime stats for stage timings and backend correctness deltas
- prototype transport layers ranging from local to queue-based dispatched payloads
- a latency-budget framing for future suffix-only NPU work

## Prototype Backend Ladder

The current `PI05` suffix backend ladder is:

- `local`
- `dummy_local`
- `serialized_local`
- `numpy_local`
- `dispatched_numpy_local`
- `queued_dispatched_numpy_local`
- `delayed_queued_dispatched_numpy_local`
- `npu_stub_local`

These are not eight different policies. They are eight increasingly strict ways of packaging and transporting the same suffix rollout request.

The intent is to move from:

- direct local execution

toward:

- explicit payload serialization
- explicit bytes-oriented dispatch
- queue-based request/response semantics
- configurable artificial transport delay
- future NPU lifecycle stubs

without losing correctness against the local fallback.

## What Was Validated

### 1. Explicit Prefix/Suffix Runtime Boundary

`PI05Policy` now exposes a staged runtime shape rather than only a monolithic inference call.

Key policy-level runtime methods now include:

- `build_prefix_context(...)`
- `rollout_action_chunk(...)`
- `compare_suffix_backend(...)`

This makes suffix-only backend experimentation possible without pretending the whole policy must move together.

### 2. Runtime-Integrated Backend Validation

The runtime path can now validate the active suffix backend against the local fallback while running.

Relevant config/runtime support:

- `policy.suffix_backend=...`
- `suffix_backend_check=true`
- `runtime_stats_output=...`

### 3. Simulator-Backed P3 Prototype Path

The branch includes a simulator-backed `LIBERO` prototype path for backend experiments.

Ready-to-run configs include:

- [examples/inference/libero_rocm_p3_proto.yaml](examples/inference/libero_rocm_p3_proto.yaml)
- [examples/inference/libero_rocm_p3_proto_numpy.yaml](examples/inference/libero_rocm_p3_proto_numpy.yaml)
- [examples/inference/libero_rocm_p3_proto_dispatched.yaml](examples/inference/libero_rocm_p3_proto_dispatched.yaml)
- [examples/inference/libero_rocm_p3_proto_queued.yaml](examples/inference/libero_rocm_p3_proto_queued.yaml)
- [examples/inference/libero_rocm_p3_proto_delayed.yaml](examples/inference/libero_rocm_p3_proto_delayed.yaml)
- [examples/inference/libero_rocm_p3_npu_stub.yaml](examples/inference/libero_rocm_p3_npu_stub.yaml)

### 4. Dispatch Budget Guidance

The current working interpretation from local prototype results is:

- roughly `0-5 ms` added dispatch latency is in the green band
- roughly `5-20 ms` is still prototype-usable but should be measured explicitly
- beyond roughly `20 ms`, suffix-only value starts to erode on the current baseline

This is prototype guidance, not a deployment claim.

## Key Files

- Status summary: [P3_PROTOTYPE_STATUS.md](P3_PROTOTYPE_STATUS.md)
- Dispatch budget: [P3_LATENCY_BUDGET.md](P3_LATENCY_BUDGET.md)
- Entry criteria: [P3_ENTRY_CRITERIA.md](P3_ENTRY_CRITERIA.md)
- Full plan: [PLAN.md](PLAN.md)
- ROCm status: [ROCM_VLASH_STATUS.md](ROCM_VLASH_STATUS.md)
- ROCm env bootstrap: [tools/setup_rocm_env.sh](tools/setup_rocm_env.sh)
- ROCm runtime check: [tools/check_rocm_runtime.sh](tools/check_rocm_runtime.sh)
- ROCm launcher: [tools/run_rocm_vlash.sh](tools/run_rocm_vlash.sh)
- Backend comparison helper: [tools/check_suffix_backend.py](tools/check_suffix_backend.py)
- Backend sweep helper: [tools/sweep_suffix_backends.py](tools/sweep_suffix_backends.py)
- Prototype config baseline: [examples/inference/libero_rocm_p3_proto.yaml](examples/inference/libero_rocm_p3_proto.yaml)

## Fast Start

Validate the ROCm runtime:

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/check_rocm_runtime.sh
```

Check the CLI entry:

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/run_rocm_vlash.sh --help
```

Run the baseline ROCm smoke benchmark:

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/run_rocm_vlash.sh benchmark \
  examples/benchmarks/inference_latency_rocm_smoke.yaml \
  --policy.path=/home/amd/.cache/vlash/models/vlash-pi05-libero-async5
```

Run the P3 prototype runtime config:

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/run_rocm_vlash.sh run \
  examples/inference/libero_rocm_p3_proto.yaml
```

## Reproduce A Backend Sweep

A direct backend sweep entry is:

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm \
  /home/amd/.venvs/vlash-rocm/bin/python \
  tools/sweep_suffix_backends.py \
  examples/inference/libero_rocm_p3_proto.yaml
```

This writes per-backend JSON artifacts plus a summary table under:

```bash
outputs/libero_runtime/p3_backend_sweep
```

## Best Current Interpretation

What this branch has already proved:

- suffix rollout can be isolated as an explicit runtime concept
- the current request contract survives increasingly strict backend boundaries
- correctness can still be checked against the local fallback in the runtime path
- small dispatch costs appear tolerable on the current simulator-backed baseline

What this branch has not proved:

- that a real NPU suffix backend is already implemented
- that current AMD results match paper latency
- that simulator-backed dispatch tolerance automatically transfers to real hardware

## Recommended Next Work

If continuing from this branch, the most useful next steps are:

1. keep the local ROCm baseline stable while preserving the current suffix contract
2. validate the delayed and queued backends under more realistic runtime conditions
3. narrow the first real NPU-oriented milestone to a suffix-only integration spike
4. avoid broad offload claims until transport overhead fits inside the current budget bands
