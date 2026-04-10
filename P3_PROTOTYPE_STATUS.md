# P3 Prototype Status

Last updated: 2026-04-11

## Scope

This document summarizes what the current `P3-prototype` has already proved on the `pi05-rocm-vlash` branch.

It does not claim that formal `P3` has started. It only records the current prototype state.

## Current Position

The branch now has:

- a confirmed `P2.5` simulator-backed no-NPU baseline
- an explicit suffix backend interface
- multiple prototype backend selections
- a correctness comparison path
- a runtime-integrated backend validation mode

## Confirmed P2.5 Baseline

The simulator-backed no-NPU recommendation is currently:

```yaml
policy:
  n_action_steps: 8
  suffix_backend: local

action_quant_ratio: 2
inference_overlap_steps: 1
```

This recommendation was established through:

- initial `LIBERO` sweep
- longer confirmation sweep

Key interpretation:

- `action_quant_ratio=2` is still the strongest loop-cadence lever
- async overlap mainly changes chunk scheduling
- overlap does not materially reduce single staged inference compute time
- `q2/o1` remains preferable to `q2/o2`

## Prototype Backend Layers

The current prototype supports six suffix backend modes:

- `local`
- `dummy_local`
- `serialized_local`
- `numpy_local`
- `dispatched_numpy_local`
- `queued_dispatched_numpy_local`

### `local`

The current default behavior-preserving suffix backend.

Purpose:

- baseline execution path
- fallback reference for all prototype comparisons

### `dummy_local`

The first non-default backend path.

Purpose:

- prove that suffix backend selection works through config and runtime plumbing
- prove that a non-default backend can still match local fallback exactly

Validated:

- config override works
- backend name switches correctly
- comparison result is exact (`max_abs_diff = 0.0`)

### `serialized_local`

The first prototype backend with non-local execution semantics.

Purpose:

- simulate a future heterogeneous backend boundary
- explicitly serialize suffix request data
- move payload through CPU tensors
- reconstruct runtime objects on the original device
- then call the local rollout path

This is the most important current prototype result because it validates:

- request packaging
- payload contract
- device restoration
- correctness against the local fallback

Validated:

- config override works
- runtime can execute with `serialized_local`
- comparison result is exact (`max_abs_diff = 0.0`)

### `numpy_local`

The strictest current prototype backend.

Purpose:

- serialize the suffix request through CPU `numpy` arrays
- preserve dtype and device metadata explicitly
- reconstruct the request on the original device
- exercise a boundary closer to real cross-process or cross-runtime transport

This extends `serialized_local` by validating that the contract survives a
less Torch-native transport format.

Validated:

- config override works
- backend name switches correctly
- comparison result is exact (`max_abs_diff = 0.0`)

### `dispatched_numpy_local`

The newest prototype backend adds an explicit dispatch hop on top of `numpy_local`.

Purpose:

- serialize the suffix request into the existing numpy payload contract
- wrap that payload into a bytes envelope
- force an explicit dispatch/receive boundary before reconstruction
- validate that the current suffix contract survives a more standalone execution shape

This is still local in where compute happens, but it is less local in how the
request is transported and restored.

Validated:

- config override works
- backend name switches correctly
- comparison result remains exact (`max_abs_diff = 0.0`)

### `queued_dispatched_numpy_local`

This backend pushes the dispatched prototype one step further by routing the
bytes envelope through an explicit request/response queue and worker thread.

Purpose:

- simulate a lightweight suffix service boundary instead of a direct method hop
- validate that the current payload contract survives queued dispatch semantics
- keep correctness checks inside the same runtime/reporting path used by the other backends

Validated:

- config override works
- backend name switches correctly
- comparison result remains exact (`max_abs_diff = 0.0`)

## Runtime Validation Mode

The prototype no longer depends only on offline checks.

Current runtime support:

- `suffix_backend_check=true` in [run_config.py](/home/amd/vlash/vlash/configs/run_config.py)
- offline comparison helper: [check_suffix_backend.py](/home/amd/vlash/tools/check_suffix_backend.py)
- multi-backend sweep helper: [sweep_suffix_backends.py](/home/amd/vlash/tools/sweep_suffix_backends.py)
- runtime stats now include:
  - backend name
  - latest `max_abs_diff`
  - latest `mean_abs_diff`
- ready-to-run prototype config:
  - [libero_rocm_p3_proto.yaml](/home/amd/vlash/examples/inference/libero_rocm_p3_proto.yaml)
  - [libero_rocm_p3_proto_numpy.yaml](/home/amd/vlash/examples/inference/libero_rocm_p3_proto_numpy.yaml)
  - [libero_rocm_p3_proto_dispatched.yaml](/home/amd/vlash/examples/inference/libero_rocm_p3_proto_dispatched.yaml)
  - [libero_rocm_p3_proto_queued.yaml](/home/amd/vlash/examples/inference/libero_rocm_p3_proto_queued.yaml)

This means `vlash run` can now validate prototype backend correctness during staged inference launches.

Representative runtime result:

```json
"suffix_backend": {
  "name": "serialized_local",
  "max_abs_diff": 0.0,
  "mean_abs_diff": 0.0
}
```

This has now been validated in two ways:

- offline prototype comparison via [check_suffix_backend.py](/home/amd/vlash/tools/check_suffix_backend.py)
- runtime-integrated validation via `suffix_backend_check=true`

Minimal runtime entry point:

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm \
LIBERO_CONFIG_PATH=/home/amd/.libero \
tools/run_rocm_vlash.sh run examples/inference/libero_rocm_p3_proto.yaml
```

Stricter numpy-payload runtime entry point:

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm \
LIBERO_CONFIG_PATH=/home/amd/.libero \
tools/run_rocm_vlash.sh run examples/inference/libero_rocm_p3_proto_numpy.yaml
```

Dispatched-bytes runtime entry point:

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm \
LIBERO_CONFIG_PATH=/home/amd/.libero \
tools/run_rocm_vlash.sh run examples/inference/libero_rocm_p3_proto_dispatched.yaml
```

Queued-dispatch runtime entry point:

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm \
LIBERO_CONFIG_PATH=/home/amd/.libero \
tools/run_rocm_vlash.sh run examples/inference/libero_rocm_p3_proto_queued.yaml
```

Prototype backend sweep entry point:

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm \
LIBERO_CONFIG_PATH=/home/amd/.libero \
python tools/sweep_suffix_backends.py \
  examples/inference/libero_rocm_p3_proto_dispatched.yaml
```

Latest full sweep summary under one shared simulator config:

| backend | max_abs_diff | mean_abs_diff | loop_avg_ms | stage_total_ms | suffix_mean_ms |
|---|---:|---:|---:|---:|---:|
| `local` | `0.0` | `0.0` | `1044.75` | `1887.77` | `858.68` |
| `serialized_local` | `0.0` | `0.0` | `1033.08` | `1864.49` | `846.23` |
| `numpy_local` | `0.0` | `0.0` | `1024.68` | `1848.43` | `844.20` |
| `dispatched_numpy_local` | `0.0` | `0.0` | `1035.09` | `1868.56` | `848.03` |

The corresponding output bundle is at:

- [summary.md](/home/amd/vlash/outputs/libero_runtime/p3_backend_sweep_full/summary.md)

## What The Prototype Has Proved

The current prototype has proved:

- the suffix boundary is explicit enough to support backend selection
- a non-default suffix backend can be selected through config
- correctness can be compared against the local fallback backend
- runtime can surface backend correctness in structured stats
- an explicit serialized payload path can preserve correctness in simulator-backed runtime
- a stricter `numpy` payload path can also preserve correctness
- a dispatched bytes-envelope path can also preserve correctness
- a queued request/response dispatch path can also preserve correctness
- multiple prototype backends can now be swept under one simulator config and compared with a shared summary format
- the current backend ladder still shows negligible runtime spread under the same simulator config, which supports using correctness-preserving contract strictness as the main prototype axis for now

## What The Prototype Has Not Proved

The current prototype has not proved:

- real acceleration benefit from suffix-only offload
- correctness under real robot hardware conditions
- that the current payload contract is final
- that the current split is sufficient for a real NPU backend without further changes

## Practical Interpretation

The prototype is now strong enough to justify further backend-interface work.

It is not yet strong enough to justify:

- deployment claims
- NPU-specific optimization claims
- switching the branch's main focus fully away from `P2.5`

## Recommended Next Step

The next sensible step is to either:

1. document this prototype state as the current branch checkpoint, or
2. build one more backend variant that exercises a stricter contract than `dispatched_numpy_local`

The first option is lower risk.
The second option is higher value if the branch wants to continue deeper into prototype backend design.
