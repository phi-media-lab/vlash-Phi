<!-- markdownlint-disable MD001 MD041 -->

# vlash-pi05-gfx1150-baseline

中文说明见：[README.zh-CN.md](README.zh-CN.md)

This branch is the baseline branch for running `VLASH` `PI05` inference on the local AMD `gfx1150` machine.

It is branch-specific by design. It is not the generic upstream README, and it is not the suffix-backend prototype branch. The purpose of this branch is to preserve a stable, reproducible ROCm baseline for the `Radeon 890M / gfx1150` machine before any extra runtime experiments are layered on top.

## Branch Goal

The goal of this branch is to preserve the smallest practical baseline that answers these questions clearly:

- can `PI05Policy` load and run on the local AMD GPU through ROCm
- what environment and scripts are required to reproduce that result
- what are the current best-known settings and observed limits on this machine

This is a baseline branch, not a deployment claim and not a future NPU branch.

## Machine Target

This branch is specifically for the local `gfx1150` machine:

- CPU: `AMD Ryzen AI 9 HX PRO 370`
- GPU: `AMD Radeon 890M`
- ROCm arch: `gfx1150`
- ROCm runtime: `/opt/rocm-7.2.1`

If future `MI300X` work continues, it should live on a separate branch rather than be merged into this baseline.

## What This Branch Preserves

This branch is meant to preserve the AMD ROCm baseline for:

- ROCm environment creation
- runtime library setup
- public checkpoint loading
- CLI-level benchmark reproducibility
- best-known inference settings on this machine
- basic `vlash run` compatibility on mock and simulator-backed paths

## What Works

Current validated baseline properties:

- ROCm-backed torch can detect and use the AMD GPU
- `PI05Policy.from_pretrained(...)` runs on GPU
- `vlash benchmark` runs end to end through ROCm
- `compile_model=true` works and improves latency on this machine
- split staged inference (`build_prefix_context -> rollout_action_chunk`) is stable when staged helpers avoid cudagraphs
- `vlash run` works on local mock runtime and `LIBERO` simulator paths
- camera/image feature remapping works in runtime and benchmark paths

## What Does Not Hold

This branch does not claim that:

- current latency matches the paper
- current AMD ROCm results are deployment-grade
- existing `fuse_qkv` / `fuse_gate_up` settings improve latency here
- this baseline should also stand in for future `MI300X` results

## Key Files

- Status summary: [ROCM_VLASH_STATUS.md](ROCM_VLASH_STATUS.md)
- Implementation plan: [PLAN.md](PLAN.md)
- ROCm env bootstrap: [tools/setup_rocm_env.sh](tools/setup_rocm_env.sh)
- ROCm runtime check: [tools/check_rocm_runtime.sh](tools/check_rocm_runtime.sh)
- ROCm launcher: [tools/run_rocm_vlash.sh](tools/run_rocm_vlash.sh)
- Checkpoint compatibility check: [tools/check_checkpoint_compat.py](tools/check_checkpoint_compat.py)
- Smoke benchmark config: [examples/benchmarks/inference_latency_rocm_smoke.yaml](examples/benchmarks/inference_latency_rocm_smoke.yaml)
- Simulator runtime config: [examples/inference/libero_rocm.yaml](examples/inference/libero_rocm.yaml)

## Fast Start

Create the ROCm environment:

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/setup_rocm_env.sh
```

Validate the runtime:

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/check_rocm_runtime.sh
```

Check the CLI entry:

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/run_rocm_vlash.sh --help
```

## Public Checkpoint

The baseline was validated with:

- `mit-han-lab/vlash-pi05-libero-async5`

Expected local path:

```bash
/home/amd/.cache/vlash/models/vlash-pi05-libero-async5
```

## Reproduce The ROCm Smoke Benchmark

Non-compiled:

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/run_rocm_vlash.sh benchmark \
  examples/benchmarks/inference_latency_rocm_smoke.yaml \
  --policy.path=/home/amd/.cache/vlash/models/vlash-pi05-libero-async5
```

Compiled:

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/run_rocm_vlash.sh benchmark \
  examples/benchmarks/inference_latency_rocm_smoke.yaml \
  --policy.path=/home/amd/.cache/vlash/models/vlash-pi05-libero-async5 \
  --policy.compile_model=true \
  --warmup_steps=1 \
  --num_samples=5
```

## Best Known Settings On This Machine

Use:

```yaml
policy:
  device: cuda
  dtype: bfloat16
  compile_model: true
  compile_mode: max-autotune
  staged_compile_mode: max-autotune-no-cudagraphs
  fuse_qkv: false
  fuse_gate_up: false
```

These settings are based on local measurements on this `gfx1150` machine, not on original upstream defaults.

## Best Known Results

For `pi05` on this machine:

- compiled steady-state latency is roughly `690-710 ms`
- a representative `num_samples=5` run gave:
  - mean `711.18 ms`
  - median `694.79 ms`
  - throughput `1.41 FPS`

Current fusion conclusion on this hardware:

- `fuse_qkv=false`, `fuse_gate_up=false` is effectively optimal
- enabling only one of the two fusions is slower
- enabling both together is about tied with disabling both

## Shared `pi05_base` Bridge Checkpoint

This branch also carries a bridge path for loading an `openpi pi05_base`-derived PyTorch checkpoint into `VLASH`:

- shared checkpoint: `/home/amd/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch`
- source of weights: local `openpi` `pi05_base` JAX checkpoint converted to `model.safetensors`
- current bridge status: loads successfully, produces finite outputs on GPU, and is suitable for model-level comparison work

Current model-level measurements on this `gfx1150` machine use a fixed shared input and zero-pad the original `14`-D state into the `32`-D `PI05` contract expected by the shared checkpoint. These are not CLI `LIBERO` runtime numbers; they are direct `PI05Policy` measurements.

Shared-checkpoint results observed so far:

- eager short run (`warmup=1`, `runs=5`):
  - mean `8.26 s`
  - median `8.27 s`
  - throughput `0.121 Hz`
- compiled short run (`compile_model=true`, `warmup_after_compile=1`, `runs=3`):
  - compile warmup `8.09 s`
  - steady-state mean `7.77 s`
  - throughput `0.129 Hz`

Compiled behavior on this machine is now better understood:

- full `policy.predict_action_chunk(...)` works with `compile_mode=max-autotune`
- split staged inference (`build_prefix_context -> rollout_action_chunk`) is not stable with default cudagraph-backed `max-autotune`
- using `staged_compile_mode=max-autotune-no-cudagraphs` restores staged correctness without changing the full-path compile setting

Measured compiled warmup tiers for the shared bridge path:

- fresh-cache full compiled first call: about `420 s`
- warm-cache new-process first call: about `24-25 s`
- same-process steady-state full call: about `4.4-4.8 s`

Shared-checkpoint stage breakdown is dominated by prefix compute:

- eager prefix `7.38 s`, suffix `0.89 s`
- compiled prefix `6.93 s`, suffix `0.84 s`
- prefix remains about `89%` of end-to-end latency

Current interpretation:

- the `openpi pi05_base -> vlash` bridge now works well enough for finite-value GPU inference
- `compile_model=true` helps, but only modestly on this machine, roughly `6%` in the current short benchmark
- compiled warmup cost is dominated by inductor/triton compile+autotune, not by cudagraphs alone
- this path is still much slower than the public `vlash-pi05-libero-async5` baseline and should be treated as a checkpoint-compatibility bridge, not the default serving path

## Notes

- `action_quant_ratio` is a control-loop knob in `run.py`; it does not reduce raw `predict_action_chunk()` latency
- `ROCM_EXPERIMENTAL_ATTENTION=1` did not produce a meaningful speedup in local testing
- for short-lived processes on `gfx1150`, eager remains the safer default baseline; compiled is more appropriate for warm long-lived services
- the smoke benchmark remaps `lerobot/pusht` image/state features to the public LIBERO checkpoint so that the full CLI path can be exercised

## Recommended Next Work

If continuing from this baseline branch, the most useful next steps are:

1. keep the ROCm baseline reproducible and stable
2. validate the real `vlash run` path on real hardware
3. measure control-loop parameters such as `action_quant_ratio` under more realistic runtime conditions
4. fork experimental runtime work into separate branches rather than mixing it into this baseline
