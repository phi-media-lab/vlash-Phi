<!-- markdownlint-disable MD001 MD041 -->

# pi05-rocm-vlash

中文说明见：[README.zh-CN.md](README.zh-CN.md)

This branch is a local AMD ROCm bring-up of VLASH focused on `pi05` inference.

It is not a generic upstream README. It only documents the parts that matter for this branch:

- running VLASH on AMD ROCm
- loading the public `pi05` checkpoint
- reproducing the local benchmark path
- understanding the current performance limits on this machine

## Branch Goal

The goal of this branch is to make `PI05Policy` inference run reproducibly on:

- AMD Radeon 890M
- ROCm 7.2.1
- PyTorch 2.9.1 ROCm wheels

The current validated path is benchmark-oriented, not paper-level deployment reproduction.

## Current Machine Assumption

This branch was validated on:

- CPU: AMD Ryzen AI 9 HX PRO 370
- GPU: AMD Radeon 890M
- ROCm arch: `gfx1150`
- ROCm runtime: `/opt/rocm-7.2.1`

## What Works

- ROCm-backed torch can use the AMD GPU
- `PI05Policy.from_pretrained(...)` runs on GPU
- `vlash benchmark` runs end-to-end through ROCm
- `compile_model=true` works and improves latency
- camera/image feature remapping is supported in runtime and benchmark paths

## What Does Not Hold

- This branch does not reach the paper's latency numbers
- `fuse_qkv` and `fuse_gate_up` do not improve latency on this AMD setup
- The smoke benchmark is a compatibility benchmark, not a task-faithful evaluation

## Key Files

- Status summary: [ROCM_VLASH_STATUS.md](ROCM_VLASH_STATUS.md)
- ROCm env bootstrap: [tools/setup_rocm_env.sh](tools/setup_rocm_env.sh)
- ROCm runtime check: [tools/check_rocm_runtime.sh](tools/check_rocm_runtime.sh)
- ROCm launcher: [tools/run_rocm_vlash.sh](tools/run_rocm_vlash.sh)
- Checkpoint compatibility check: [tools/check_checkpoint_compat.py](tools/check_checkpoint_compat.py)
- Smoke benchmark config: [examples/benchmarks/inference_latency_rocm_smoke.yaml](examples/benchmarks/inference_latency_rocm_smoke.yaml)

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

The branch was validated with:

- `mit-han-lab/vlash-pi05-libero-async5`

Expected local path:

```bash
/home/amd/.cache/vlash/models/vlash-pi05-libero-async5
```

## Reproduce The AMD ROCm Smoke Benchmark

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
  fuse_qkv: false
  fuse_gate_up: false
```

These settings are based on measured results on this AMD ROCm machine, not on the original upstream defaults.

## Best Known Results

For `pi05` on this machine:

- compiled steady-state latency is roughly `690-710 ms`
- a representative `num_samples=5` run gave:
  - mean `711.18 ms`
  - median `694.79 ms`
  - throughput `1.41 FPS`

Fusion matrix on this hardware:

- `fuse_qkv=false`, `fuse_gate_up=false` is effectively optimal
- enabling only one of the two fusions is slower
- enabling both together is about tied with disabling both

## Notes

- `action_quant_ratio` is a control-loop knob in `run.py`; it does not reduce raw `predict_action_chunk()` latency
- `ROCM_EXPERIMENTAL_ATTENTION=1` did not produce a meaningful speedup in local testing
- the benchmark adapts `lerobot/pusht` image/state features to the public LIBERO checkpoint so that the full CLI path can be exercised

## Recommended Next Work

If continuing from this branch, the most useful next steps are:

1. Evaluate a smaller checkpoint or policy on AMD ROCm
2. Validate the real `vlash run` runtime path on AMD GPU
3. Measure `action_quant_ratio` in a real robot control-loop setting
