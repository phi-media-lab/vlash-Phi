<!-- markdownlint-disable MD001 MD041 -->

# pi05-rocm-vlash

English version: [README.md](README.md)

这个分支是一个面向 AMD ROCm 的 VLASH 本地 bring-up 分支，重点只放在 `pi05` 推理。

它不是上游仓库的通用说明文档，只保留这个分支真正相关的信息：

- 如何在 AMD ROCm 上运行 VLASH
- 如何加载公开的 `pi05` checkpoint
- 如何复现当前本地 benchmark 路径
- 当前这台机器上的性能上限和已知限制

## 分支目标

这个分支的目标，是让 `PI05Policy` 能够在下面这套环境中稳定、可复现地运行：

- AMD Radeon 890M
- ROCm 7.2.1
- PyTorch 2.9.1 ROCm wheels

当前已经打通的是 benchmark / 推理验证路径，不是论文级部署复现。

## 当前机器假设

这个分支是在下面这台机器上验证的：

- CPU: AMD Ryzen AI 9 HX PRO 370
- GPU: AMD Radeon 890M
- ROCm arch: `gfx1150`
- ROCm runtime: `/opt/rocm-7.2.1`

## 当前已验证可用

- ROCm 版 torch 能识别并使用 AMD GPU
- `PI05Policy.from_pretrained(...)` 能在 GPU 上运行
- `vlash benchmark` 能在 AMD GPU 上走完整条 CLI 路径
- `compile_model=true` 可用，并且能明显降低延迟
- runtime 和 benchmark 都支持 camera / image feature remap

## 当前不成立的事

- 这个分支还没有达到论文中的延迟水平
- `fuse_qkv` 和 `fuse_gate_up` 在当前 AMD ROCm 环境下没有带来收益
- 当前 smoke benchmark 是兼容性验证，不是任务语义严格对齐的评测

## 关键文件

- 状态总览：[ROCM_VLASH_STATUS.md](ROCM_VLASH_STATUS.md)
- ROCm 环境脚本：[tools/setup_rocm_env.sh](tools/setup_rocm_env.sh)
- ROCm 运行时检查：[tools/check_rocm_runtime.sh](tools/check_rocm_runtime.sh)
- ROCm 启动入口：[tools/run_rocm_vlash.sh](tools/run_rocm_vlash.sh)
- checkpoint 兼容性检查：[tools/check_checkpoint_compat.py](tools/check_checkpoint_compat.py)
- smoke benchmark 配置：[examples/benchmarks/inference_latency_rocm_smoke.yaml](examples/benchmarks/inference_latency_rocm_smoke.yaml)

## 快速开始

创建 ROCm 环境：

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/setup_rocm_env.sh
```

检查运行时：

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/check_rocm_runtime.sh
```

确认 CLI 入口可用：

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/run_rocm_vlash.sh --help
```

## 公开 Checkpoint

这个分支当前验证使用的是：

- `mit-han-lab/vlash-pi05-libero-async5`

本地默认路径：

```bash
/home/amd/.cache/vlash/models/vlash-pi05-libero-async5
```

## 复现 AMD ROCm Smoke Benchmark

非编译路径：

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/run_rocm_vlash.sh benchmark \
  examples/benchmarks/inference_latency_rocm_smoke.yaml \
  --policy.path=/home/amd/.cache/vlash/models/vlash-pi05-libero-async5
```

编译路径：

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/run_rocm_vlash.sh benchmark \
  examples/benchmarks/inference_latency_rocm_smoke.yaml \
  --policy.path=/home/amd/.cache/vlash/models/vlash-pi05-libero-async5 \
  --policy.compile_model=true \
  --warmup_steps=1 \
  --num_samples=5
```

## 当前机器上的最优已知设置

建议使用：

```yaml
policy:
  device: cuda
  dtype: bfloat16
  compile_model: true
  fuse_qkv: false
  fuse_gate_up: false
```

这些不是沿用上游默认值，而是基于这台 AMD ROCm 机器的实测结果得到的。

## 当前最优已知结果

在这台机器上跑 `pi05`：

- 编译后稳态延迟大约在 `690-710 ms`
- 一个代表性的 `num_samples=5` 结果是：
  - mean `711.18 ms`
  - median `694.79 ms`
  - throughput `1.41 FPS`

关于 fuse 的结论：

- `fuse_qkv=false`, `fuse_gate_up=false` 基本就是最优组合
- 单独打开其中任何一个都会变慢
- 两个同时打开和两个都关闭基本打平

## 备注

- `action_quant_ratio` 是 `run.py` 里的控制循环参数，不会降低原始 `predict_action_chunk()` 延迟
- `ROCM_EXPERIMENTAL_ATTENTION=1` 在本地测试里没有带来可见收益
- 当前 benchmark 会把 `lerobot/pusht` 的 image/state 特征适配到公开 LIBERO checkpoint 上，目的是把完整 CLI 推理链路跑通

## 建议的下一步

如果继续沿这个分支往前做，最值得做的是：

1. 测更小的 checkpoint / policy 在 AMD ROCm 上的延迟
2. 验证真实 `vlash run` 路径是否也能在 AMD GPU 上稳定工作
3. 在真实机器人控制循环里评估 `action_quant_ratio`
