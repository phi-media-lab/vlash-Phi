<!-- markdownlint-disable MD001 MD041 -->

# vlash-pi05-gfx1150-baseline

English version: [README.md](README.md)

这个分支是本地 AMD `gfx1150` 机器上运行 `VLASH PI05` 的基线分支。

它是一个明确的 branch-specific README。它不是上游仓库的通用说明，也不是 suffix-backend prototype 分支。这个分支的目标，是在任何额外 runtime 实验叠加之前，先保留一条稳定、可复现的 `Radeon 890M / gfx1150` ROCm 基线。

## 分支目标

这个分支要清楚回答的是：

- `PI05Policy` 能不能通过 ROCm 在本地 AMD GPU 上加载并运行
- 复现这个结果需要哪些环境和脚本
- 这台机器上当前最优已知设置和性能边界是什么

这是一条 baseline 分支，不是部署结论，也不是未来 NPU 分支。

## 目标机器

这个分支专门对应本地 `gfx1150` 机器：

- CPU: `AMD Ryzen AI 9 HX PRO 370`
- GPU: `AMD Radeon 890M`
- ROCm arch: `gfx1150`
- ROCm runtime: `/opt/rocm-7.2.1`

如果后面继续做 `MI300X` 相关工作，应当放在单独分支上，而不是继续混进这条 baseline 分支。

## 这个分支要保留什么

这条分支要保留的是 AMD ROCm 基线本身，包括：

- ROCm 环境创建
- runtime library 设置
- 公开 checkpoint 加载
- CLI benchmark 的可复现路径
- 当前机器上的 best-known inference settings
- `vlash run` 在 mock 和 simulator-backed 路径上的基础兼容性

## 当前已验证可用

当前已经验证成立的基线能力：

- ROCm 版 torch 能检测并使用 AMD GPU
- `PI05Policy.from_pretrained(...)` 能在 GPU 上运行
- `vlash benchmark` 能通过 ROCm 端到端跑通
- `compile_model=true` 可用，并且能改善这台机器上的延迟
- 分阶段推理（`build_prefix_context -> rollout_action_chunk`）在 staged helper 避开 cudagraphs 时是稳定的
- `vlash run` 能在本地 mock runtime 和 `LIBERO` simulator 路径上工作
- runtime 和 benchmark 都支持 camera/image feature remap

## 当前不成立的事

这个分支不声称：

- 当前延迟已经达到论文数字
- 当前 AMD ROCm 结果已经达到部署级别
- 现有 `fuse_qkv` / `fuse_gate_up` 在这里有收益
- 这条 baseline 能直接替代未来的 `MI300X` 结果

## 关键文件

- 状态总览：[ROCM_VLASH_STATUS.md](ROCM_VLASH_STATUS.md)
- 实施计划：[PLAN.md](PLAN.md)
- 完整系统对比计划：[LIBERO_SYSTEM_COMPARISON_PLAN.md](LIBERO_SYSTEM_COMPARISON_PLAN.md)
- ROCm 环境脚本：[tools/setup_rocm_env.sh](tools/setup_rocm_env.sh)
- ROCm 运行时检查：[tools/check_rocm_runtime.sh](tools/check_rocm_runtime.sh)
- ROCm 启动入口：[tools/run_rocm_vlash.sh](tools/run_rocm_vlash.sh)
- checkpoint 兼容性检查：[tools/check_checkpoint_compat.py](tools/check_checkpoint_compat.py)
- smoke benchmark 配置：[examples/benchmarks/inference_latency_rocm_smoke.yaml](examples/benchmarks/inference_latency_rocm_smoke.yaml)
- simulator runtime 配置：[examples/inference/libero_rocm.yaml](examples/inference/libero_rocm.yaml)

## 快速开始

创建 ROCm 环境：

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/setup_rocm_env.sh
```

检查运行时：

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/check_rocm_runtime.sh
```

确认 CLI 入口：

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/run_rocm_vlash.sh --help
```

## 公开 Checkpoint

当前 baseline 验证使用的是：

- `mit-han-lab/vlash-pi05-libero-async5`

本地默认路径：

```bash
/home/amd/.cache/vlash/models/vlash-pi05-libero-async5
```

## 复现 ROCm Smoke Benchmark

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
  compile_mode: max-autotune
  staged_compile_mode: max-autotune-no-cudagraphs
  fuse_qkv: false
  fuse_gate_up: false
```

这些设置来自这台 `gfx1150` 机器上的本地实测，不是沿用上游默认值。

## 当前最优已知结果

在这台机器上跑 `pi05`：

- 编译后稳态延迟大约在 `690-710 ms`
- 一个代表性的 `num_samples=5` 结果是：
  - mean `711.18 ms`
  - median `694.79 ms`
  - throughput `1.41 FPS`

当前这台机器上的 fuse 结论：

- `fuse_qkv=false`, `fuse_gate_up=false` 基本就是最优组合
- 单独打开其中任何一个都会更慢
- 两个同时打开和两个都关闭大致打平

## Shared `pi05_base` Bridge Checkpoint

这条分支同时保留了一条把 `openpi pi05_base` 派生出的 PyTorch checkpoint 接到 `VLASH` 的 bridge 路线：

- shared checkpoint：`/home/amd/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch`
- 权重来源：本地 `openpi` `pi05_base` JAX checkpoint 转成的 `model.safetensors`
- 当前 bridge 状态：已经能稳定加载，GPU 推理输出有限值，可用于模型级对比

这组模型级数据使用固定 shared input，并把原始 `14` 维 state 按当前 bridge 约定零填充到 shared checkpoint 期望的 `32` 维 `PI05` contract。它们不是 CLI `LIBERO` runtime 数字，而是直接测 `PI05Policy` 本体。

目前测到的 shared-checkpoint 结果：

- eager 短测（`warmup=1`, `runs=5`）：
  - mean `8.26 s`
  - median `8.27 s`
  - throughput `0.121 Hz`
- compiled 短测（`compile_model=true`, `warmup_after_compile=1`, `runs=3`）：
  - 首次 compile warmup `8.09 s`
  - 稳态 mean `7.77 s`
  - throughput `0.129 Hz`

这台机器上的 compiled 行为现在已经比较清楚：

- 完整 `policy.predict_action_chunk(...)` 路径可以继续使用 `compile_mode=max-autotune`
- 分阶段推理（`build_prefix_context -> rollout_action_chunk`）在默认带 cudagraph 的 `max-autotune` 下不稳定
- 把 staged helper 改成 `staged_compile_mode=max-autotune-no-cudagraphs` 后，分阶段路径恢复正确性，同时不影响 full path 的 compile 设置

shared bridge 这条路径目前测到的 compiled warmup 分三档：

- fresh-cache 首次 full compiled 调用：约 `420 s`
- warm-cache 新进程首调：约 `24-25 s`
- 同进程 steady-state full 调用：约 `4.4-4.8 s`

shared-checkpoint 的 stage breakdown 明显还是 prefix 主导：

- eager prefix `7.38 s`，suffix `0.89 s`
- compiled prefix `6.93 s`，suffix `0.84 s`
- prefix 仍然约占端到端延迟的 `89%`

当前解释应该是：

- `openpi pi05_base -> vlash` bridge 已经足够支撑有限值 GPU 推理
- `compile_model=true` 有帮助，但提升不大，这轮短测大约是 `6%`
- compiled 的 warmup 代价主要来自 inductor/triton 的 compile 和 autotune，不只是 cudagraphs 本身
- 这条路径目前仍明显慢于公开的 `vlash-pi05-libero-async5` baseline，更适合当 checkpoint compatibility bridge，而不是默认 serving 路径

## 备注

- `action_quant_ratio` 是 `run.py` 里的控制循环参数，不会降低原始 `predict_action_chunk()` 延迟
- `ROCM_EXPERIMENTAL_ATTENTION=1` 在本地测试里没有带来明显收益
- 对 `gfx1150` 这类短命进程场景，eager 仍然是更稳的默认 baseline；compiled 更适合预热后的长驻服务
- smoke benchmark 会把 `lerobot/pusht` 的 image/state 特征映射到公开 LIBERO checkpoint 上，以便完整跑通 CLI 路径

## 当前 `LIBERO` 系统级对比状态

这条分支现在已经包含一个统一的 `LIBERO` 对比 harness：

- [tools/libero_system_compare.py](tools/libero_system_compare.py)

它会把：

- `openpi` 的 `pi05_libero_pytorch`
- `vlash` 的 `mit-han-lab/vlash-pi05-libero-async5`

放进同一个 `LIBERO` driver、同一组 task id / seed 和同一套 JSON 输出结构里。

当前在 `gfx1150` 上已经验证的系统级里程碑：

- 两个系统都能完成 `2-step` smoke，不崩
- 两个系统都能完成同 task/seed 的 `20-step` runtime smoke
- 两个系统都能完成 tasks `{0,1}`、seeds `{0,1}` 的 `50-step` sweep

这台机器上 `50-step` 聚合结果目前是：

- `openpi`
  - load `29.60 s`
  - 首动作延迟 `1527.90 ms`
  - 平均单步延迟 `143.40 ms`
  - 平均 episode 时长 `8.72 s`
- `vlash`
  - load `33.92 s`
  - 首动作延迟 `1172.94 ms`
  - 平均单步延迟 `44.73 ms`
  - 平均 episode 时长 `3.75 s`

当前解释：

- 以这轮 `gfx1150` 的 task-level 结果看，`vlash` 比 `openpi` 更快
- 但这些结果仍然主要是 runtime 对比，不是最终 task-success 结论
- harness 里当前对 `openpi` 显式关闭了 PyTorch compile，避免把首轮 compile 成本混进 smoke/runtime bring-up 结果

## MI300X 迁移就绪度

基于当前证据，下一阶段已经适合迁移到 `MI300X`。

在 `gfx1150` 上已经收掉的问题：

- shared `openpi pi05_base -> vlash` bridge 在模型核心层面已经成立
- 两个完整系统都能在统一 `LIBERO` harness 下运行
- task-level smoke 和短 sweep 已经不再卡在 bring-up blocker

而 `gfx1150` 已经不适合继续承担的是：

- 最终性能结论
- 更长 horizon 的 task-success 结论
- 面向部署级 GPU 的系统判断

当前建议：

- 继续保留这条分支作为 `gfx1150` baseline 和本地 harness 参考
- 下一阶段迁移到 `MI300X`
- 在 `MI300X` 上并行做两条：
  - shared `pi05_base` 的核心模型对比
  - `LIBERO` 任务级系统对比

## 建议的下一步

如果继续沿这条 baseline 分支往前做，最值得做的是：

1. 保持 ROCm baseline 稳定且可复现
2. 在真实硬件上验证 `vlash run` 主路径
3. 在更接近真实运行时的条件下评估 `action_quant_ratio` 这类控制参数
4. 把实验性 runtime 工作放到单独分支，而不是继续混进这条 baseline 分支
