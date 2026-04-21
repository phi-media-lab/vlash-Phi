<!-- markdownlint-disable MD001 MD041 -->

# suffix-dispatch-budget

English version: [README.md](README.md)

这个分支是一个专门面向 `suffix backend` 分离实验的 VLASH 原型分支。

它不是上游仓库的通用 README，也不再只是一个 AMD ROCm bring-up 分支。这个分支的重点，是在保持 AMD ROCm 基线可运行的前提下，把 `PI05` 的 suffix rollout 变成一个显式运行时边界，并围绕这个边界做：比较、dispatch、排队、延迟注入，以及未来异构后端替换前的预算评估。

## 分支目标

这个分支要回答的是一个很具体的问题：

- `PI05` 的 suffix 路径到底能被拆分到什么程度
- 在保持正确性的前提下
- 在额外 dispatch 开销开始侵蚀系统价值之前

落到工程上，这个分支主要做三件事：

1. 把 suffix rollout 边界显式化
2. 验证一组越来越严格的 suffix backend 原型
3. 为未来面向 NPU 的 suffix-only 工作估一个实际可用的 dispatch latency budget

## 这个分支是干什么的

这条分支适合放这些内容：

- suffix backend 接口改动
- dispatch / queue / delayed transport 原型
- 运行时内联的 suffix backend 正确性检查
- latency budget 和 P3 prototype 设计工作

它不应该再充当纯 ROCm bring-up 首页。ROCm bring-up 现在是这个分支的前置条件，不是主叙事。

## 当前基线假设

当前这些原型工作依赖一个稳定的 AMD ROCm 本地基线。

当前已验证机器基线：

- CPU: `AMD Ryzen AI 9 HX PRO 370`
- GPU: `AMD Radeon 890M`
- ROCm arch: `gfx1150`
- ROCm runtime: `/opt/rocm-7.2.1`
- PyTorch: `2.9.1+rocm7.2.1`

当前 prototype 路径使用的公开 checkpoint：

- `mit-han-lab/vlash-pi05-libero-async5`

本地默认路径：

```bash
/home/amd/.cache/vlash/models/vlash-pi05-libero-async5
```

## 这个分支新增了什么

相对一个“只要能在 ROCm 上推理”的基线，这个分支额外引入了：

- `PI05Policy` 中显式的 `prefix -> suffix` staged runtime
- 通过 config 选择 suffix backend 的能力
- 针对内建 local fallback 的 backend 对比检查
- stage timings 和 backend correctness delta 的 runtime stats
- 从 local 到 queue-based dispatched payload 的一组原型 transport 形态
- 一个面向未来 suffix-only NPU 工作的 latency budget 视角

## 当前的 Prototype Backend 梯度

当前 `PI05` 支持的 suffix backend 梯度是：

- `local`
- `dummy_local`
- `serialized_local`
- `numpy_local`
- `dispatched_numpy_local`
- `queued_dispatched_numpy_local`
- `delayed_queued_dispatched_numpy_local`
- `npu_stub_local`

它们不是八个不同模型，而是同一个 suffix rollout request 在越来越严格边界语义下的八种包装/运输方式。

它的演进方向是从：

- 直接本地执行

逐步走向：

- 显式 payload 序列化
- 显式 bytes-oriented dispatch
- queue-based request/response 语义
- 可配置的人工 transport delay
- 未来 NPU 生命周期 stub

同时仍然要求和 local fallback 保持正确性一致。

## 已验证的内容

### 1. Prefix / Suffix 显式运行时边界

`PI05Policy` 现在不只是一个整体推理调用，它已经暴露出 staged runtime 结构。

关键接口包括：

- `build_prefix_context(...)`
- `rollout_action_chunk(...)`
- `compare_suffix_backend(...)`

这让 suffix-only backend 实验成为可能，而不必假设整个 policy 必须整体迁移。

### 2. 运行时内联 Backend 校验

当前 runtime 路径已经可以在运行时把 active suffix backend 和 local fallback 进行比较。

相关配置/运行时支持包括：

- `policy.suffix_backend=...`
- `suffix_backend_check=true`
- `runtime_stats_output=...`

### 3. 基于 Simulator 的 P3 Prototype 路径

当前分支已经提供了一条基于 `LIBERO` simulator 的 backend prototype 路径。

现成配置包括：

- [examples/inference/libero_rocm_p3_proto.yaml](examples/inference/libero_rocm_p3_proto.yaml)
- [examples/inference/libero_rocm_p3_proto_numpy.yaml](examples/inference/libero_rocm_p3_proto_numpy.yaml)
- [examples/inference/libero_rocm_p3_proto_dispatched.yaml](examples/inference/libero_rocm_p3_proto_dispatched.yaml)
- [examples/inference/libero_rocm_p3_proto_queued.yaml](examples/inference/libero_rocm_p3_proto_queued.yaml)
- [examples/inference/libero_rocm_p3_proto_delayed.yaml](examples/inference/libero_rocm_p3_proto_delayed.yaml)
- [examples/inference/libero_rocm_p3_npu_stub.yaml](examples/inference/libero_rocm_p3_npu_stub.yaml)

### 4. Dispatch Budget 指引

根据当前本地 prototype 结果，当前可工作的解释是：

- 额外 `0-5 ms` 的 dispatch latency 基本还在 green band
- 额外 `5-20 ms` 仍然可以做 prototype，但必须显式测量
- 超过大约 `20 ms`，当前 baseline 下 suffix-only 的系统价值会开始被侵蚀

这只是 prototype guidance，不是部署结论。

## 关键文件

- 原型状态总结：[P3_PROTOTYPE_STATUS.md](P3_PROTOTYPE_STATUS.md)
- Dispatch budget：[P3_LATENCY_BUDGET.md](P3_LATENCY_BUDGET.md)
- 进入 P3 的准入条件：[P3_ENTRY_CRITERIA.md](P3_ENTRY_CRITERIA.md)
- 完整计划：[PLAN.md](PLAN.md)
- ROCm 状态：[ROCM_VLASH_STATUS.md](ROCM_VLASH_STATUS.md)
- ROCm 环境脚本：[tools/setup_rocm_env.sh](tools/setup_rocm_env.sh)
- ROCm 运行时检查：[tools/check_rocm_runtime.sh](tools/check_rocm_runtime.sh)
- ROCm 启动入口：[tools/run_rocm_vlash.sh](tools/run_rocm_vlash.sh)
- Backend 对比工具：[tools/check_suffix_backend.py](tools/check_suffix_backend.py)
- Backend sweep 工具：[tools/sweep_suffix_backends.py](tools/sweep_suffix_backends.py)
- Prototype 基线配置：[examples/inference/libero_rocm_p3_proto.yaml](examples/inference/libero_rocm_p3_proto.yaml)

## 快速开始

检查 ROCm runtime：

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/check_rocm_runtime.sh
```

确认 CLI 入口：

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/run_rocm_vlash.sh --help
```

跑基线 ROCm smoke benchmark：

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/run_rocm_vlash.sh benchmark \
  examples/benchmarks/inference_latency_rocm_smoke.yaml \
  --policy.path=/home/amd/.cache/vlash/models/vlash-pi05-libero-async5
```

跑 P3 prototype runtime 配置：

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm tools/run_rocm_vlash.sh run \
  examples/inference/libero_rocm_p3_proto.yaml
```

## 复现 Backend Sweep

直接 sweep 的入口是：

```bash
VENV_DIR=/home/amd/.venvs/vlash-rocm \
  /home/amd/.venvs/vlash-rocm/bin/python \
  tools/sweep_suffix_backends.py \
  examples/inference/libero_rocm_p3_proto.yaml
```

它会把每个 backend 的 JSON 结果和汇总表写到：

```bash
outputs/libero_runtime/p3_backend_sweep
```

## 当前最合理的解释

这条分支现在已经证明了：

- suffix rollout 可以被抽成一个显式 runtime 概念
- 当前 request contract 可以跨越来越严格的 backend 边界仍然成立
- backend 正确性可以在 runtime 路径里继续对照 local fallback 检查
- 在当前 simulator-backed baseline 下，小 dispatch 开销似乎还能接受

这条分支还没有证明：

- 真正的 NPU suffix backend 已经实现
- 当前 AMD 结果已经达到论文 latency
- simulator-backed dispatch 容忍度可以直接外推到真实硬件

## 建议的下一步

如果继续沿这条分支往前做，最值得做的是：

1. 保持本地 ROCm baseline 稳定，同时别破坏当前 suffix contract
2. 在更接近真实运行时的条件下验证 delayed 和 queued backend
3. 把第一个真实 NPU 方向里程碑收窄成 suffix-only integration spike
4. 在 transport overhead 还没落到当前 budget band 以内之前，不要做过宽的 offload 结论
