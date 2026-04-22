#!/usr/bin/env python3
"""Common LIBERO harness for system-level openpi vs vlash comparison.

This script has two modes:
1. Driver mode (default): owns the LIBERO environment, episode loop, metrics, and
   spawns system workers.
2. Worker mode (--worker): loads one policy stack (openpi or vlash) inside its own
   Python environment and serves step-wise actions over a local IPC socket.

The first implementation intentionally prioritizes a single shared harness and a
single JSON schema over the most optimized IPC path.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Sequence
from multiprocessing.connection import Client, Listener
from typing import Any

import numpy as np


DEFAULT_ROCM_LD_LIBRARY_PATH = "/opt/rocm-7.2.1/lib:/opt/rocm/lib"
DEFAULT_OPENPI_PYTHON = "/home/amd/projects/openpi-Phi/.venv/bin/python"
DEFAULT_VLASH_PYTHON = "/home/amd/.venvs/vlash-rocm/bin/python"
DEFAULT_OPENPI_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
DEFAULT_VLASH_CHECKPOINT = "/home/amd/.cache/vlash/models/vlash-pi05-libero-async5"
DEFAULT_OUTPUT_ROOT = "/home/amd/vlash/outputs/system_compare"
CAMERA_NAMES = ["agentview_image", "robot0_eye_in_hand_image"]
CAMERA_NAME_MAPPING = {
    "agentview_image": "image",
    "robot0_eye_in_hand_image": "wrist_image",
}


@dataclasses.dataclass(slots=True)
class RunSpec:
    systems: list[str]
    task_suite_name: str
    task_ids: list[int]
    seeds: list[int]
    max_steps: int
    episode_index: int
    num_steps_wait: int
    observation_width: int
    observation_height: int
    control_hz: float
    output_root: pathlib.Path


@dataclasses.dataclass(slots=True)
class EpisodeMetrics:
    system: str
    task_suite_name: str
    task_id: int
    task_language: str
    seed: int
    episode_index: int
    max_steps: int
    steps_executed: int
    success: bool
    terminated: bool
    truncated: bool
    total_reward: float
    duration_s: float
    first_action_latency_ms: float | None
    step_latency_ms_avg: float | None
    step_latency_ms_p50: float | None
    step_latency_ms_p95: float | None
    step_latency_ms_max: float | None
    worker_act_ms_avg: float | None
    worker_model_ms_avg: float | None
    failure: str | None
    final_info: dict[str, Any]


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _float_ms(value_s: float | None) -> float | None:
    return None if value_s is None else round(value_s * 1000.0, 3)


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _make_output_dir(output_root: pathlib.Path) -> pathlib.Path:
    output_dir = output_root / _timestamp()
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _augment_env(base_env: dict[str, str], extra_env: Sequence[str], extra_pythonpath: Sequence[str]) -> dict[str, str]:
    env = dict(base_env)
    rocm_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = DEFAULT_ROCM_LD_LIBRARY_PATH + (f":{rocm_ld}" if rocm_ld else "")

    for item in extra_env:
        key, _, value = item.partition("=")
        if not key or not _:
            raise ValueError(f"Invalid env override {item!r}; expected NAME=VALUE")
        env[key] = value

    if extra_pythonpath:
        existing = env.get("PYTHONPATH", "")
        joined = ":".join(extra_pythonpath)
        env["PYTHONPATH"] = joined + (f":{existing}" if existing else "")

    return env


class WorkerClient:
    def __init__(
        self,
        *,
        system: str,
        python_exe: str,
        script_path: pathlib.Path,
        worker_args: list[str],
        worker_env: dict[str, str],
    ):
        self.system = system
        self._tmpdir = tempfile.TemporaryDirectory(prefix=f"libero-compare-{system}-")
        self._socket_path = os.path.join(self._tmpdir.name, "worker.sock")
        self._listener = Listener(self._socket_path, family="AF_UNIX")
        cmd = [python_exe, str(script_path), "--worker", "--system", system, "--socket-path", self._socket_path, *worker_args]
        self._proc = subprocess.Popen(cmd, env=worker_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self._conn = self._listener.accept()
        except Exception:
            self.close(force=True)
            raise

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._conn.send(payload)
        response = self._conn.recv()
        if not response.get("ok", False):
            raise RuntimeError(
                f"{self.system} worker failed on {payload.get('cmd')}: {response.get('error')}\n"
                f"{response.get('traceback', '')}"
            )
        return response

    def load(self) -> dict[str, Any]:
        return self._request({"cmd": "load"})

    def reset_episode(self, task: str) -> dict[str, Any]:
        return self._request({"cmd": "reset_episode", "task": task})

    def act(self, observation: dict[str, np.ndarray]) -> dict[str, Any]:
        return self._request({"cmd": "act", "observation": observation})

    def close(self, force: bool = False) -> None:
        try:
            if not force:
                self._request({"cmd": "close"})
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass
        try:
            self._listener.close()
        except Exception:
            pass
        try:
            if force:
                self._proc.kill()
            else:
                self._proc.wait(timeout=5)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        finally:
            self._tmpdir.cleanup()

    def collect_logs(self) -> dict[str, str]:
        stdout = ""
        stderr = ""
        if self._proc.stdout is not None:
            stdout = self._proc.stdout.read()
        if self._proc.stderr is not None:
            stderr = self._proc.stderr.read()
        return {"stdout": stdout, "stderr": stderr}


def _flatten_libero_observation(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "state": np.asarray(observation["agent_pos"], dtype=np.float32),
        "image": np.ascontiguousarray(observation["pixels"]["image"]),
        "wrist_image": np.ascontiguousarray(observation["pixels"]["wrist_image"]),
    }


def _create_libero_env(task_suite_name: str, task_id: int, episode_index: int, num_steps_wait: int, width: int, height: int):
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv

    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    env = LiberoEnv(
        task_suite=task_suite,
        task_id=task_id,
        task_suite_name=task_suite_name,
        camera_name=CAMERA_NAMES,
        camera_name_mapping=CAMERA_NAME_MAPPING,
        obs_type="pixels_agent_pos",
        observation_width=width,
        observation_height=height,
        episode_index=episode_index,
        num_steps_wait=num_steps_wait,
    )
    task_language = task_suite.get_task(task_id).language
    return env, task_language


def _summarize_episode(
    *,
    system: str,
    task_suite_name: str,
    task_id: int,
    task_language: str,
    seed: int,
    episode_index: int,
    max_steps: int,
    step_latencies_s: list[float],
    worker_act_s: list[float],
    worker_model_ms: list[float],
    steps_executed: int,
    success: bool,
    terminated: bool,
    truncated: bool,
    total_reward: float,
    duration_s: float,
    failure: str | None,
    final_info: dict[str, Any],
) -> EpisodeMetrics:
    avg_step = statistics.fmean(step_latencies_s) if step_latencies_s else None
    avg_worker = statistics.fmean(worker_act_s) if worker_act_s else None
    avg_worker_model = statistics.fmean(worker_model_ms) if worker_model_ms else None
    return EpisodeMetrics(
        system=system,
        task_suite_name=task_suite_name,
        task_id=task_id,
        task_language=task_language,
        seed=seed,
        episode_index=episode_index,
        max_steps=max_steps,
        steps_executed=steps_executed,
        success=success,
        terminated=terminated,
        truncated=truncated,
        total_reward=round(total_reward, 6),
        duration_s=round(duration_s, 6),
        first_action_latency_ms=_float_ms(step_latencies_s[0] if step_latencies_s else None),
        step_latency_ms_avg=_float_ms(avg_step),
        step_latency_ms_p50=_float_ms(_percentile(step_latencies_s, 50)),
        step_latency_ms_p95=_float_ms(_percentile(step_latencies_s, 95)),
        step_latency_ms_max=_float_ms(max(step_latencies_s) if step_latencies_s else None),
        worker_act_ms_avg=_float_ms(avg_worker),
        worker_model_ms_avg=round(avg_worker_model, 3) if avg_worker_model is not None else None,
        failure=failure,
        final_info=final_info,
    )


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def run_driver(args: argparse.Namespace) -> int:
    run_spec = RunSpec(
        systems=list(args.systems),
        task_suite_name=args.task_suite_name,
        task_ids=_parse_csv_ints(args.task_ids),
        seeds=_parse_csv_ints(args.seeds),
        max_steps=args.max_steps,
        episode_index=args.episode_index,
        num_steps_wait=args.num_steps_wait,
        observation_width=args.observation_width,
        observation_height=args.observation_height,
        control_hz=args.control_hz,
        output_root=pathlib.Path(args.output_root),
    )
    script_path = pathlib.Path(__file__).resolve()
    output_dir = _make_output_dir(run_spec.output_root)
    _write_json(output_dir / "run_spec.json", run_spec)

    system_summaries: list[dict[str, Any]] = []

    for system in run_spec.systems:
        system_dir = output_dir / system
        system_dir.mkdir(parents=True, exist_ok=True)
        if system == "openpi":
            python_exe = args.openpi_python
            worker_args = [
                "--checkpoint-path", args.openpi_checkpoint_path,
                "--device", args.openpi_device,
                "--openpi-config-name", args.openpi_config_name,
            ]
            if args.openpi_norm_stats_source:
                worker_args.extend(["--openpi-norm-stats-source", args.openpi_norm_stats_source])
            if args.openpi_pytorch_compile_mode is not None:
                worker_args.extend(["--openpi-pytorch-compile-mode", args.openpi_pytorch_compile_mode])
            worker_env = _augment_env(os.environ, args.openpi_env, args.openpi_pythonpath)
        else:
            python_exe = args.vlash_python
            worker_args = [
                "--checkpoint-path", args.vlash_checkpoint_path,
                "--device", args.vlash_device,
                "--vlash-policy-type", args.vlash_policy_type,
                "--vlash-compile-model", str(args.vlash_compile_model).lower(),
                "--vlash-compile-mode", args.vlash_compile_mode,
                "--vlash-staged-compile-mode", args.vlash_staged_compile_mode,
                "--vlash-action-quant-ratio", str(args.vlash_action_quant_ratio),
                "--vlash-inference-overlap-steps", str(args.vlash_inference_overlap_steps),
                "--vlash-suffix-backend", args.vlash_suffix_backend,
            ]
            worker_env = _augment_env(os.environ, args.vlash_env, args.vlash_pythonpath)

        worker = WorkerClient(
            system=system,
            python_exe=python_exe,
            script_path=script_path,
            worker_args=worker_args,
            worker_env=worker_env,
        )
        logs: dict[str, str] | None = None
        episodes: list[EpisodeMetrics] = []
        load_info: dict[str, Any] | None = None
        try:
            load_info = worker.load()
            _write_json(system_dir / "load.json", load_info)

            for task_id in run_spec.task_ids:
                for seed in run_spec.seeds:
                    env, task_language = _create_libero_env(
                        task_suite_name=run_spec.task_suite_name,
                        task_id=task_id,
                        episode_index=run_spec.episode_index,
                        num_steps_wait=run_spec.num_steps_wait,
                        width=run_spec.observation_width,
                        height=run_spec.observation_height,
                    )
                    episode_name = f"task{task_id:02d}-seed{seed:03d}"
                    failure: str | None = None
                    final_info: dict[str, Any] = {}
                    start_s = time.perf_counter()
                    step_latencies_s: list[float] = []
                    worker_act_s: list[float] = []
                    worker_model_ms: list[float] = []
                    steps_executed = 0
                    total_reward = 0.0
                    success = False
                    terminated = False
                    truncated = False
                    try:
                        observation, reset_info = env.reset(seed=seed)
                        final_info = dict(reset_info)
                        worker.reset_episode(task_language)
                        for step in range(run_spec.max_steps):
                            obs_payload = _flatten_libero_observation(observation)
                            loop_start = time.perf_counter()
                            response = worker.act(obs_payload)
                            elapsed_s = time.perf_counter() - loop_start
                            step_latencies_s.append(elapsed_s)
                            worker_act_s.append(response.get("worker_act_s", elapsed_s))
                            model_ms = response.get("model_ms")
                            if model_ms is not None and math.isfinite(model_ms):
                                worker_model_ms.append(float(model_ms))
                            action = np.asarray(response["action"], dtype=np.float32)
                            observation, reward, terminated, truncated, info = env.step(action)
                            final_info = dict(info)
                            total_reward += float(reward)
                            steps_executed = step + 1
                            success = bool(info.get("is_success", False))
                            if success or terminated or truncated:
                                break
                            if run_spec.control_hz > 0:
                                remaining = (1.0 / run_spec.control_hz) - elapsed_s
                                if remaining > 0:
                                    time.sleep(remaining)
                    except Exception as exc:
                        failure = f"{type(exc).__name__}: {exc}"
                    finally:
                        try:
                            env.close()
                        except Exception:
                            pass

                    metrics = _summarize_episode(
                        system=system,
                        task_suite_name=run_spec.task_suite_name,
                        task_id=task_id,
                        task_language=task_language,
                        seed=seed,
                        episode_index=run_spec.episode_index,
                        max_steps=run_spec.max_steps,
                        step_latencies_s=step_latencies_s,
                        worker_act_s=worker_act_s,
                        worker_model_ms=worker_model_ms,
                        steps_executed=steps_executed,
                        success=success,
                        terminated=terminated,
                        truncated=truncated,
                        total_reward=total_reward,
                        duration_s=time.perf_counter() - start_s,
                        failure=failure,
                        final_info=final_info,
                    )
                    episodes.append(metrics)
                    _write_json(system_dir / f"{episode_name}.json", metrics)
        finally:
            worker.close(force=False)
            logs = worker.collect_logs()
            _write_json(system_dir / "worker_logs.json", logs)

        summary = {
            "system": system,
            "load": load_info,
            "episodes": episodes,
        }
        _write_json(system_dir / "summary.json", summary)
        system_summaries.append(summary)

    _write_json(output_dir / "summary.json", system_summaries)
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))
    return 0


def _worker_reply(conn, payload: dict[str, Any]) -> None:
    conn.send({"ok": True, **payload})


def _worker_error(conn, exc: Exception) -> None:
    conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


def run_worker(args: argparse.Namespace) -> int:
    conn = Client(args.socket_path, family="AF_UNIX")
    runtime: dict[str, Any] = {"loaded": False}
    try:
        while True:
            request = conn.recv()
            cmd = request["cmd"]
            try:
                if cmd == "load":
                    started = time.perf_counter()
                    if args.system == "openpi":
                        runtime.update(_load_openpi_runtime(args))
                    elif args.system == "vlash":
                        runtime.update(_load_vlash_runtime(args))
                    else:
                        raise ValueError(f"Unsupported system {args.system!r}")
                    runtime["loaded"] = True
                    _worker_reply(
                        conn,
                        {
                            "load_s": round(time.perf_counter() - started, 6),
                            "system": args.system,
                            "device": runtime.get("device"),
                            "checkpoint_path": args.checkpoint_path,
                            "details": runtime.get("details", {}),
                        },
                    )
                elif cmd == "reset_episode":
                    task = request.get("task") or ""
                    _reset_worker_episode(runtime, task)
                    _worker_reply(conn, {"task": task})
                elif cmd == "act":
                    started = time.perf_counter()
                    action, model_ms = _worker_act(runtime, request["observation"])
                    _worker_reply(
                        conn,
                        {
                            "action": np.asarray(action, dtype=np.float32),
                            "worker_act_s": round(time.perf_counter() - started, 6),
                            "model_ms": model_ms,
                        },
                    )
                elif cmd == "close":
                    _close_worker(runtime)
                    _worker_reply(conn, {"closed": True})
                    break
                else:
                    raise ValueError(f"Unknown worker command {cmd!r}")
            except Exception as exc:
                _worker_error(conn, exc)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return 0


def _load_openpi_runtime(args: argparse.Namespace) -> dict[str, Any]:
    import dataclasses
    from openpi.policies.policy_config import create_trained_policy
    import openpi.shared.download as download
    from openpi.training import checkpoints as training_checkpoints
    from openpi.training import config as training_config
    from openpi_client.action_chunk_broker import ActionChunkBroker

    train_config = training_config.get_config(args.openpi_config_name)
    train_config = dataclasses.replace(
        train_config,
        model=dataclasses.replace(
            train_config.model,
            pytorch_compile_mode=args.openpi_pytorch_compile_mode,
        ),
    )
    norm_stats = None
    if args.openpi_norm_stats_source:
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        if data_config.asset_id is None:
            raise ValueError("Train config does not define an asset_id for norm stats loading")
        norm_assets_dir = pathlib.Path(download.maybe_download(args.openpi_norm_stats_source)) / "assets"
        norm_stats = training_checkpoints.load_norm_stats(norm_assets_dir, data_config.asset_id)
    policy = create_trained_policy(
        train_config,
        args.checkpoint_path,
        pytorch_device=args.device,
        norm_stats=norm_stats,
    )
    action_horizon = getattr(train_config.model, "action_horizon", 50)
    broker = ActionChunkBroker(policy, action_horizon=action_horizon)
    return {
        "kind": "openpi",
        "device": args.device,
        "policy": policy,
        "broker": broker,
        "prompt": "",
        "details": {
            "config_name": args.openpi_config_name,
            "action_horizon": action_horizon,
            "norm_stats_source": args.openpi_norm_stats_source,
            "pytorch_compile_mode": args.openpi_pytorch_compile_mode,
        },
    }


class _DummyLiberoRobot:
    robot_type = "libero_robot"

    @property
    def action_features(self) -> dict[str, type[float]]:
        return {f"action_{index}": float for index in range(7)}


def _load_vlash_runtime(args: argparse.Namespace) -> dict[str, Any]:
    from lerobot.configs.policies import PreTrainedConfig
    from vlash.policies.factory import get_policy_class

    config = PreTrainedConfig.from_pretrained(args.checkpoint_path)
    config.device = args.device
    config.compile_model = args.vlash_compile_model
    config.compile_mode = args.vlash_compile_mode
    if hasattr(config, "staged_compile_mode"):
        config.staged_compile_mode = args.vlash_staged_compile_mode
    if hasattr(config, "suffix_backend"):
        config.suffix_backend = args.vlash_suffix_backend
    policy_cls = get_policy_class(args.vlash_policy_type)
    policy = policy_cls.from_pretrained(args.checkpoint_path, config=config)
    return {
        "kind": "vlash",
        "device": args.device,
        "policy": policy,
        "robot": _DummyLiberoRobot(),
        "manager": None,
        "task": "",
        "details": {
            "policy_type": args.vlash_policy_type,
            "compile_model": args.vlash_compile_model,
            "compile_mode": args.vlash_compile_mode,
            "staged_compile_mode": args.vlash_staged_compile_mode,
            "action_quant_ratio": args.vlash_action_quant_ratio,
            "inference_overlap_steps": args.vlash_inference_overlap_steps,
            "suffix_backend": args.vlash_suffix_backend,
        },
    }


def _make_vlash_manager(runtime: dict[str, Any], task: str, args: argparse.Namespace):
    from vlash.run import VLASHAsyncManager

    effective_overlap_steps = args.vlash_inference_overlap_steps * args.vlash_action_quant_ratio
    return VLASHAsyncManager(
        policy=runtime["policy"],
        robot=runtime["robot"],
        single_task=task,
        overlap_steps=effective_overlap_steps,
        image_feature_map=None,
        suffix_backend_check=False,
    )


def _reset_worker_episode(runtime: dict[str, Any], task: str) -> None:
    if runtime["kind"] == "openpi":
        runtime["prompt"] = task
        runtime["broker"].reset()
    else:
        runtime["task"] = task
        runtime["policy"].reset()
        runtime["manager"] = _make_vlash_manager(runtime, task, runtime["args"])


def _worker_act(runtime: dict[str, Any], observation: dict[str, np.ndarray]) -> tuple[np.ndarray, float | None]:
    if runtime["kind"] == "openpi":
        payload = {
            "observation/state": np.asarray(observation["state"], dtype=np.float32),
            "observation/image": np.ascontiguousarray(observation["image"]),
            "observation/wrist_image": np.ascontiguousarray(observation["wrist_image"]),
            "prompt": runtime["prompt"],
        }
        result = runtime["broker"].infer(payload)
        action = np.asarray(result["actions"], dtype=np.float32)
        model_ms = result.get("policy_timing", {}).get("infer_ms")
        return action, None if model_ms is None else float(model_ms)

    if runtime["manager"] is None:
        runtime["manager"] = _make_vlash_manager(runtime, runtime["task"], runtime["args"])
    frame = {
        "observation.state": np.asarray(observation["state"], dtype=np.float32),
        "observation.images.image": np.ascontiguousarray(observation["image"]),
        "observation.images.wrist_image": np.ascontiguousarray(observation["wrist_image"]),
    }
    action_dict = runtime["manager"].get_action(frame)
    action = np.asarray([action_dict[f"action_{index}"] for index in range(7)], dtype=np.float32)
    return action, None


def _close_worker(runtime: dict[str, Any]) -> None:
    if runtime.get("kind") == "vlash":
        try:
            runtime["policy"].reset()
        except Exception:
            pass
    elif runtime.get("kind") == "openpi":
        try:
            runtime["broker"].reset()
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help="Internal worker mode")
    parser.add_argument("--system", choices=["openpi", "vlash"], help="System name in worker mode")
    parser.add_argument("--socket-path", help="AF_UNIX socket path in worker mode")
    parser.add_argument("--checkpoint-path", help="Checkpoint path for worker mode")
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--systems", nargs="+", choices=["openpi", "vlash"], default=["openpi", "vlash"])
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--task-ids", default="0")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--observation-width", type=int, default=256)
    parser.add_argument("--observation-height", type=int, default=256)
    parser.add_argument("--control-hz", type=float, default=0.0, help="Optional fixed control cadence; 0 disables sleeping")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--openpi-python", default=DEFAULT_OPENPI_PYTHON)
    parser.add_argument("--openpi-checkpoint-path", default=DEFAULT_OPENPI_CHECKPOINT)
    parser.add_argument("--openpi-config-name", default="pi05_libero")
    parser.add_argument("--openpi-device", default="cuda:0")
    parser.add_argument("--openpi-norm-stats-source")
    parser.add_argument("--openpi-pytorch-compile-mode", default=None)
    parser.add_argument("--openpi-env", action="append", default=[])
    parser.add_argument("--openpi-pythonpath", action="append", default=[])

    parser.add_argument("--vlash-python", default=DEFAULT_VLASH_PYTHON)
    parser.add_argument("--vlash-checkpoint-path", default=DEFAULT_VLASH_CHECKPOINT)
    parser.add_argument("--vlash-device", default="cuda:0")
    parser.add_argument("--vlash-policy-type", default="pi05")
    parser.add_argument("--vlash-compile-model", type=lambda raw: raw.lower() == "true", default=True)
    parser.add_argument("--vlash-compile-mode", default="max-autotune")
    parser.add_argument("--vlash-staged-compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--vlash-action-quant-ratio", type=int, default=1)
    parser.add_argument("--vlash-inference-overlap-steps", type=int, default=1)
    parser.add_argument("--vlash-suffix-backend", default="local")
    parser.add_argument("--vlash-env", action="append", default=[])
    parser.add_argument("--vlash-pythonpath", action="append", default=[])

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.worker:
        if not args.system or not args.socket_path or not args.checkpoint_path:
            parser.error("--worker requires --system, --socket-path, and --checkpoint-path")
        if args.system == "vlash":
            # Keep worker args available for staged manager reconstruction.
            args.vlash_compile_model = bool(args.vlash_compile_model)
        runtime_args = argparse.Namespace(**vars(args))
        # Make args visible inside runtime dicts.
        globals()["_worker_args_snapshot"] = runtime_args
        return run_worker(runtime_args)
    return run_driver(args)


# Attach parsed args to vlash runtime after loading.
_original_load_vlash_runtime = _load_vlash_runtime

def _load_vlash_runtime(args: argparse.Namespace) -> dict[str, Any]:  # type: ignore[no-redef]
    runtime = _original_load_vlash_runtime(args)
    runtime["args"] = args
    return runtime


if __name__ == "__main__":
    raise SystemExit(main())
