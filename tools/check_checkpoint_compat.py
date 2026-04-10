#!/usr/bin/env python3

"""Static compatibility checker for VLASH checkpoints.

This tool compares a local or Hugging Face checkpoint against a VLASH run
config and reports whether the checkpoint can be used with the current
`vlash run` implementation without code changes.

It intentionally avoids importing torch or lerobot so it can run in a thin
environment and still answer the most important compatibility questions:

- Is this checkpoint in a format current VLASH loaders understand?
- Do the checkpoint's image feature names match the robot camera names?
- Does the run config request async overlap while the checkpoint disables
  compile_model by default?
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def fetch_json(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP error while fetching {url}: {exc.code} {exc.reason}\n{body}") from exc


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "PyYAML is required to parse run configs. Install `pyyaml` or run in the VLASH env."
        ) from exc

    with path.open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"Run config at {path} did not parse into a mapping.")
    return data


def hf_model_api(repo_id: str) -> str:
    escaped = urllib.parse.quote(repo_id, safe="/")
    return f"https://huggingface.co/api/models/{escaped}"


def hf_raw_file(repo_id: str, filename: str) -> str:
    escaped = urllib.parse.quote(repo_id, safe="/")
    return f"https://huggingface.co/{escaped}/raw/main/{filename}"


def load_checkpoint_metadata(model_ref: str) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    model_path = Path(model_ref)
    if model_path.exists():
        if model_path.is_dir():
            config_path = model_path / "config.json"
            if not config_path.is_file():
                raise SystemExit(f"Local checkpoint directory is missing config.json: {config_path}")
            with config_path.open() as handle:
                config = json.load(handle)
            metadata = {
                "id": str(model_path),
                "source": "local",
                "siblings": [{"rfilename": p.name} for p in model_path.iterdir()],
                "private": False,
                "gated": False,
                "disabled": False,
            }
            return str(model_path), metadata, config
        raise SystemExit(f"Expected a directory for local checkpoint, got file: {model_path}")

    metadata = fetch_json(hf_model_api(model_ref))
    siblings = {item["rfilename"] for item in metadata.get("siblings", [])}
    config = None
    if "config.json" in siblings:
        config = fetch_json(hf_raw_file(model_ref, "config.json"))
    return model_ref, metadata, config


def infer_model_image_features(config: dict[str, Any]) -> list[str]:
    input_features = config.get("input_features", {})
    image_keys = []
    for key, value in input_features.items():
        if not isinstance(value, dict):
            continue
        if value.get("type") == "VISUAL" or key.startswith("observation.images."):
            image_keys.append(key)
    return sorted(image_keys)


def normalize_image_feature_name(name: str) -> str:
    return name if name.startswith("observation.images.") else f"observation.images.{name}"


def infer_robot_image_features(run_config: dict[str, Any]) -> list[str]:
    robot = run_config.get("robot", {})
    cameras = robot.get("cameras", {}) if isinstance(robot, dict) else {}
    if not isinstance(cameras, dict):
        return []
    return sorted(f"observation.images.{name}" for name in cameras)


def infer_camera_feature_map(run_config: dict[str, Any]) -> dict[str, str]:
    mapping = run_config.get("camera_feature_map", {})
    if not isinstance(mapping, dict):
        return {}
    return {
        normalize_image_feature_name(str(target)): normalize_image_feature_name(str(source))
        for target, source in mapping.items()
    }


def has_supported_loader_files(metadata: dict[str, Any]) -> tuple[bool, list[str]]:
    siblings = {item["rfilename"] for item in metadata.get("siblings", [])}
    missing = []
    for required in ("config.json", "model.safetensors"):
        if required not in siblings:
            missing.append(required)
    return not missing, missing


def summarize_issues(
    run_config: dict[str, Any],
    model_config: dict[str, Any] | None,
    metadata: dict[str, Any],
) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    notes: list[str] = []

    supported_loader, missing_files = has_supported_loader_files(metadata)
    if not supported_loader:
        issues.append(
            "checkpoint format is unsupported by current VLASH loaders; "
            f"missing required files: {', '.join(missing_files)}"
        )

    if model_config is None:
        issues.append("checkpoint has no config.json, so input/output feature compatibility cannot be validated")
        return issues, notes

    model_images = infer_model_image_features(model_config)
    robot_images = infer_robot_image_features(run_config)
    camera_feature_map = infer_camera_feature_map(run_config)
    resolved_model_images = set()
    unknown_targets = sorted(set(camera_feature_map) - set(model_images))
    unknown_sources = sorted(set(camera_feature_map.values()) - set(robot_images))
    if unknown_targets or unknown_sources:
        issues.append(
            "run config camera_feature_map is invalid; "
            f"unknown_targets={unknown_targets or []}, unknown_sources={unknown_sources or []}"
        )
    else:
        for model_image in model_images:
            if model_image in camera_feature_map:
                resolved_model_images.add(model_image)
            elif model_image in robot_images:
                resolved_model_images.add(model_image)
        missing_model_images = sorted(set(model_images) - resolved_model_images)
        if robot_images and model_images and missing_model_images:
            issues.append(
                "robot camera features do not satisfy checkpoint image features even after camera_feature_map; "
                f"missing={missing_model_images}, robot={robot_images}, model={model_images}, "
                f"camera_feature_map={camera_feature_map}"
            )

    inference_overlap_steps = run_config.get("inference_overlap_steps", 0)
    compile_model = bool(model_config.get("compile_model", False))
    if inference_overlap_steps > 0 and not compile_model:
        notes.append(
            "run config uses async overlap but checkpoint default compile_model=false; "
            "you must override policy.compile_model=true"
        )

    policy_type = model_config.get("type")
    if policy_type not in {"pi0", "pi05"}:
        issues.append(f"checkpoint policy type is {policy_type!r}, not one of current VLASH runtime types ('pi0', 'pi05')")

    task = run_config.get("single_task")
    if task in {None, "<task description>"}:
        notes.append("run config still has an empty or placeholder single_task and will fail until you set it")

    policy_path = ((run_config.get("policy") or {}) if isinstance(run_config.get("policy"), dict) else {})
    if policy_path.get("path") == "<path to the policy checkpoint>":
        notes.append("run config still has a placeholder policy.path")

    return issues, notes


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether a VLASH checkpoint is compatible with a run config.")
    parser.add_argument("--model", required=True, help="Hugging Face repo id or local checkpoint directory")
    parser.add_argument(
        "--run-config",
        required=True,
        help="Path to a VLASH run YAML config, for example examples/inference/async.yaml",
    )
    args = parser.parse_args()

    run_config_path = Path(args.run_config)
    if not run_config_path.is_file():
        raise SystemExit(f"Run config not found: {run_config_path}")

    run_config = load_yaml(run_config_path)
    model_ref, metadata, model_config = load_checkpoint_metadata(args.model)
    issues, notes = summarize_issues(run_config, model_config, metadata)

    model_images = infer_model_image_features(model_config or {})
    robot_images = infer_robot_image_features(run_config)
    camera_feature_map = infer_camera_feature_map(run_config)

    print(f"Model: {model_ref}")
    print(f"Run config: {run_config_path}")
    print(f"Public: {not metadata.get('private', False)}  Gated: {metadata.get('gated', False)}")
    print(f"Model files: {', '.join(item['rfilename'] for item in metadata.get('siblings', []))}")
    if model_config is not None:
        print(f"Policy type: {model_config.get('type')}")
        print(f"Model image features: {model_images}")
    print(f"Robot image features: {robot_images}")
    if camera_feature_map:
        print(f"Camera feature map: {camera_feature_map}")

    if issues:
        print("\nCompatibility: INCOMPATIBLE")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("\nCompatibility: COMPATIBLE WITH CURRENT LOADER/RUNTIME CHECKS")

    if notes:
        print("\nNotes:")
        for note in notes:
            print(f"- {note}")

    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
