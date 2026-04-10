#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-/home/amd/.venvs/rocm-probe-312}"
ROCM_ROOT="${ROCM_ROOT:-/opt/rocm-7.2.1}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing ROCm venv at ${VENV_DIR}" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${ROCM_ROOT}/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export PATH="${VENV_DIR}/bin:${PATH}"

cd "${ROOT_DIR}"
exec "${VENV_DIR}/bin/python" - <<'PY'
import json
import torch

payload = {
    "torch_version": torch.__version__,
    "torch_hip_version": torch.version.hip,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
}

if payload["cuda_available"] and payload["device_count"] > 0:
    payload["device_name"] = torch.cuda.get_device_name(0)
    payload["bf16_supported"] = torch.cuda.is_bf16_supported()
else:
    payload["device_name"] = None
    payload["bf16_supported"] = None

print(json.dumps(payload, indent=2))
PY
