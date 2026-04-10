#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-/home/amd/.venvs/rocm-probe-312}"
SOURCE_ENV_PYTHON="${SOURCE_ENV_PYTHON:-/home/amd/.miniforge3/envs/vlash/bin/python}"
ROCM_ROOT="${ROCM_ROOT:-/opt/rocm-7.2.1}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
TMP_REQ="$(mktemp)"

cleanup() {
  rm -f "${TMP_REQ}"
}
trap cleanup EXIT

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Missing ${PYTHON_BIN}. Install Python 3.12 first or override PYTHON_BIN." >&2
  exit 1
fi

if [[ ! -x "${SOURCE_ENV_PYTHON}" ]]; then
  echo "Missing source environment python at ${SOURCE_ENV_PYTHON}" >&2
  echo "This script reuses the already-working package set from the existing vlash env." >&2
  exit 1
fi

if [[ ! -d "${ROCM_ROOT}" ]]; then
  echo "Missing ROCm runtime at ${ROCM_ROOT}" >&2
  exit 1
fi

echo "[1/7] Creating venv at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

export PATH="${VENV_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_ROOT}/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"

echo "[2/7] Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "[3/7] Installing ROCm PyTorch stack"
python -m pip install \
  "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl" \
  "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl" \
  "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl" \
  "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl"

echo "[4/7] Exporting package set from source env"
"${SOURCE_ENV_PYTHON}" -m pip freeze \
  | grep -vE '^(torch|torchvision|torchaudio|torchcodec|triton|bitsandbytes|nvidia-.*|numpy)==|^-e ' \
  > "${TMP_REQ}"

echo "[5/7] Reinstalling compatible Python packages without touching ROCm torch"
python -m pip install --no-deps -r "${TMP_REQ}"

echo "[6/7] Installing local VLASH package without dependency resolution"
python -m pip install -e "${ROOT_DIR}" --no-deps

echo "[7/7] Verifying ROCm runtime"
python - <<'PY'
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
print(json.dumps(payload, indent=2))
PY

cat <<EOF

ROCm VLASH environment created.

Next steps:
  export LD_LIBRARY_PATH="${ROCM_ROOT}/lib:/opt/rocm/lib:\${LD_LIBRARY_PATH:-}"
  export PATH="${VENV_DIR}/bin:\${PATH}"
  tools/check_rocm_runtime.sh
  tools/run_rocm_vlash.sh benchmark examples/benchmarks/inference_latency_rocm_smoke.yaml \\
    --policy.path=/home/amd/.cache/vlash/models/vlash-pi05-libero-async5
EOF
