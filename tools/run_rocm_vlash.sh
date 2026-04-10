#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-/home/amd/.venvs/rocm-probe-312}"
ROCM_ROOT="${ROCM_ROOT:-/opt/rocm-7.2.1}"
ROCM_EXPERIMENTAL_ATTENTION="${ROCM_EXPERIMENTAL_ATTENTION:-}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing ROCm venv at ${VENV_DIR}" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${ROCM_ROOT}/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export PATH="${VENV_DIR}/bin:${PATH}"

if [[ -n "${ROCM_EXPERIMENTAL_ATTENTION}" ]]; then
  export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${ROCM_EXPERIMENTAL_ATTENTION}"
fi

cd "${ROOT_DIR}"
exec "${VENV_DIR}/bin/python" -m vlash.cli "$@"
