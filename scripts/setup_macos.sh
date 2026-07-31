#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
DEPS_DIR="${TRELLIS_DEPS_DIR:-$HOME/.cache/trellis2/source-deps}"
PRIMARY_TORCH="2.13.0"
PRIMARY_TORCHVISION="0.28.0"
FALLBACK_TORCH="2.11.0"
FALLBACK_TORCHVISION="0.26.0"

MTLGEMM_SHA="867aec8234299a7fe1ede7f802c8debe5a939a82"
MTLDIFFRAST_SHA="4668cd91cb6d27f5e264731f94a06841fbf7aab8"
MTLBVH_SHA="23f441c470ce1f537e1fd836f3ffb5b8245f7975"
MTLMESH_SHA="212079e55772cff3d648a21372392c37e0643f3b"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "ERROR: this installer supports Apple Silicon macOS only" >&2
    exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python 3.11 is required (set PYTHON_BIN if it is installed elsewhere)" >&2
    exit 2
fi
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))'; then
    echo "ERROR: $PYTHON_BIN is not Python 3.11" >&2
    exit 2
fi

if [[ "${SKIP_METAL:-0}" != "1" ]] && ! xcrun --find metal >/dev/null 2>&1; then
    echo "Metal Toolchain is missing; downloading the matching Xcode component..."
    xcodebuild -downloadComponent MetalToolchain
fi

if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python"
PIP=("$PYTHON" -m pip)
"${PIP[@]}" install --upgrade 'pip==26.0.1' 'setuptools==80.9.0' 'wheel==0.46.3' 'ninja==1.13.0' 'pybind11==3.0.1'
"${PIP[@]}" install -r "$ROOT/requirements_macos_core.txt"

install_torch_pair() {
    local torch_version="$1"
    local torchvision_version="$2"
    "${PIP[@]}" uninstall -y torch torchvision mtldiffrast mtlbvh cumesh flex_gemm o_voxel >/dev/null 2>&1 || true
    "${PIP[@]}" install "torch==$torch_version" "torchvision==$torchvision_version"
}

checkout_dependency() {
    local name="$1"
    local url="$2"
    local revision="$3"
    local destination="$DEPS_DIR/$name-$revision"
    local attempt
    if [[ -d "$destination/.git" ]] && [[ "$(git -C "$destination" rev-parse HEAD)" == "$revision" ]]; then
        DEP_PATH="$destination"
        return 0
    fi
    mkdir -p "$DEPS_DIR"
    for attempt in 1 2 3; do
        rm -rf "$destination"
        if git -c http.version=HTTP/1.1 clone --filter=blob:none "$url" "$destination" \
            && git -C "$destination" -c http.version=HTTP/1.1 fetch --depth 1 origin "$revision" \
            && git -C "$destination" switch --detach "$revision" \
            && git -C "$destination" -c http.version=HTTP/1.1 submodule update --init --recursive; then
            DEP_PATH="$destination"
            return 0
        fi
        echo "Dependency checkout failed ($name, attempt $attempt/3); retrying..." >&2
    done
    echo "ERROR: failed to checkout pinned dependency $name@$revision" >&2
    return 1
}

install_metal_stack() {
    export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-12.0}"
    checkout_dependency mtlbvh https://github.com/pedronaugusto/mtlbvh.git "$MTLBVH_SHA" || return 1
    "${PIP[@]}" install --no-build-isolation "$DEP_PATH" || return 1
    checkout_dependency mtldiffrast https://github.com/pedronaugusto/mtldiffrast.git "$MTLDIFFRAST_SHA" || return 1
    "${PIP[@]}" install --no-build-isolation "$DEP_PATH" || return 1
    checkout_dependency mtlmesh https://github.com/pedronaugusto/mtlmesh.git "$MTLMESH_SHA" || return 1
    "${PIP[@]}" install --no-build-isolation "$DEP_PATH" || return 1
    checkout_dependency mtlgemm https://github.com/pedronaugusto/mtlgemm.git "$MTLGEMM_SHA" || return 1
    "${PIP[@]}" install --no-build-isolation "$DEP_PATH" || return 1
    BUILD_TARGET=cpu "${PIP[@]}" install --no-build-isolation -e "$ROOT/o-voxel" || return 1
}

install_torch_pair "$PRIMARY_TORCH" "$PRIMARY_TORCHVISION"
resolved_torch="$PRIMARY_TORCH"
resolved_torchvision="$PRIMARY_TORCHVISION"

if [[ "${SKIP_METAL:-0}" == "1" ]]; then
    BUILD_TARGET=cpu "${PIP[@]}" install --no-build-isolation -e "$ROOT/o-voxel"
else
    if ! install_metal_stack || ! "$PYTHON" "$ROOT/scripts/probe_macos.py" --require-metal; then
        echo "Primary PyTorch pair failed the Metal build/probe; retrying the prescribed ABI fallback..." >&2
        install_torch_pair "$FALLBACK_TORCH" "$FALLBACK_TORCHVISION"
        install_metal_stack
        "$PYTHON" "$ROOT/scripts/probe_macos.py" --require-metal
        resolved_torch="$FALLBACK_TORCH"
        resolved_torchvision="$FALLBACK_TORCHVISION"
    fi
fi

"${PIP[@]}" check
"$PYTHON" "$ROOT/scripts/probe_macos.py"

echo
echo "TRELLIS.2 Apple environment is ready."
echo "  source $VENV_DIR/bin/activate"
echo "  torch==$resolved_torch torchvision==$resolved_torchvision"
if ! "$PYTHON" -c 'from huggingface_hub import get_token; raise SystemExit(not bool(get_token()))' 2>/dev/null; then
    echo "  Hugging Face auth is missing: run '$VENV_DIR/bin/hf auth login' before downloading gated weights."
fi
