#!/usr/bin/env bash
# ==============================================================================
# Pipeline Image Builder & GPU Compatibility Verification Probe
#
# Builds the container images for all five pipeline stages and verifies that 
# host GPU capabilities match image kernel architectures before execution.
#
# Must run ON THE GPU HOST (not a workstation):
#   1. Images total ~50 GB in disk space.
#   2. Post-build capability probes require physical GPU access via NVIDIA Container Toolkit.
#
# Usage:
#   bash docker/build_all.sh              # Build and verify all 5 images
#   bash docker/build_all.sh esmfold      # Build and verify a single stage
# ==============================================================================

set -euo pipefail

# Navigate to project root relative to script location
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Target image list
readonly ALL_IMAGES=(rfdiffusion proteinmpnn esmfold boltz2 colabfold)
readonly TARGETS=("${@:-${ALL_IMAGES[@]}}")

# -----------------------------------------------------------------------------
# Logging Utilities
# -----------------------------------------------------------------------------
log_info()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
log_warn()  { printf '\033[1;33m-- warning: %s\033[0m\n' "$*"; }
log_fatal() { printf '\033[1;31m!! %s\033[0m\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# Pre-Flight Checks & Privileged Group Self-Elevation
# -----------------------------------------------------------------------------
# If the current process lacks active docker group privileges, add user and re-exec.
if ! groups | grep -qw docker; then
    if ! id -nG "$USER" | grep -qw docker; then
        log_info "Adding $USER to the docker group..."
        sudo usermod -aG docker "$USER"
    fi
    
    log_info "Re-executing script with active docker group privileges..."
    exec sg docker -c "$0 $*"
fi

command -v docker >/dev/null 2>&1 || log_fatal "Docker executable not found on PATH."

# Storage Pre-flight: Inspect available disk space on the Docker storage partition (~80 GB recommended)
avail_gb=$(df -BG --output=avail /var/lib/docker 2>/dev/null | tail -n 1 | tr -dc '0-9' || echo 0)
if [[ "${avail_gb:-0}" -lt 80 ]]; then
    log_warn "Only ${avail_gb}GB free where Docker stores images (/var/lib/docker); ~80GB recommended."
fi

# -----------------------------------------------------------------------------
# Build Phase
# -----------------------------------------------------------------------------
built_images=()

for name in "${TARGETS[@]}"; do
    dockerfile="docker/$name/Dockerfile"
    [[ -f "$dockerfile" ]] || log_fatal "Unknown image target or missing Dockerfile: docker/$name/Dockerfile"
    
    log_info "Building binder-${name}:latest"
    docker build --progress=plain -t "binder-${name}:latest" "docker/${name}"
    built_images+=("binder-${name}:latest")
done

log_info "Successfully built: ${built_images[*]}"

# -----------------------------------------------------------------------------
# GPU Compatibility Probes
# -----------------------------------------------------------------------------
# Every pipeline image can fail on specific GPU architectures without throwing a
# startup error.
#   - PyTorch images : Mismatched CUDA arch raises only when a tensor reaches the GPU 
#                      ("no kernel image is available"), minutes into a design run.
#   - ColabFold      : JAX silently falls back to CPU (~100x slowdown without errors).
#   - ESMFold        : Requires dual checks—PyTorch sm_90 compatibility does not 
#                      guarantee OpenFold custom C++/CUDA extension compatibility.
# -----------------------------------------------------------------------------

# The shared cache
readonly BINDER_CACHE="${BINDER_CACHE:-$HOME/.cache/binder-pipeline}"
mkdir -p "$BINDER_CACHE"

# A probe MUST run the way the campaign runs
probe() {
    local name="$1" ep="$2" label="$3"; shift 3
    # Skip probe if target wasn't selected in this run
    [[ " ${TARGETS[*]} " == *" $name "* ]] || return 0

    log_info "$label"
    docker run --rm --gpus all \
        --user "$(id -u):$(id -g)" \
        -v "$BINDER_CACHE:/cache" \
        --entrypoint "$ep" "binder-${name}:latest" "$@"
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
    log_warn "nvidia-smi not detected. Skipping GPU verification probes."
    log_warn "Images are UNVERIFIED against hardware—run on GPU host before starting campaigns."
else

# Common PyTorch verification snippet: forces a tensor onto VRAM
readonly TORCH_PROBE='import torch, sys
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() returned False—stage would run on CPU.")
cap = torch.cuda.get_device_capability()
torch.zeros(1).cuda()
print(f"OK: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} | PyTorch v{torch.__version__} | CUDA v{torch.version.cuda}")'

# --- 1. RFdiffusion Probe ---
probe rfdiffusion python "Checking RFdiffusion CUDA/GPU compatibility" -c "$TORCH_PROBE" \
|| log_fatal "RFdiffusion image cannot run on this GPU.
       If running on H100/B200 (sm_90+), CUDA 11.6 lacks sm_90 kernels (see MANIFEST Section 6.1).
       Switch to an A10/A100 host or update image base to PyTorch 2.x / CUDA 12."

# --- 2. ProteinMPNN Probe ---
probe proteinmpnn python "Checking ProteinMPNN CUDA/GPU compatibility" -c "$TORCH_PROBE" \
|| log_fatal "ProteinMPNN image cannot run on this GPU.
       ProteinMPNN falls back to CPU silently (protein_mpnn_run.py:68), leading to significant slowdowns.
       CUDA 11.8 covers sm_80-sm_90; verify host driver and runtime configurations."

# --- 3. Boltz-2 Probe (PyTorch Layer) ---
probe boltz2 python "Checking Boltz-2 CUDA/GPU compatibility" -c "$TORCH_PROBE" \
|| log_fatal "Boltz-2 image cannot run on this GPU.
       PyTorch 2.6/cu126 covers sm_80-sm_90; check NVIDIA host driver or nvidia-container-toolkit setup."

# --- 3b. Boltz-2 Probe (cuEquivariance Kernel Layer) ---
readonly CUEQ_PROBE='
import sys, torch
try:
    import cuequivariance_ops_torch  # forces the dlopen of libcue_ops.so
except Exception as e:
    sys.exit(f"cuEquivariance kernels are NOT loadable: {type(e).__name__}: {e}")

from cuequivariance_torch import triangle_multiplicative_update

n, d = 32, 128
dev = "cuda"
x = torch.randn(1, n, n, d, device=dev)
mask = torch.ones(1, n, n, device=dev)
vec = lambda: torch.randn(d, device=dev)
mat = lambda o: torch.randn(o, d, device=dev)
try:
    out = triangle_multiplicative_update(
        x, direction="outgoing", mask=mask,
        norm_in_weight=vec(), norm_in_bias=vec(),
        p_in_weight=mat(2 * d), g_in_weight=mat(2 * d),
        norm_out_weight=vec(), norm_out_bias=vec(),
        p_out_weight=mat(d), g_out_weight=mat(d),
        eps=1e-5,
    )
    torch.cuda.synchronize()
except Exception as e:
    sys.exit(f"cuEquivariance kernel loaded but FAILED to execute: {type(e).__name__}: {e}")
print(f"OK: cuEquivariance triangle kernel executed (fused layer norm + triton JIT), "
      f"out={tuple(out.shape)} | cuBLAS >= 12.5 satisfied | CUDA v{torch.version.cuda}")'

probe boltz2 python "Checking Boltz-2 cuEquivariance triangle kernels" -c "$CUEQ_PROBE" \
|| log_fatal "Boltz-2 cuEquivariance kernels unusable on this host.
       These accelerate the triangular multiplicative update; without them boltz
       must run with --no_kernels (correct results, slower).
       If the error mentions 'cublasGemmGroupedBatchedEx', the base image's CUDA
       is below 12.5 -- cuequivariance-ops-cu12 requires nvidia-cublas-cu12>=12.5.0.
       Current base is pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime."

# --- 4. ESMFold Probe (PyTorch Layer) ---
probe esmfold python "Checking ESMFold PyTorch/GPU compatibility" -c "$TORCH_PROBE" \
|| log_fatal "ESMFold image cannot run on this GPU at the PyTorch level."

# --- 5. ESMFold Probe (OpenFold C++/CUDA Extension Layer) ---
readonly OPENFOLD_PROBE='
import re, subprocess, sys, torch
try:
    import attn_core_inplace_cuda as m
except ImportError as exc:
    sys.exit(f"OpenFold CUDA extension missing: {exc}")

out = subprocess.run(["cuobjdump", "--list-elf", m.__file__], capture_output=True, text=True)
if out.returncode != 0:
    sys.exit(f"cuobjdump failed ({out.returncode}): {out.stderr.strip()[:200]}")

archs = sorted({int(n) for n in re.findall(r"sm_(\d+)", out.stdout)})
if not archs:
    sys.exit("cuobjdump listed no sm_ architectures; check tool output format.")

major, minor = torch.cuda.get_device_capability()
dev = major * 10 + minor
print(f"OpenFold compiled kernels: {archs} | Device: sm_{dev}")

# Binary compatibility: a cubin runs on a device of the SAME major whose minor is
# equal or HIGHER (compiled minor <= device minor). sm_80 therefore covers sm_86.
if not any(a // 10 == major and a % 10 <= minor for a in archs):
    sys.exit(f"No compiled OpenFold kernel matches sm_{dev}")
print("OK: OpenFold custom CUDA kernels cover this device.")'

probe esmfold python "Checking ESMFold OpenFold kernel architecture binary compatibility" -c "$OPENFOLD_PROBE" \
    || log_fatal "ESMFold OpenFold extension cannot serve this GPU.
       OpenFold binaries compile up to sm_80 with no PTX fallback; H100/B200 (sm_90+) is unsupported.
       Use an A10/A100 host system."

# --- 6. ColabFold Probe (JAX Backend) ---
# JAX falls back to CPU silently if GPU allocation fails. This probe verifies active GPU backend execution.
readonly COLABFOLD_PROBE='
import sys, jax, jax.numpy as jnp
backend = jax.default_backend()
devices = jax.devices()
print(f"JAX v{jax.__version__} | Backend: {backend!r} | Devices: {devices}")

if backend != "gpu":
    sys.exit(f"JAX selected {backend!r} backend. ColabFold would silently run on CPU (~100x slowdown). "
             "Check host driver (>= 525.60.13 for CUDA 12) and nvidia-container-toolkit installation.")

# Force a tensor computation to verify active execution on GPU memory
jnp.zeros(1).block_until_ready()
print("OK: JAX backend successfully executing on GPU.")'

probe colabfold python "Checking ColabFold JAX/GPU compatibility" -c "$COLABFOLD_PROBE" \
|| log_fatal "ColabFold image cannot access GPU hardware.
       JAX falls back to CPU silently without raising runtime errors; ensure driver and container toolkit are properly configured."

fi

log_info "All targeted image builds and capability probes verified successfully."