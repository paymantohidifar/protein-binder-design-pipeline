#!/usr/bin/env bash
# ==============================================================================
# Lambda Cloud GPU Host Bootstrap Script for Binder Design Pipeline
#
# Prepares a fresh GPU host by validating environment prerequisites, setting up
# Docker with GPU passthrough, creating shared weight caches, materializing 
# host Python environments, and prefetching model parameters.
#
# Usage:
#   bash scripts/bootstrap_instance.sh                 # Standard host setup (~2 min)
#   bash scripts/bootstrap_instance.sh --check         # Dry-run diagnostic check
#   bash scripts/bootstrap_instance.sh --weights-only  # Download RFdiffusion checkpoints only
#   bash scripts/bootstrap_instance.sh --images        # Build 5 tool images (~50 GB, 30-60 min)
#   bash scripts/bootstrap_instance.sh --prefetch      # Eagerly pull lazy model weights (~18 GB)
#   bash scripts/bootstrap_instance.sh --all           # Build images + prefetch all weights
# ==============================================================================

set -euo pipefail

# Navigate to project root relative to script location
cd "$(dirname "${BASH_SOURCE[0]}")/.."
readonly REPO_ROOT="$PWD"

# Global Configuration Parameters
readonly BINDER_CACHE="${BINDER_CACHE:-$HOME/.cache/binder-pipeline}"
readonly RFDIFFUSION_WEIGHTS_URL="http://files.ipd.uw.edu/pub/RFdiffusion"
readonly PIPELINE_IMAGES=(rfdiffusion proteinmpnn esmfold boltz2 colabfold)

# ------------------------------------------------------------------------------
# Command-Line Argument Parsing
# ------------------------------------------------------------------------------
MODE="full"         # Options: full | weights | check
DO_IMAGES=false    # Build tool Docker containers (~50 GB)
DO_PREFETCH=false  # Eagerly download lazy weights (~18 GB)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --weights-only) MODE="weights" ;;
        --check)        MODE="check" ;;
        --images)       DO_IMAGES=true ;;
        --prefetch)     DO_PREFETCH=true ;;
        --all)          DO_IMAGES=true; DO_PREFETCH=true ;;
        -h|--help)      sed -n '2,25p' "$0"; exit 0 ;;
        *)              printf 'Error: Unknown option %s\n' "$1" >&2; exit 2 ;;
    esac
    shift
done

# Guard: Mutating flags are incompatible with read-only check mode
if [[ "$MODE" == "check" ]] && { [[ "$DO_IMAGES" == true ]] || [[ "$DO_PREFETCH" == true ]]; }; then
    printf 'Error: --check cannot be combined with --images, --prefetch, or --all\n' >&2
    exit 2
fi

# ------------------------------------------------------------------------------
# Terminal Output Formatting Helpers
# ------------------------------------------------------------------------------
log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; } # Blue section header
ok()   { printf '   \033[1;32mok\033[0m  %s\n' "$*"; } # Green success badge
warn() { printf '   \033[1;33m--\033[0m  %s\n' "$*"; } # Yellow warning badge
bad()  { printf '   \033[1;31m!!\033[0m  %s\n' "$*"; } # Red error badge

FAILURES=0
note_fail() { bad "$*"; FAILURES=$((FAILURES + 1)); }

# ==============================================================================
# 1. GPU AND DRIVER VERIFICATION
# ==============================================================================
log "GPU and driver"

if command -v nvidia-smi >/dev/null 2>&1; then
    # Query installed GPU specs
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
        | while IFS= read -r line; do ok "$line"; done

    # Query compute capability architecture code
    CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -dc '0-9' || echo "")
    
    if [[ -n "$CAP" && "$CAP" -ge 90 ]]; then
        warn "Compute capability sm_${CAP}: Official RFdiffusion image (CUDA 11.6) lacks sm_90+ kernels."
        warn "Use an A10/A100 host, or rebuild RFdiffusion against PyTorch 2.x / CUDA 12 (see MANIFEST Section 6.1)."
    else
        ok "Compute capability sm_${CAP} is compatible with RFdiffusion base images."
    fi

    # Query total GPU VRAM (MiB)
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -dc '0-9' || echo 0)
    
    if [[ -z "$VRAM" || "$VRAM" -eq 0 ]]; then
        warn "Could not read VRAM from nvidia-smi."
    elif [[ "$VRAM" -lt 20000 ]]; then
        note_fail "${VRAM} MiB VRAM is below the 24 GB pipeline floor (ESMFold requires ~16 GB minimum)."
    elif [[ "$VRAM" -lt 40000 ]]; then
        warn "${VRAM} MiB VRAM: Workable, but Step 4 complex predictions on large targets may hit OOM."
    else
        ok "${VRAM} MiB VRAM available."
    fi
else
    note_fail "nvidia-smi not found—this host lacks an active NVIDIA GPU driver."
fi

# ==============================================================================
# 2. DISK SPACE CHECKS
# ==============================================================================
log "Disk space"

for mount in / "$HOME"; do
    avail=$(df -BG --output=avail "$mount" 2>/dev/null | tail -n 1 | tr -dc '0-9' || echo 0)
    
    if [[ "${avail:-0}" -lt 250 ]]; then
        warn "$mount has ${avail}G free; ~250G recommended (Images ~50G, Weights ~30G, Workspace Outputs)."
    else
        ok "$mount has ${avail}G free."
    fi
done

# ==============================================================================
# 3. DOCKER & NVIDIA CONTAINER TOOLKIT SETUP
# ==============================================================================
log "Docker and NVIDIA container toolkit"

# If the current process lacks active docker group privileges, add user and re-exec.
if ! groups | grep -qw docker; then
    if ! id -nG "$USER" | grep -qw docker; then
        log "Adding $USER to the docker group..."
        sudo usermod -aG docker "$USER"
    fi
fi

if ! command -v docker >/dev/null 2>&1; then
    if [[ "$MODE" == "check" ]]; then
        note_fail "Docker is not installed."
    else
        log "Installing Docker"
        curl -fsSL https://get.docker.com | sudo sh
        sudo usermod -aG docker "$USER"
        warn "Added $USER to the docker group. Log out and back in, then re-run this script."
    fi
else
    ok "Docker $(docker --version | awk '{print $3}' | tr -d ,)"
fi

# Test container GPU passthrough capabilities
if command -v docker >/dev/null 2>&1; then
    if docker run --rm --gpus all ubuntu:22.04 true 2>/dev/null; then
        ok "GPU passthrough functional."
    elif [[ "$MODE" == "check" ]]; then
        note_fail "Docker GPU passthrough failing (NVIDIA Container Toolkit missing or unconfigured)."
    else
        log "Installing nvidia-container-toolkit"
        distribution=$(. /etc/os-release; echo "$ID$VERSION_ID")
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
            | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -fsSL "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" \
            | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
            | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
        
        docker run --rm --gpus all ubuntu:22.04 true \
            && ok "GPU passthrough functional." \
            || note_fail "GPU passthrough still failing post-installation."
    fi
fi

# ==============================================================================
# 4. SHARED MODEL WEIGHT CACHE DIRECTORY
# ==============================================================================
log "Shared weight cache"

if [[ "$MODE" != "check" ]]; then
    mkdir -p "$BINDER_CACHE"/{rfdiffusion/models,hf,torch,boltz,colabfold}
fi

if [[ -d "$BINDER_CACHE" ]]; then
    ok "$BINDER_CACHE ($(du -sh "$BINDER_CACHE" 2>/dev/null | cut -f1))"
else
    note_fail "$BINDER_CACHE directory missing."
fi

# ==============================================================================
# 5. HOST PYTHON ENVIRONMENT (PIXI)
# ==============================================================================
log "Host Python environment (Pixi)"

PIXI_ENV="${PIXI_ENV:-default}"
export PATH="$HOME/.pixi/bin:$PATH"
PIXI_READY=false

# --- 5a. Pixi Executable Verification ---
if command -v pixi >/dev/null 2>&1; then
    ok "Pixi $(pixi --version | awk '{print $2}')"
    PIXI_READY=true
elif [[ "$MODE" == "check" ]]; then
    warn "Pixi is not installed (full bootstrap mode will install it)."
else
    log "Installing Pixi package manager"
    if curl -fsSL https://pixi.sh/install.sh | bash; then
        export PATH="$HOME/.pixi/bin:$PATH"
        hash -r
        if command -v pixi >/dev/null 2>&1; then
            ok "Pixi $(pixi --version | awk '{print $2}') installed."
            PIXI_READY=true
        else
            warn "Pixi installed but executable not found on PATH ($HOME/.pixi/bin)."
        fi
    else
        warn "Pixi installation failed."
    fi
fi

# --- 5b. Locked Environment Materialization ---
if [[ "$PIXI_READY" == true ]]; then
    if [[ "$MODE" == "check" ]]; then
        if [[ -d ".pixi/envs/$PIXI_ENV" ]]; then
            ok "Pixi environment '$PIXI_ENV' present."
        else
            warn "Pixi environment '$PIXI_ENV' not materialized."
            PIXI_READY=false
        fi
    else
        if pixi install --environment "$PIXI_ENV"; then
            ok "Pixi environment '$PIXI_ENV' materialized from pixi.lock."
        else
            note_fail "Pixi install failed for environment '$PIXI_ENV'."
            PIXI_READY=false
        fi
    fi
fi

# --- 5c. Pip Fallback ---
if [[ "$PIXI_READY" != true && "$MODE" == "full" ]]; then
    warn "Pixi unavailable—falling back to pip + requirements.txt (non-version-locked)."
    python3 -m pip install --user --quiet -r requirements.txt \
        && ok "requirements.txt installed." \
        || note_fail "pip install failed."
fi

# --- 5d. Interpreter Selector Wrapper ---
host_python() {
    if [[ "$PIXI_READY" == true ]]; then
        pixi run --environment "$PIXI_ENV" python "$@"
    else
        python3 "$@"
    fi
}

# --- 5e. Dependency Verification Probe ---
host_python - <<'PY' || note_fail "Host Python dependencies incomplete."
import importlib.util
import sys

REQUIRED = ("numpy", "Bio", "yaml", "pandas", "py3Dmol", "pytest")
missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]

if missing:
    print(f"   !!  Missing dependencies: {missing}")
    sys.exit(1)

import numpy, pandas, Bio
print(f"   ok  Python v{sys.version.split()[0]} | NumPy v{numpy.__version__} | "
      f"Pandas v{pandas.__version__} | BioPython v{Bio.__version__}")
PY

# ==============================================================================
# 6. RFDIFFUSION WEIGHTS DOWNLOAD
# ==============================================================================
log "RFdiffusion model weights"

if [[ "$MODE" != "check" ]]; then
    for spec in \
        "6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt" \
        "e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt"
    do
        target="$BINDER_CACHE/rfdiffusion/models/$(basename "$spec")"
        if [[ -s "$target" ]]; then
            ok "$(basename "$spec") present."
        else
            wget -q --show-progress -O "$target" "$RFDIFFUSION_WEIGHTS_URL/$spec" \
                || { rm -f "$target"; note_fail "Failed to download $(basename "$spec")"; }
        fi
    done
else
    for f in Base_ckpt.pt Complex_base_ckpt.pt; do
        [[ -s "$BINDER_CACHE/rfdiffusion/models/$f" ]] && ok "$f present." || warn "$f missing."
    done
fi

# ==============================================================================
# 7. CONTAINER IMAGES
# ==============================================================================
log "Container images"

if [[ "$DO_IMAGES" == true ]]; then
    if ! command -v docker >/dev/null 2>&1; then
        note_fail "--images requested but Docker is not installed."
    else
        log "Building container images (estimated duration: 30-60 minutes)"
        if bash docker/build_all.sh; then
            ok "All container images built successfully."
        else
            note_fail "Image build pipeline failed."
        fi
    fi
fi

if command -v docker >/dev/null 2>&1; then
    for name in "${PIPELINE_IMAGES[@]}"; do
        if docker image inspect "binder-$name:latest" >/dev/null 2>&1; then
            size=$(docker image inspect "binder-$name:latest" \
                --format '{{.Size}}' 2>/dev/null | awk '{printf "%.1fG", $1/1e9}' || echo "?")
            ok "binder-$name:latest ($size)"
        else
            warn "binder-$name:latest missing (build via: bash docker/build_all.sh $name)"
        fi
    done
fi

# ==============================================================================
# 7b. MODEL WEIGHT PREFETCH
# ==============================================================================
if [[ "$DO_PREFETCH" == true ]]; then
    log "Prefetching model weights"

    if ! command -v docker >/dev/null 2>&1; then
        note_fail "--prefetch requested but Docker is not installed."
    else
        prefetch_run() {
            local image="$1"; shift
            docker run --rm --gpus all \
                --user "$(id -u):$(id -g)" \
                -v "$BINDER_CACHE:/cache" \
                -e HF_HOME=/cache/hf -e TORCH_HOME=/cache/torch \
                --entrypoint "$1" "$image" "${@:2}"
        }

        # --- ProteinMPNN ---
        if docker image inspect binder-proteinmpnn:latest >/dev/null 2>&1; then
            ok "ProteinMPNN weights baked into image."
        fi

        # --- ESMFold (~10 GB) ---
        if docker image inspect binder-esmfold:latest >/dev/null 2>&1; then
            log "ESMFold weights (~10 GB)"
            if prefetch_run binder-esmfold:latest python -c \
                'import esm; esm.pretrained.esmfold_v1(); print("ESMFold weights cached")'; then
                ok "ESMFold weights cached."
            else
                warn "ESMFold prefetch failed—will download on first execution."
            fi
        else
            warn "Skipping ESMFold prefetch: Image binder-esmfold:latest not built."
        fi

        # --- ColabFold (~5 GB) ---
        if docker image inspect binder-colabfold:latest >/dev/null 2>&1; then
            log "ColabFold AF2 multimer_v3 parameters (~5 GB)"
            if prefetch_run binder-colabfold:latest \
                python -m colabfold.download alphafold2_multimer_v3; then
                ok "ColabFold parameters cached."
            else
                warn "ColabFold prefetch failed—will download on first execution."
            fi
        else
            warn "Skipping ColabFold prefetch: Image binder-colabfold:latest not built."
        fi

        # --- Boltz-2 (~6.2 GB) ---
        if docker image inspect binder-boltz2:latest >/dev/null 2>&1; then
            log "Boltz-2 weights + CCD (~6.2 GB)"
            if prefetch_run binder-boltz2:latest python -c \
                'from pathlib import Path; from boltz.main import download_boltz2; download_boltz2(Path("/cache/boltz")); print("Boltz-2 weights cached")'; then
                ok "Boltz-2 weights cached."
            else
                warn "Boltz-2 prefetch failed—will download on first prediction."
            fi
        else
            warn "Skipping Boltz-2 prefetch: Image binder-boltz2:latest not built."
        fi
    fi
fi

# ==============================================================================
# 8. HOST TEST SUITE
# ==============================================================================
log "Host test suite"

if [[ "$MODE" == "full" ]]; then
    host_python -m pytest tests/ -q 2>&1 | tail -n 3
fi

# ==============================================================================
# 9. FINAL SUMMARY & VERDICT
# ==============================================================================
log "Summary"

if [[ "$FAILURES" -eq 0 ]]; then
    printf '   \033[1;32mGO\033[0m -- Host environment ready.\n'
    
    MISSING_IMAGES=0
    if command -v docker >/dev/null 2>&1; then
        for name in "${PIPELINE_IMAGES[@]}"; do
            docker image inspect "binder-$name:latest" >/dev/null 2>&1 || \
                MISSING_IMAGES=$((MISSING_IMAGES + 1))
        done
    fi

    if [[ "$MISSING_IMAGES" -gt 0 ]]; then
        echo "   $MISSING_IMAGES of ${#PIPELINE_IMAGES[@]} images unbuilt."
        if [[ "$PIXI_READY" == true ]]; then
            echo "   Next step: pixi run bootstrap-all"
        else
            echo "   Next step: bash scripts/bootstrap_instance.sh --all"
        fi
    elif [[ "$DO_PREFETCH" != true ]]; then
        echo "   Container images ready. First execution will fetch ~18 GB of weights."
        echo "   Eager prefetch command: bash scripts/bootstrap_instance.sh --prefetch"
    else
        echo "   All pipeline images and weights verified and ready."
    fi

    if [[ "$PIXI_READY" == true ]]; then
        echo "   Next step: pixi run register-kernel # (registers Jupyter kernel for Pixi environment)"
        echo "   Environment validation: pixi run verify"
    else
        echo "   Interactive notebook command: jupyter lab notebooks/01_binder_design_prototype.ipynb"
    fi
else
    printf '   \033[1;31mNO-GO\033[0m -- Encountered %d blocking issue(s) during verification.\n' "$FAILURES"
    exit 1
fi