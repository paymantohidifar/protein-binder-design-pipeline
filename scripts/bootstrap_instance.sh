#!/usr/bin/env bash
# Prepare a fresh Lambda Cloud GPU instance for the binder design pipeline.
#
# This pipeline runs every tool in its own container, so the host needs only 
# Docker, the NVIDIA container toolkit, the host-side Python deps, and the 
# shared weight cache.
#
# Usage:
#   bash scripts/bootstrap_instance.sh                 # host setup, ~2 min
#   bash scripts/bootstrap_instance.sh --check         # verify, change nothing
#   bash scripts/bootstrap_instance.sh --weights-only  # RFdiffusion ckpts only
#   bash scripts/bootstrap_instance.sh --images        # + build 5 images (~50 GB)
#   bash scripts/bootstrap_instance.sh --prefetch      # + pull lazy weights
#   bash scripts/bootstrap_instance.sh --all           # everything
#
# The default stays deliberately cheap so it can be re-run to re-diagnose a
# half-configured host. --images and --prefetch are the hour-long parts, kept
# behind flags for that reason; they can be combined.
#
# Host Python is managed by pixi (pixi.toml + pixi.lock); this script installs
# pixi if absent. Set PIXI_ENV=ci for a headless host that needs no JupyterLab:
#   PIXI_ENV=ci bash scripts/bootstrap_instance.sh
#
# Exit immediately if a command fails (-e), if an uninitialized variable is used (-u),
# or if any command in a pipeline fails (-o pipefail).
set -euo pipefail

# Change directory to the parent directory of this script (ensures relative paths work).
cd "$(dirname "$0")/.."

# Store the absolute path of the repository root directory.
REPO_ROOT="$PWD"

# Set default cache directory for model weights (~/.cache/binder-pipeline if unset).
BINDER_CACHE="${BINDER_CACHE:-$HOME/.cache/binder-pipeline}"

# Base URL for downloading RFdiffusion pre-trained weights from IPD.
RFDIFFUSION_WEIGHTS_URL="http://files.ipd.uw.edu/pub/RFdiffusion"

# ------------------------------------------------------------------------------
# COMMAND-LINE OPTION PARSING
# ------------------------------------------------------------------------------
MODE=full          # full | weights | check -- what the core sections do
DO_IMAGES=false    # build the five tool containers (~50 GB, 30-60 min)
DO_PREFETCH=false  # pull the weights that otherwise download on first run

# A loop rather than a case on $1, so flags combine: --images --prefetch.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --weights-only) MODE=weights ;;                  # RFdiffusion ckpts only
        --check)        MODE=check ;;                    # verify, change nothing
        --images)       DO_IMAGES=true ;;                # also build images
        --prefetch)     DO_PREFETCH=true ;;              # also pull lazy weights
        --all)          DO_IMAGES=true; DO_PREFETCH=true ;;
        -h|--help)      sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

# Building and prefetching both mutate the host, so they are incompatible with
# a read-only check. Fail loudly rather than silently ignoring the flag.
if [[ "$MODE" == check ]] && { [[ "$DO_IMAGES" == true ]] || [[ "$DO_PREFETCH" == true ]]; }; then
    echo "--check cannot be combined with --images/--prefetch/--all" >&2
    exit 2
fi

# ------------------------------------------------------------------------------
# TERMINAL OUTPUT FORMATTING HELPERS (ANSI CODES)
# ------------------------------------------------------------------------------
log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; } # Blue header line
ok()   { printf '   \033[1;32mok\033[0m  %s\n' "$*"; } # Green "ok" badge
warn() { printf '   \033[1;33m--\033[0m  %s\n' "$*"; } # Yellow warning badge
bad()  { printf '   \033[1;31m!!\033[0m  %s\n' "$*"; } # Red failure badge

# Track total blocking failures encountered during execution
FAILURES=0
note_fail() { bad "$*"; FAILURES=$((FAILURES + 1)); }

# ==============================================================================
# 1. GPU AND DRIVER VERIFICATION
# ==============================================================================
log "GPU and driver"

# Check if the nvidia-smi tool is available on the system
if command -v nvidia-smi >/dev/null; then
    # Print installed GPU name, total memory, and driver version
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
        | while IFS= read -r line; do ok "$line"; done

    # Query compute capability (e.g., "8.9" or "9.0") and strip non-digit characters
    # Same guard: an nvidia-smi that exists but errors must not kill the run.
    CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -dc '0-9' || echo "")
    
    # Hopper GPUs (sm_90+, such as H100) lack compiled CUDA 11.6 kernels in the stock image
    if [[ -n "$CAP" && "$CAP" -ge 90 ]]; then
        warn "compute capability sm_${CAP}: the official RFdiffusion image (CUDA 11.6)"
        warn "has no kernels for this GPU. Use A10/A100, or rebuild that image"
        warn "against torch 2.x/cu12. See MANIFEST Section 6.1."
    else
        ok "compute capability sm_${CAP} is compatible with the RFdiffusion image"
    fi

    # Query total GPU VRAM in MiB
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9' || echo 0)
    
    # Verify VRAM meets the minimum requirements for structure prediction models
    if [[ -z "$VRAM" ]]; then
        warn "could not read VRAM from nvidia-smi"
    elif [[ "$VRAM" -lt 20000 ]]; then
        note_fail "${VRAM} MiB VRAM is below the 24 GB floor (ESMFold needs ~16 GB)"
    elif [[ "$VRAM" -lt 40000 ]]; then
        warn "${VRAM} MiB VRAM: workable, but Step 4 on a large target may OOM"
    else
        ok "${VRAM} MiB VRAM"
    fi
else
    note_fail "nvidia-smi not found -- this is not a GPU host"
fi

# ==============================================================================
# 2. DISK SPACE CHECKS
# ==============================================================================
log "Disk"

# Iterate over root drive and user home drive to verify sufficient storage space
for mount in / "$HOME"; do
    # Read available disk space in Gigabytes
    # `|| echo 0` is load-bearing: set -e plus pipefail means a failing df
    # (missing mount, unsupported flag) would otherwise abort the script. A
    # diagnostic must report and continue, never die halfway through.
    avail=$(df -BG --output=avail "$mount" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
    
    # Ensure at least 250 GB is free for container images, weights, and run outputs
    if [[ "${avail:-0}" -lt 250 ]]; then
        warn "$mount has ${avail}G free; ~250G needed (images ~50G, weights ~30G, outputs)"
    else
        ok "$mount has ${avail}G free"
    fi
done

# ==============================================================================
# 3. DOCKER & NVIDIA CONTAINER TOOLKIT SETUP
# ==============================================================================
log "Docker and NVIDIA container toolkit"

# Verify whether Docker is installed
if ! command -v docker >/dev/null; then
    if [[ "$MODE" == check ]]; then
        note_fail "docker not installed"
    else
        # Install Docker if missing and in full setup mode
        log "Installing Docker"
        curl -fsSL https://get.docker.com | sudo sh
        sudo usermod -aG docker "$USER"
        warn "added $USER to the docker group -- log out and back in, then re-run"
    fi
else
    ok "docker $(docker --version | awk '{print $3}' | tr -d ,)"
fi

# If Docker exists, test GPU access inside containers
if command -v docker >/dev/null; then
    # Test if containers can communicate with GPU
    if docker run --rm --gpus all ubuntu:22.04 true 2>/dev/null; then
        ok "GPU passthrough works"
    elif [[ "$MODE" == check ]]; then
        note_fail "docker cannot see the GPU (nvidia-container-toolkit missing?)"
    else
        # Install and configure NVIDIA Container Toolkit if GPU access fails
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
        
        # Re-test GPU container passthrough
        docker run --rm --gpus all ubuntu:22.04 true \
            && ok "GPU passthrough works" \
            || note_fail "GPU passthrough still failing after install"
    fi
fi

# ==============================================================================
# 4. SHARED MODEL WEIGHT CACHE DIRECTORY
# ==============================================================================
log "Shared weight cache"

# Create weight subdirectories inside the designated cache location
if [[ "$MODE" != check ]]; then
    mkdir -p "$BINDER_CACHE"/{rfdiffusion/models,hf,torch,boltz,colabfold}
fi

# Display current size of the weight cache directory
if [[ -d "$BINDER_CACHE" ]]; then
    ok "$BINDER_CACHE ($(du -sh "$BINDER_CACHE" 2>/dev/null | cut -f1))"
else
    note_fail "$BINDER_CACHE missing"
fi

# ==============================================================================
# 5. HOST PYTHON ENVIRONMENT (PIXI)
# ==============================================================================
log "Host Python environment (pixi)"

# Which pixi environment to materialise. `default` adds JupyterLab, needed to
# serve the prototype notebook over an SSH tunnel. Override with PIXI_ENV=ci on
# a headless host that only runs the test suite (~400 MB lighter).
PIXI_ENV="${PIXI_ENV:-default}"

# pixi installs itself to ~/.pixi/bin. Its installer appends that to the shell
# rc file, which does nothing for this already-running non-interactive process,
# so prepend it before probing.
export PATH="$HOME/.pixi/bin:$PATH"

# True once host Python is driven by the locked pixi environment. Sections 8
# onward branch on this to pick an interpreter.
PIXI_READY=false

# ------------------------------------------------------------------------------
# 5a. pixi itself
# ------------------------------------------------------------------------------
if command -v pixi >/dev/null; then
    ok "pixi $(pixi --version | awk '{print $2}')"
    PIXI_READY=true
elif [[ "$MODE" == check ]]; then
    warn "pixi not installed (full mode would install it)"
else
    log "Installing pixi"
    # Official installer: unpacks to ~/.pixi, needs no sudo, touches no system
    # python. Nothing else in this script requires root for host Python.
    if curl -fsSL https://pixi.sh/install.sh | bash; then
        export PATH="$HOME/.pixi/bin:$PATH"
        hash -r   # drop the shell's cached "pixi: not found" lookup
        if command -v pixi >/dev/null; then
            ok "pixi $(pixi --version | awk '{print $2}') installed"
            PIXI_READY=true
        else
            warn "pixi installed but not found on PATH under $HOME/.pixi/bin"
        fi
    else
        warn "pixi installer failed (no egress to pixi.sh?)"
    fi
fi

# ------------------------------------------------------------------------------
# 5b. The locked environment
# ------------------------------------------------------------------------------
if [[ "$PIXI_READY" == true ]]; then
    if [[ "$MODE" == check ]]; then
        # Confirm the environment is materialised without creating it.
        if [[ -d ".pixi/envs/$PIXI_ENV" ]]; then
            ok "pixi environment '$PIXI_ENV' present"
        else
            warn "pixi environment '$PIXI_ENV' not installed yet"
            PIXI_READY=false
        fi
    else
        # Installs from pixi.lock, so this reproduces exact builds rather than
        # re-solving. python/numpy/pandas versions match what the test suite was
        # verified against.
        if pixi install --environment "$PIXI_ENV"; then
            ok "pixi environment '$PIXI_ENV' installed from pixi.lock"
        else
            note_fail "pixi install failed for environment '$PIXI_ENV'"
            PIXI_READY=false
        fi
    fi
fi

# ------------------------------------------------------------------------------
# 5c. pip fallback
# ------------------------------------------------------------------------------
# requirements.txt exists for exactly this case: a host where pixi could not be
# installed. Bounds there are loose, so it adapts to the system interpreter --
# but it is NOT reproducible, and a campaign run this way is not guaranteed to
# match one run under pixi.lock.
if [[ "$PIXI_READY" != true && "$MODE" == full ]]; then
    warn "using pip + requirements.txt -- not version-locked"
    python3 -m pip install --user --quiet -r requirements.txt \
        && ok "requirements.txt installed" \
        || note_fail "pip install failed"
fi

# ------------------------------------------------------------------------------
# 5d. Interpreter selector used by later sections
# ------------------------------------------------------------------------------
# Runs host Python inside the pixi environment when available, otherwise against
# the system interpreter. `pixi run` forwards stdin, so heredocs work either way.
host_python() {
    if [[ "$PIXI_READY" == true ]]; then
        pixi run --environment "$PIXI_ENV" python "$@"
    else
        python3 "$@"
    fi
}

# ------------------------------------------------------------------------------
# 5e. Verify the imports the pipeline actually needs
# ------------------------------------------------------------------------------
# Checks the real environment rather than trusting the installer's exit code,
# and prints resolved versions so a mismatch against pixi.lock is visible here
# instead of surfacing mid-campaign.
host_python - <<'PY' || note_fail "host python deps incomplete"
import importlib.util
import sys

# Mirrors the dependency list in pixi.toml. Import names, not package names:
# biopython imports as Bio, pyyaml as yaml, py3dmol as py3Dmol.
REQUIRED = ("numpy", "Bio", "yaml", "pandas", "py3Dmol", "pytest")

missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
if missing:
    print(f"   !!  missing: {missing}")
    sys.exit(1)

import numpy, pandas, Bio
print(f"   ok  python {sys.version.split()[0]}, numpy {numpy.__version__}, "
      f"pandas {pandas.__version__}, biopython {Bio.__version__}")
PY

# ==============================================================================
# 6. RFDIFFUSION WEIGHTS DOWNLOAD
# ==============================================================================
log "RFdiffusion weights"

if [[ "$MODE" != check ]]; then
    # Fetch required model checkpoints if not already present on disk
    for spec in \
        "6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt" \
        "e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt"
    do
        target="$BINDER_CACHE/rfdiffusion/models/$(basename "$spec")"
        if [[ -s "$target" ]]; then
            ok "$(basename "$spec") already present"
        else
            wget -q --show-progress -O "$target" "$RFDIFFUSION_WEIGHTS_URL/$spec" \
                || { rm -f "$target"; note_fail "failed to download $(basename "$spec")"; }
        fi
    done
else
    # In check mode, simply confirm whether weight files exist
    for f in Base_ckpt.pt Complex_base_ckpt.pt; do
        [[ -s "$BINDER_CACHE/rfdiffusion/models/$f" ]] && ok "$f" || warn "$f not downloaded"
    done
fi

# ==============================================================================
# 7. CONTAINER IMAGES
# ==============================================================================
log "Container images"

# All five images, in the order build_all.sh builds them.
PIPELINE_IMAGES=(rfdiffusion proteinmpnn esmfold boltz2 colabfold)

# Build only when asked: ~50 GB and 30-60 min. docker/build_all.sh owns the
# build logic (including the sm_90 compatibility probe for RFdiffusion), so this
# delegates rather than duplicating it.
if [[ "$DO_IMAGES" == true ]]; then
    if ! command -v docker >/dev/null; then
        note_fail "--images requested but docker is not installed"
    else
        log "Building images (this takes 30-60 minutes)"
        if bash docker/build_all.sh; then
            ok "all images built"
        else
            note_fail "image build failed -- see output above"
        fi
    fi
fi

# Verify, whether or not we just built. A missing image is a warning rather than
# a failure: the host is still usable for everything up to that stage.
if command -v docker >/dev/null; then
    for name in "${PIPELINE_IMAGES[@]}"; do
        if docker image inspect "binder-$name:latest" >/dev/null 2>&1; then
            # Report size, so a truncated or failed build is visible here.
            size=$(docker image inspect "binder-$name:latest" \
                --format '{{.Size}}' 2>/dev/null | awk '{printf "%.1fG", $1/1e9}' || echo "?")
            ok "binder-$name:latest ($size)"
        else
            warn "binder-$name:latest not built  (bash docker/build_all.sh $name)"
        fi
    done
fi

# ==============================================================================
# 7b. MODEL WEIGHT PREFETCH
# ==============================================================================
# RFdiffusion aside (section 6), each tool downloads its own weights the first
# time it runs. That is ~18 GB paid during the first design run -- on a
# GPU-billed instance you are paying GPU-hours to download, and a failure
# surfaces as an opaque in-container error rather than a bootstrap NO-GO.
# Prefetching moves that cost here, where it is diagnosable.
#
# Each tool is its own downloader, so this needs the images to exist already.
if [[ "$DO_PREFETCH" == true ]]; then
    log "Prefetching model weights"

    if ! command -v docker >/dev/null; then
        note_fail "--prefetch requested but docker is not installed"
    else
        # Shared cache mounted exactly as pipeline.docker mounts it, so the
        # files land where the pipeline will look for them.
        prefetch_run() {
            local image="$1"; shift
            docker run --rm --gpus all \
                -v "$BINDER_CACHE:/cache" \
                -e HF_HOME=/cache/hf -e TORCH_HOME=/cache/torch \
                --entrypoint "$1" "$image" "${@:2}"
        }

        # --- ProteinMPNN: nothing to do, weights ship inside the image -------
        if docker image inspect binder-proteinmpnn:latest >/dev/null 2>&1; then
            ok "proteinmpnn weights are baked into the image"
        fi

        # --- ESMFold: ~10 GB (esm2_t36_3B_UR50D + folding trunk) -------------
        if docker image inspect binder-esmfold:latest >/dev/null 2>&1; then
            log "ESMFold weights (~10 GB)"
            if prefetch_run binder-esmfold:latest python -c \
                'import esm; esm.pretrained.esmfold_v1(); print("esmfold weights cached")'; then
                ok "ESMFold weights cached"
            else
                warn "ESMFold prefetch failed -- will download on first run instead"
            fi
        else
            warn "skipping ESMFold prefetch: image not built"
        fi

        # --- ColabFold: ~5 GB of AF2 params ---------------------------------
        if docker image inspect binder-colabfold:latest >/dev/null 2>&1; then
            log "ColabFold AF2 params (~5 GB)"
            if prefetch_run binder-colabfold:latest python -m colabfold.download; then
                ok "ColabFold params cached"
            else
                warn "ColabFold prefetch failed -- will download on first run instead"
            fi
        else
            warn "skipping ColabFold prefetch: image not built"
        fi

        # --- Boltz-2: NOT prefetched, deliberately --------------------------
        # Boltz downloads its weights on first `boltz predict`, and exposes no
        # download-only subcommand I could verify. The alternatives were to run
        # a real throwaway prediction (needs the GPU, several minutes, and is
        # not obviously idempotent) or to guess at a flag. Neither is worth it
        # for ~3 GB, so this is left lazy and stated rather than faked.
        warn "Boltz-2 (~3 GB) downloads on its first prediction -- no download-only entry point"
    fi
fi

# ==============================================================================
# 8. HOST TEST SUITE
# ==============================================================================
log "Host test suite"

# Run the project unit tests in full setup mode, inside whichever environment
# section 5 settled on. 166 tests, no GPU required -- a failure here means the
# host environment is wrong, before any GPU time is spent.
if [[ "$MODE" == full ]]; then
    host_python -m pytest tests/ -q 2>&1 | tail -3
fi

# ==============================================================================
# 9. FINAL SUMMARY & VERDICT
# ==============================================================================
log "Summary"

# Output final status badge based on recorded errors
if [[ "$FAILURES" -eq 0 ]]; then
    printf '   \033[1;32mGO\033[0m -- host ready.\n'
    # Point at whatever is still outstanding, rather than a fixed next step.
    MISSING_IMAGES=0
    if command -v docker >/dev/null; then
        for name in "${PIPELINE_IMAGES[@]}"; do
            docker image inspect "binder-$name:latest" >/dev/null 2>&1 || \
                MISSING_IMAGES=$((MISSING_IMAGES + 1))
        done
    fi

    if [[ "$MISSING_IMAGES" -gt 0 ]]; then
        echo "   $MISSING_IMAGES of ${#PIPELINE_IMAGES[@]} images still unbuilt."
        if [[ "$PIXI_READY" == true ]]; then
            echo "   Next: pixi run bootstrap-all   (build images + prefetch weights)"
        else
            echo "   Next: bash scripts/bootstrap_instance.sh --all"
        fi
    elif [[ "$DO_PREFETCH" != true ]]; then
        echo "   Images ready. First run will download ~18 GB of weights;"
        echo "   avoid that with: bash scripts/bootstrap_instance.sh --prefetch"
    else
        echo "   Images and weights ready."
    fi

    if [[ "$PIXI_READY" == true ]]; then
        echo "   Notebook: pixi run lab   (add --no-browser, SSH-tunnel 8888)"
        echo "   Verify anytime: pixi run verify"
    else
        echo "   Notebook: jupyter lab notebooks/01_binder_design_prototype.ipynb"
    fi
else
    printf '   \033[1;31mNO-GO\033[0m -- %d blocking problem(s) above.\n' "$FAILURES"
    exit 1
fi