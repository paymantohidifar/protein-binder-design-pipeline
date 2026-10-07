#!/usr/bin/env bash
# Build the five pipeline images.
#
# Must run ON THE GPU HOST, not on a workstation: the images total ~50 GB and
# RFdiffusion's compile step wants the CUDA toolkit present.
#
# Usage:
#   bash docker/build_all.sh              # build all five
#   bash docker/build_all.sh esmfold      # build one
#
# Safe to re-run; Docker layer cache makes unchanged images near-instant.
set -euo pipefail

cd "$(dirname "$0")/.."
IMAGES=(rfdiffusion proteinmpnn esmfold boltz2 colabfold)
TARGETS=("${@:-${IMAGES[@]}}")

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31m!! %s\033[0m\n' "$*" >&2; exit 1; }

command -v docker >/dev/null || fail "docker not found"

# Disk is the usual failure: ~50 GB of images plus build layers.
avail_gb=$(df -BG --output=avail /var/lib/docker 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
if [[ "${avail_gb:-0}" -lt 80 ]]; then
    printf '\033[1;33m-- warning: only %sG free where docker stores images; ~80G recommended\033[0m\n' "$avail_gb"
fi

built=()
for name in "${TARGETS[@]}"; do
    [[ -f "docker/$name/Dockerfile" ]] || fail "unknown image: $name"
    log "Building binder-$name:latest"
    docker build --progress=plain -t "binder-$name:latest" "docker/$name"
    built+=("binder-$name:latest")
done

log "Built: ${built[*]}"

# RFdiffusion's CUDA 11.6 pin emits no sm_90 kernels, so an H100 host fails at
# the first tensor copy -- minutes into a design run, with an opaque message.
# Surface it here instead.
if [[ " ${TARGETS[*]} " == *" rfdiffusion "* ]] && command -v nvidia-smi >/dev/null; then
    log "Checking RFdiffusion CUDA/GPU compatibility"
    if docker run --rm --gpus all binder-rfdiffusion:latest \
        -c 'import torch; cap=torch.cuda.get_device_capability(); torch.zeros(1).cuda(); \
            print(f"OK {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} torch={torch.__version__}")' \
        2>/dev/null; then
        :
    else
        docker run --rm --gpus all --entrypoint python binder-rfdiffusion:latest -c \
            'import torch; cap=torch.cuda.get_device_capability(); torch.zeros(1).cuda(); \
             print(f"OK {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} torch={torch.__version__}")' \
        || fail "RFdiffusion image cannot use this GPU. If this is an H100/B200 (sm_90+),
       the upstream CUDA 11.6 pin has no kernels for it -- see MANIFEST Section 6.1.
       Use an A10/A100 host, or rebuild this image against torch 2.x/cu12."
    fi
fi

log "Done. Next: bash scripts/bootstrap_instance.sh --weights-only"
