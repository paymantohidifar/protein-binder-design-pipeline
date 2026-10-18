# ==============================================================================
# PIPELINE COMPONENT: Boltz-2
# ==============================================================================
# Role: All-atom precision gate and complex structure/confidence prediction engine.
# Upstream Reference: https://github.com/jwohlwend/boltz
#
# TECHNICAL NOTES:
# 1. Base Image: Built on PyTorch 2.6.0 / CUDA 12.6.
# 2. CUDA Linking Stub: Creates an explicit linker symlink (/opt/cuda-link/libcuda.so)
#    to satisfy Triton's JIT compiler (-lcuda) when running inside unprivileged
#    containers without full driver development headers.
# 3. Cache Storage Isolation: Redirects all model weight downloads, Numba JIT caches,
#    and Triton/cuEquivariance caches to the external /cache mount point to keep root
#    partitions clean and persist assets across container runs.
# ==============================================================================

# ==============================================================================
# BASE IMAGE & GPU RUNTIME
# ==============================================================================
# Pulls PyTorch's official runtime image (PyTorch 2.6.0 + CUDA 12.6 + cuDNN 9)
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# ==============================================================================
# BUILD ARGUMENTS & ENVIRONMENT VARIABLES
# ==============================================================================
# Pin Boltz package version for exact campaign reproducibility
ARG BOLTZ_VERSION=2.2.1

# Pin cuEquivariance version (unbounded as >=0.5.0 in upstream boltz[cuda])
ARG CUEQUIVARIANCE_VERSION=0.10.0

# PYTHONDONTWRITEBYTECODE=1     : Prevents writing temporary .pyc bytecode files to disk
# PYTHONUNBUFFERED=1            : Streams stdout/stderr directly to terminal logs without buffering
# BOLTZ_CACHE / NUMBA_CACHE_DIR : Redirects model weights and JIT outputs onto host-mounted storage
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BOLTZ_CACHE=/cache/boltz \
    NUMBA_CACHE_DIR=/cache/numba \
    XDG_CACHE_HOME=/cache \
    CUEQ_TRITON_CACHE_DIR=/cache/cuequivariance-triton \
    TRITON_CACHE_DIR=/cache/triton

# ==============================================================================
# SYSTEM PACKAGES SETUP
# ==============================================================================
# Update package list quietly and install build essentials for C++ extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# LINKER STUB SETUP FOR TRITON JIT
# ==============================================================================
# Creates a fallback libcuda.so symlink in /opt/cuda-link so Triton's JIT compilation
# (-lcuda) resolves cleanly whether running on a full devel host or a runtime driver mount.
RUN mkdir -p /opt/cuda-link \
    && if [ -e /usr/local/cuda/lib64/stubs/libcuda.so ]; then \
           ln -s /usr/local/cuda/lib64/stubs/libcuda.so /opt/cuda-link/libcuda.so; \
           echo "cuda-link: Using CUDA toolkit stub"; \
       else \
           ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /opt/cuda-link/libcuda.so; \
           echo "cuda-link: Symlinked to runtime driver library (resolves under --gpus)"; \
       fi \
    && ls -l /opt/cuda-link

ENV LIBRARY_PATH=/opt/cuda-link

# ==============================================================================
# PYTHON PACKAGE INSTALLATION
# ==============================================================================
# Install Boltz and cuEquivariance accelerated kernels
RUN pip install --no-cache-dir \
        "boltz[cuda]==${BOLTZ_VERSION}" \
        "cuequivariance-torch==${CUEQUIVARIANCE_VERSION}" \
        "cuequivariance-ops-torch-cu12==${CUEQUIVARIANCE_VERSION}"

# ==============================================================================
# BUILD-TIME STACK CONSISTENCY GUARDS
# ==============================================================================
# 1. Verify PyTorch version stability and torchvision operator registration
RUN python -c "import torch, torchvision; \
from torchvision.ops import nms; \
assert torch.__version__.startswith('2.6.'), \
    f'PyTorch upgraded off base version: {torch.__version__}'; \
print(f'PyTorch v{torch.__version__} | torchvision v{torchvision.__version__} - Operators verified OK')"

# 2. Non-fatal check: verify cuEquivariance kernel module importability
RUN python -c "import torch; \
import cuequivariance_ops_torch; \
from cuequivariance_ops_torch import triangle_multiplicative_update; \
print('cuEquivariance triangle kernels loadable OK')" \
    || echo "NOTE: cuEquivariance kernels unverified at build time (no driver mounted); build_all.sh will verify on GPU host."

# 3. Confirm Numba honors NUMBA_CACHE_DIR environment variable
RUN python -c "from numba.core import config; \
import numba; \
assert config.CACHE_DIR == '/cache/numba', \
    f'Numba v{numba.__version__} ignored NUMBA_CACHE_DIR (Active CACHE_DIR: {config.CACHE_DIR!r})'; \
print(f'Numba v{numba.__version__} honors NUMBA_CACHE_DIR -> {config.CACHE_DIR}')"

# ==============================================================================
# VOLUMES & WORKSPACE CONFIGURATION
# ==============================================================================
# Declare /cache as an external mount point so the ~6.2 GB of downloaded model weights
# and JIT outputs outlive container rebuilds
VOLUME /cache

# Set /work as default working directory where target input specs and output files resolve
WORKDIR /work

# ==============================================================================
# EXECUTION ENTRYPOINT
# ==============================================================================
# Configure default container executable command to run Boltz-2 prediction
ENTRYPOINT ["boltz", "predict", \
            "--cache", "/cache/boltz", \
            "--output_format", "pdb", \
            "--write_full_pae"]