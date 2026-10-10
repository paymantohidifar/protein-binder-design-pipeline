# ==============================================================================
# PIPELINE COMPONENT: RFdiffusion
# ==============================================================================
# Role: $SE(3)$-equivariant backbone diffusion for de novo protein design.
#
# Upstream Reference: https://github.com/RosettaCommons/RFdiffusion
#
# THREE LOAD-BEARING PIPELINE ADAPTATIONS FROM UPSTREAM:
# 1. Self-Contained Build: Clones a pinned git commit hash rather than using
#    `COPY . /app/RFdiffusion/`. This allows `docker build docker/rfdiffusion`
#    to run self-contained without requiring the entire repository in the context.
# 2. Host Storage Mounting: Defines `WORKDIR /work` and `VOLUME /cache` to align
#    with `pipeline/docker.py`, ensuring large weights (~10 GB) and output files
#    live entirely on host `/home` storage rather than root.
# 3. Interpreter Symlink: Creates a `/usr/bin/python` symlink so probing scripts
#    (e.g., GPU compute capability checks) can invoke `python` directly.
#
# HARDWARE CONSTRAINTS & WEIGHT STORAGE:
# - Hardware Pin: Built on CUDA 11.6 / PyTorch 1.12. Emits no `sm_90` kernels.
#   Runs on A10 (`sm_86`) and A100 (`sm_80`), but CANNOT run on H100 or newer.
# - Weight Caching: Model checkpoints reside on the host `/cache` volume and are
#   passed at runtime via Hydra flags (e.g., `inference.model_directory_path=/cache/rfdiffusion/models`).

# ==============================================================================
# BASE IMAGE & GPU RUNTIME
# ==============================================================================
# Pulls NVIDIA's official enterprise runtime image (Ubuntu 20.04 + CUDA 11.6 + cuDNN 8).
# The 'runtime' variant excludes build compilers (nvcc), saving ~2-3 GB on the root drive.
FROM nvcr.io/nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

# ==============================================================================
# BUILD ARGUMENTS & ENVIRONMENT VARIABLES
# ==============================================================================
# Pin the exact upstream commit SHA from RosettaCommons/RFdiffusion for build reproducibility
ARG RFDIFFUSION_COMMIT=b44206a2a79f219bb1a649ea50603a284c225050

# DEBIAN_FRONTEND=noninteractive : Suppresses interactive OS prompts during apt installs
# PYTHONUNBUFFERED=1             : Streams stdout/stderr directly to terminal without log buffering
# PYTHONDONTWRITEBYTECODE=1      : Prevents writing temporary .pyc bytecode files to disk
# DGLBACKEND=pytorch             : Configures Deep Graph Library to use PyTorch as backend
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DGLBACKEND=pytorch

# ==============================================================================
# SYSTEM PACKAGES & PYTHON ENVIRONMENT SETUP
# ==============================================================================
# 1. Update package list quietly (-q)
# 2. Install minimal dependencies (git, Python 3.9, pip) without optional bloat
# 3. Upgrade pip without saving wheel packages in cache
# 4. Create symlink from /usr/bin/python3.9 -> /usr/bin/python
# 5. Purge apt index files and autoremove to keep this Docker layer lightweight
RUN apt-get -q update \
    && apt-get install --no-install-recommends -y \
        git \
        python3.9 \
        python3-pip \
    && python3.9 -m pip install -q -U --no-cache-dir pip \
    && ln -sf /usr/bin/python3.9 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

# ==============================================================================
# CODEBASE RETRIEVAL
# ==============================================================================
# Set working directory to /app
WORKDIR /app

# Clone the RFdiffusion repository and quietly check out the exact pinned commit SHA
RUN git clone https://github.com/RosettaCommons/RFdiffusion.git \
    && cd RFdiffusion \
    && git checkout --quiet ${RFDIFFUSION_COMMIT}

# ==============================================================================
# PYTHON DEPENDENCIES & PACKAGE INSTALLATION
# ==============================================================================
# Single pip invocation to consolidate all dependencies into a single image layer:
# - dgl & torch: Exact CUDA 11.6 (+cu116) binaries matching upstream pins
# - e3nn: Euclidean 3D coordinate-equivariant neural network library
# - wandb: Experiment tracking and logging
# - pynvml: NVIDIA Management Library Python bindings for monitoring GPU VRAM
# - dllogger: NVIDIA utility installed directly from GitHub (not available on PyPI)
# - decorator, hydra-core, pyrsistent: Configuration management and script utilities
# - /app/RFdiffusion/env/SE3Transformer: Local module for coordinate-equivariant transformations
#
# Final command installs RFdiffusion itself without overriding installed dependency pins (--no-deps)
RUN python3.9 -m pip install -q --no-cache-dir \
        dgl==1.0.2+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html \
        torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 \
        e3nn==0.3.3 \
        wandb==0.12.0 \
        pynvml==11.0.0 \
        git+https://github.com/NVIDIA/dllogger#egg=dllogger \
        decorator==5.1.0 \
        hydra-core==1.3.2 \
        pyrsistent==0.19.3 \
        /app/RFdiffusion/env/SE3Transformer \
    && python3.9 -m pip install --no-cache-dir /app/RFdiffusion --no-deps

# ==============================================================================
# VOLUMES & WORKSPACE CONFIGURATION
# ==============================================================================
# Declare /cache as an external mount point for model checkpoints (~10 GB), preventing
# root partition bloat and keeping weights intact across image rebuilds
VOLUME /cache

# Set /work as the execution workspace where host project files/inputs resolve relative paths
WORKDIR /work

# Configure default container executable entrypoint to run RFdiffusion inference
ENTRYPOINT ["python3.9", "/app/RFdiffusion/scripts/run_inference.py"]