# RFdiffusion -- SE(3)-equivariant backbone diffusion
#
# Upstream reference:
#   https://github.com/RosettaCommons/RFdiffusion/blob/main/docker/Dockerfile
#
# Package set and pins are taken verbatim from the upstream Dockerfile and from
# env/SE3Transformer/requirements.txt.
#
# Three deliberate differences from upstream, each load-bearing for this
# pipeline:
#
#   1. Clones a pinned commit instead of `COPY . /app/RFdiffusion/`. Upstream's
#      form requires the build context to be a checkout of the repo
#      (`docker build -f docker/Dockerfile .`). Cloning makes
#      `docker build docker/rfdiffusion` self-contained, which is how
#      docker/build_all.sh invokes it, and the SHA makes the image
#      reproducible.
#   2. WORKDIR /work and VOLUME /cache, matching how pipeline/docker.py mounts
#      the repo and the shared weight cache.
#   3. A `python` symlink, so `docker run --entrypoint python` works for the
#      sm_ compatibility probe without hardcoding `python3.9`.
#
# CUDA 11.6 / PyTorch 1.12 are upstream's pins. this image emits no sm_90 kernels and
# therefore CANNOT run on an H100 or newer. A10 (sm_86) and A100 (sm_80) work.
#
# Weights stay on the host cache volume rather than in the image: the full
# checkpoint set is ~10 GB and outlives any rebuild. Callers pass the location
# through hydra, e.g. inference.model_directory_path=/cache/rfdiffusion/models
# or a single checkpoint via inference.ckpt_override_path.
#
# ==============================================================================
# BASE IMAGE & GPU RUNTIME
# ==============================================================================
# Uses NVIDIA's official enterprise runtime image (Ubuntu 20.04 + CUDA 11.6 + cuDNN 8).
# The 'runtime' variant excludes compiler tools (like nvcc), saving ~2-3 GB of disk space.
FROM nvcr.io/nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

# ==============================================================================
# BUILD ARGUMENTS & ENVIRONMENT VARIABLES
# ==============================================================================
# Pin the specific git commit hash from RosettaCommons/RFdiffusion for absolute reproducibility
ARG RFDIFFUSION_COMMIT=b44206a2a79f219bb1a649ea50603a284c225050

# DEBIAN_FRONTEND=noninteractive : Suppresses interactive prompts during apt installs
# PYTHONUNBUFFERED=1             : Streams stdout/stderr directly to logs without buffering
# DGLBACKEND=pytorch             : Sets Deep Graph Library backend to PyTorch
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    DGLBACKEND=pytorch

# ==============================================================================
# SYSTEM PACKAGES & PYTHON ENVIRONMENT SETUP
# ==============================================================================
# 1. Update package list quietly (-q)
# 2. Install minimal required packages (git, Python 3.9, pip) without extra bloat
# 3. Upgrade pip to the latest version without caching wheels
# 4. Create a symlink so calling 'python' invokes 'python3.9'
# 5. Purge apt index files, autoremove, and clean to keep this layer lightweight
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

# Clone the RFdiffusion repository and quietly check out the pinned commit
RUN git clone https://github.com/RosettaCommons/RFdiffusion.git \
    && cd RFdiffusion \
    && git checkout --quiet ${RFDIFFUSION_COMMIT}

# ==============================================================================
# PYTHON DEPENDENCIES & PACKAGE INSTALLATION
# ==============================================================================
# Single pip invocation to install all dependencies in one Docker layer:
# - dgl & torch: Pinned versions compiled specifically for CUDA 11.6 (+cu116)
# - e3nn: Euclidean 3D equivariant neural network library
# - wandb: Experiment tracking and logging
# - pynvml: NVIDIA Management Library bindings for monitoring GPU usage
# - dllogger: NVIDIA logging utility installed directly from GitHub (not on PyPI)
# - decorator, hydra-core, pyrsistent: Configuration management and utilities
# - /app/RFdiffusion/env/SE3Transformer: Bundled coordinate-equivariant module
#
# Final command installs RFdiffusion package itself without altering installed dependencies (--no-deps)
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
# VOLUMES & EXECUTION CONFIGURATION
# ==============================================================================
# Declare /cache as a mount point for external model weights (~10 GB), keeping them out of the image
VOLUME /cache

# Set /work as default workspace where host files/repos will be mounted for relative path resolution
WORKDIR /work

# Configure default command to execute RFdiffusion's inference script upon container launch
ENTRYPOINT ["python3.9", "/app/RFdiffusion/scripts/run_inference.py"]
