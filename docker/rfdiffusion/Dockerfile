# ==============================================================================
# PIPELINE COMPONENT: RFdiffusion
# ==============================================================================
# Role: $SE(3)$-equivariant backbone diffusion for de novo protein design.
# Upstream Reference: https://github.com/RosettaCommons/RFdiffusion
#
# TECHNICAL NOTES:
# 1. Self-Contained Build: Clones a pinned git commit hash directly to make the
#    build independent of the host directory context.
# 2. Hardware Constraints: CUDA 11.6 / PyTorch 1.12 binary stack. Runs on A10/A100
#    GPUs (`sm_80`/`sm_86`), but CANNOT run on H100+ (`sm_90`).
# 3. Cache & User Permissions: Grants write permissions to the package directory
#    so runtime schedules can be cached cleanly when running as non-root users.
# ==============================================================================

# ==============================================================================
# BASE IMAGE & GPU RUNTIME
# ==============================================================================
# Pulls NVIDIA's official enterprise runtime image (Ubuntu 20.04 + CUDA 11.6 + cuDNN 8)
FROM nvcr.io/nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

# ==============================================================================
# BUILD ARGUMENTS & ENVIRONMENT VARIABLES
# ==============================================================================
# Pin upstream commit SHA for exact build reproducibility
ARG RFDIFFUSION_COMMIT=b44206a2a79f219bb1a649ea50603a284c225050

# DEBIAN_FRONTEND=noninteractive : Suppresses interactive OS prompts during apt installs
# PYTHONUNBUFFERED=1             : Streams stdout/stderr directly to terminal logs without buffering
# PYTHONDONTWRITEBYTECODE=1      : Prevents writing temporary .pyc bytecode files to disk
# DGLBACKEND=pytorch             : Configures Deep Graph Library to use PyTorch as backend
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DGLBACKEND=pytorch

# ==============================================================================
# SYSTEM PACKAGES & PYTHON ENVIRONMENT SETUP
# ==============================================================================
# Install Python 3.9 and minimal build utilities, then symlink python to python3.9
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        python3.9 \
        python3-pip \
    && python3.9 -m pip install -q -U --no-cache-dir pip \
    && ln -sf /usr/bin/python3.9 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# CODEBASE RETRIEVAL & DEPENDENCY INSTALLATION
# ==============================================================================
# Clone codebase and install pinned PyTorch, DGL, SE3Transformer, and RFdiffusion
RUN git clone https://github.com/RosettaCommons/RFdiffusion.git /app/RFdiffusion \
    && cd /app/RFdiffusion \
    && git checkout --quiet ${RFDIFFUSION_COMMIT} \
    && python3.9 -m pip install -q --no-cache-dir \
        "numpy==1.26.4" \
        dgl==1.0.2+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html \
        torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 \
        e3nn==0.3.3 \
        wandb==0.12.0 \
        pynvml==11.0.0 \
        "dllogger @ git+https://github.com/NVIDIA/dllogger" \
        decorator==5.1.0 \
        hydra-core==1.3.2 \
        pyrsistent==0.19.3 \
        /app/RFdiffusion/env/SE3Transformer \
    && python3.9 -m pip install --no-cache-dir --no-deps /app/RFdiffusion \
    && chmod -R 777 /usr/local/lib/python3.9/dist-packages/rfdiffusion

# ==============================================================================
# VOLUMES & WORKSPACE CONFIGURATION
# ==============================================================================
# Declare /cache as an external mount point for model checkpoints (~10 GB)
VOLUME /cache

# Set /work as default working directory where host project input/output files resolve
WORKDIR /work

# ==============================================================================
# EXECUTION ENTRYPOINT
# ==============================================================================
# Configure default container executable command to execute RFdiffusion inference
ENTRYPOINT ["python", "/app/RFdiffusion/scripts/run_inference.py"]