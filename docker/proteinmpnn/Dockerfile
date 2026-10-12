# ==============================================================================
# PIPELINE COMPONENT: ProteinMPNN / SolubleMPNN
# ==============================================================================
# Role: Fixed-backbone sequence design engine (generating amino acid sequences 
#       conditioned on 3D backbone coordinates).
# Upstream Reference: https://github.com/dauparas/ProteinMPNN

# ==============================================================================
# BASE IMAGE & GPU RUNTIME
# ==============================================================================
# Pulls PyTorch's official lightweight runtime image (PyTorch 2.1.0 + CUDA 11.8 + cuDNN 8).
# The 'runtime' variant excludes C++ build compilers (nvcc), saving gigabytes on the root drive.
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# ==============================================================================
# BUILD ARGUMENTS & ENVIRONMENT VARIABLES
# ==============================================================================
# Pin the specific git commit SHA from dauparas/ProteinMPNN for exact build reproducibility
ARG PROTEINMPNN_COMMIT=8907e6671bfbfc92303b5f79c4b5e6ce47cdef57

# PYTHONDONTWRITEBYTECODE=1 : Stops Python from writing temporary .pyc files to disk
# PYTHONUNBUFFERED=1        : Streams stdout/stderr directly to terminal without log buffering
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# ==============================================================================
# SYSTEM PACKAGES SETUP
# ==============================================================================
# 1. Update apt package index and install git without optional dependencies (--no-install-recommends)
# 2. Immediately purge apt package lists in the same layer to minimize final image size
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# CODEBASE & IN-TREE MODEL WEIGHTS
# ==============================================================================
# Set working directory to /opt
WORKDIR /opt

# Clone the ProteinMPNN repository self-contained and check out the exact pinned commit.
RUN git clone https://github.com/dauparas/ProteinMPNN.git \
    && cd ProteinMPNN && git checkout ${PROTEINMPNN_COMMIT}

# ==============================================================================
# VOLUMES & EXECUTION CONFIGURATION
# ==============================================================================
# Declare /cache as an external mount point for shared host caching across pipeline tools
VOLUME /cache

# Set /work as the execution workspace where input PDBs and output sequence fastas resolve
WORKDIR /work

# Configure default container executable entrypoint to run ProteinMPNN's runner script.
ENTRYPOINT ["python", "/opt/ProteinMPNN/protein_mpnn_run.py"]