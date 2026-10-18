# ==============================================================================
# PIPELINE COMPONENT: ProteinMPNN / SolubleMPNN
# ==============================================================================
# Role: Fixed-backbone sequence design engine (generating amino acid sequences 
#       conditioned on 3D backbone coordinates).
# Upstream Reference: https://github.com/dauparas/ProteinMPNN
# ==============================================================================

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

# PYTHONDONTWRITEBYTECODE=1 : Prevents writing temporary .pyc bytecode files to disk
# PYTHONUNBUFFERED=1        : Streams stdout/stderr directly to terminal logs without buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ==============================================================================
# SYSTEM PACKAGES & CODEBASE SETUP
# ==============================================================================
# 1. Install git without optional dependencies to clone the codebase
# 2. Clone the repository and checkout the pinned commit
# 3. Purge git and apt lists in the same layer to keep the image footprint minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && git clone https://github.com/dauparas/ProteinMPNN.git /opt/ProteinMPNN \
    && cd /opt/ProteinMPNN \
    && git checkout --quiet ${PROTEINMPNN_COMMIT} \
    && apt-get purge -y git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# VOLUMES & WORKSPACE CONFIGURATION
# ==============================================================================
# Declare /cache as an external mount point for shared host caching across pipeline tools
VOLUME /cache

# Set /work as default working directory where input PDBs and output FASTA files resolve
WORKDIR /work

# ==============================================================================
# EXECUTION ENTRYPOINT
# ==============================================================================
# Configure default container executable command to execute the ProteinMPNN runner
ENTRYPOINT ["python", "/opt/ProteinMPNN/protein_mpnn_run.py"]