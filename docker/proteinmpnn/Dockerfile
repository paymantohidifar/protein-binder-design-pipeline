# ==============================================================================
# PIPELINE COMPONENT: ProteinMPNN / SolubleMPNN
# ==============================================================================
# Role: Fixed-backbone sequence design engine (generating amino acid sequences 
#       conditioned on 3D backbone coordinates).
# Upstream Reference: https://github.com/dauparas/ProteinMPNN
#
# PROVENANCE & WEIGHT ARCHITECTURE:
# 1. Custom Image: Upstream provides no Dockerfile; this container is built 
#    custom for the pipeline using PyTorch's official runtime image.
# 2. In-Tree Checkpoints: Unlike RFdiffusion, ESMFold, or Boltz-2, ProteinMPNN's 
#    weights (11 checkpoints = ~74 MB total: vanilla, soluble, and ca_model) ship 
#    directly inside the cloned repository in-tree.
# 3. Zero Bootstrap Downloads: Because weights reside directly in the codebase, 
#    this image requires no external weight downloads at container startup.
# 4. Weight Resolution: ProteinMPNN resolves model weights relative to its script 
#    location via `os.path.realpath(__file__)` (`protein_mpnn_run.py`), so no 
#    environment variables are required to locate them.
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
# Includes all checkpoint files directly within /opt/ProteinMPNN/vanilla_model_weights,
# /opt/ProteinMPNN/soluble_model_weights, etc.
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
# Any CLI flags passed to `docker run` (e.g., `--pdb_path`, `--use_soluble_model`) 
# append directly to this execution string.
ENTRYPOINT ["python", "/opt/ProteinMPNN/protein_mpnn_run.py"]