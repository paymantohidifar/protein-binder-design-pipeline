# ==============================================================================
# PIPELINE COMPONENT: Boltz-2
# ==============================================================================
# Role: All-atom precision gate and complex structure/confidence prediction engine.
# Upstream Repository: https://github.com/jwohlwend/boltz
#
# CRITICAL ENTRYPOINT FLAGS (Load-Bearing for Pipeline Compatibility):
# 1. --output_format pdb
#    Boltz emits CIF by default. Because the downstream parsing pipeline lacks a CIF
#    reader, omitting this flag causes scRMSD and binder-RMSD calculations to be
#    silently skipped without raising an error (see parse_boltz2_monomer()).
#
# 2. --write_full_pae
#    Defaults to False upstream. Without this flag, the file writer receives no "pae"
#    key (data/write/writer.py), omitting pae_*.npz files. This causes _load_boltz_pae()
#    to fail and leaves pae_interaction as NaN. Since inter-chain PAE is the primary
#    Step 4 filter gate, missing this flag forces the campaign to rank solely on ipTM.
#
# MODEL WEIGHTS & CACHING ARTIFACTS:
# - On first prediction, Boltz verifies the cache directory and downloads missing weights
#   automatically (main.py:predict() creates cache.mkdir(parents, exist_ok) at line ~1111).
# - download_boltz2() fetches three artifacts (~6.21 GB total):
#     1. mols.tar         (1.86 GB) : Chemical Component Dictionary (CCD), extracted to mols/
#     2. boltz2_conf.ckpt (2.29 GB) : Structure prediction + PAE confidence weights (active)
#     3. boltz2_aff.ckpt  (2.06 GB) : Small-molecule binding affinity head (fetched but unused)
# - Cache Pathing: Boltz defaults to `~/.boltz` rather than `~/.cache/boltz`. Setting both
#   `ENV BOLTZ_CACHE=/cache/boltz` and `--cache /cache/boltz` ensures all downloaded model
#   artifacts redirect onto host `/home` storage rather than root.
# ==============================================================================

# ==============================================================================
# BASE IMAGE & GPU RUNTIME
# ==============================================================================
# Pulls PyTorch's official runtime image (PyTorch 2.4.0 + CUDA 12.1 + cuDNN 9).
# The 'runtime' variant excludes C++ compilers, saving gigabytes on the root drive.
#
# PYTHON COMPATIBILITY: Boltz requires Python >=3.10,<3.13. PyTorch 2.4.0 ships
# with Python 3.11 (via PyTorch's docker.Makefile), satisfying this constraint.
# Do NOT upgrade this base image tag to a release with Python 3.13, or pip will fail.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# ==============================================================================
# BUILD ARGUMENTS & ENVIRONMENT VARIABLES
# ==============================================================================
# Pin Boltz package version for exact campaign reproducibility
ARG BOLTZ_VERSION=2.2.1

# PYTHONDONTWRITEBYTECODE=1 : Prevents writing temporary .pyc bytecode files to disk
# PYTHONUNBUFFERED=1        : Streams stdout/stderr directly to terminal logs without buffering
# BOLTZ_CACHE               : Redirects the model weight cache directory onto mounted host storage
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BOLTZ_CACHE=/cache/boltz

# ==============================================================================
# SYSTEM PACKAGES SETUP
# ==============================================================================
# 1. Update apt package index and install git without optional recommended dependencies
# 2. Immediately purge apt index files in the same layer to minimize final image size
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# PYTHON PACKAGE INSTALLATION
# ==============================================================================
# Install the exact pinned version of Boltz with CUDA acceleration extras directly from PyPI.
# --no-cache-dir prevents pip from storing downloaded .whl files inside the container layer
RUN pip install --no-cache-dir "boltz[cuda]==${BOLTZ_VERSION}"

# ==============================================================================
# VOLUMES & WORKSPACE CONFIGURATION
# ==============================================================================
# Declare /cache as an external mount point so the ~6.2 GB of downloaded model weights
# outlive container rebuilds. pipeline/docker.py passes `-v <host cache>:/cache` at runtime.
VOLUME /cache

# Set /work as default working directory where target input specs and output PDBs resolve
WORKDIR /work

# ==============================================================================
# EXECUTION ENTRYPOINT
# ==============================================================================
# Sets default container executable. Automatically appends required flags (--cache,
# --output_format pdb, and --write_full_pae) to every invocation to guarantee pipeline output.
ENTRYPOINT ["boltz", "predict", \
            "--cache", "/cache/boltz", \
            "--output_format", "pdb", \
            "--write_full_pae"]