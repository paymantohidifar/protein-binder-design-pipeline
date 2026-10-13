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

# Pin cuEquivariance, which boltz[cuda] leaves unbounded (`>=0.5.0`).
#
# WHY THIS PIN EXISTS -- it fixes an observed runtime failure:
#   RuntimeError: operator torchvision::nms does not exist
#
# cuequivariance-ops-torch-cu12 moved `torch>=2.11` from a test-only extra to a
# HARD runtime dependency at 0.11.1:
#
#   <= 0.10.0   torch>=2.4; extra == "test"   (no effect on runtime resolution)
#   >= 0.11.1   torch>=2.11                   (forces a torch upgrade)
#
# Unpinned, pip resolves 0.12.0, which drags torch from this base image's 2.4.0
# up to 2.14.x. Nothing in boltz's tree REQUIRES torchvision, so pip never
# touches the base image's torchvision 0.19.0 -- which pins `torch==2.4.0`
# exactly and whose _C extension is compiled against that ABI. Its C++ ops then
# fail to register, and torchvision is imported anyway through a path nobody
# asked for: boltz -> pytorch_lightning 2.5.0 -> torchmetrics ->
# functional.image.arniqa -> `from torchvision import transforms`. The import
# dies registering a fake impl for an operator that was never created.
#
# 0.10.0 is the last release whose torch requirement is test-only, so it leaves
# the base image's matched torch/torchvision pair intact.
ARG CUEQUIVARIANCE_VERSION=0.10.0

# PYTHONDONTWRITEBYTECODE=1 : Prevents writing temporary .pyc bytecode files to disk
# PYTHONUNBUFFERED=1        : Streams stdout/stderr directly to terminal logs without buffering
# BOLTZ_CACHE               : Redirects the model weight cache directory onto mounted host storage
# NUMBA_CACHE_DIR           : Redirects numba's JIT cache onto the same mounted storage.
#
# NUMBA_CACHE_DIR is load-bearing, not an optimisation. boltz decorates
# _prepare_msa_arrays_inner with @numba.njit(..., cache=True)
# (data/feature/featurizer.py), and cache=True makes numba resolve a writable
# cache directory AT IMPORT TIME. Its locators are tried in order
# (numba/core/caching.py:361) and under `--user $UID:$GID` the first three all
# fail:
#
#   _UserProvidedCacheLocator  skipped unless NUMBA_CACHE_DIR is set
#   _InTreeCacheLocator        wants <site-packages>/boltz/data/feature/
#                              __pycache__ -- root-owned, not writable
#   _UserWideCacheLocator      wants $HOME/.cache/numba -- HOME is the image's
#                              /root, not writable by uid 1000
#
# leaving no locator and `RuntimeError: cannot cache function
# '_prepare_msa_arrays_inner': no locator available`. The container then exits 1
# before boltz parses a single argument. Setting this activates the first
# locator and short-circuits the rest.
#
# Pointing it under /cache (not /tmp) also persists the compiled kernels across
# `--rm` runs, so the JIT cost is paid once per campaign rather than per design.
# Note PYTHONDONTWRITEBYTECODE is unrelated: numba creates its cache dirs
# itself and never consults sys.dont_write_bytecode.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BOLTZ_CACHE=/cache/boltz \
    NUMBA_CACHE_DIR=/cache/numba

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
# --no-cache-dir prevents pip from storing downloaded .whl files inside the container layer.
# Both cuEquivariance packages are pinned together: NVIDIA releases them in
# lockstep, and it is the -ops- one that carries the torch floor.
RUN pip install --no-cache-dir \
        "boltz[cuda]==${BOLTZ_VERSION}" \
        "cuequivariance-torch==${CUEQUIVARIANCE_VERSION}" \
        "cuequivariance-ops-torch-cu12==${CUEQUIVARIANCE_VERSION}"

# ==============================================================================
# BUILD-TIME STACK CONSISTENCY GUARD
# ==============================================================================
# Fails the build, not a GPU-billed run, if any dependency moves torch off the
# base image's version or leaves torchvision built against a different ABI.
#
# This guard is the real lesson from the torchvision::nms failure: the image
# BUILT CLEANLY and only broke at the first real `boltz predict`, after the
# instance was already running. Importing torchvision.ops is the exact
# operation that was failing, so it is what gets asserted.
RUN python -c "import torch, torchvision; \
from torchvision.ops import nms; \
import cuequivariance_torch; \
assert torch.__version__.startswith('2.4.'), \
    f'torch moved to {torch.__version__}; a dependency upgraded it off the base image'; \
print(f'torch {torch.__version__} / torchvision {torchvision.__version__} -- ops register OK')"

# Confirm the installed numba actually reads NUMBA_CACHE_DIR, so a future numba
# that renames the variable fails the build rather than the first design run.
#
# This cannot verify the real failure: `docker build` runs as root, so numba's
# _InTreeCacheLocator can write into site-packages and would succeed here even
# with the variable unset. Only a run under `--user` exercises that path, which
# is why this defect reached the GPU host. What is checkable is that the knob is
# still wired up.
RUN python -c "from numba.core import config; \
import numba; \
assert config.CACHE_DIR == '/cache/numba', \
    f'numba {numba.__version__} did not read NUMBA_CACHE_DIR (CACHE_DIR={config.CACHE_DIR!r})'; \
print(f'numba {numba.__version__} honours NUMBA_CACHE_DIR -> {config.CACHE_DIR}')"

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