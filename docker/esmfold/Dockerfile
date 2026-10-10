# ==============================================================================
# PIPELINE COMPONENT: ESMFold
# ==============================================================================
# Role: Fast single-sequence monomer structure prediction gate in the design pipeline.
#
# PROVENANCE & TECHNICAL NOTES:
# 1. Provenance: No upstream Dockerfile exists in facebookresearch/esm. Follows
#    the official pip install path with [esmfold] dependencies.
# 2. Python Version (Deliberate Deviation): Upstream README recommends Python <= 3.9.
#    However, pytorch/pytorch:2.1.0 ships Python 3.10 by default via its Makefile.
#    The <= 3.9 ceiling traces to the [esmfold] extra's deepspeed==0.5.9, which is
#    sdist-only and whose classifiers stop at 3.9—the one package omitted here,
#    since OpenFold guards it via find_spec. All 17 OpenFold/ESMFold modules on the
#    import path byte-compile cleanly under 3.10. The interpreter guard and smoke
#    test keep this assumption strictly verified at build time.
# 3. GPU Architecture: Uses CUDA 11.8 / PyTorch 2.1.0 (has the same H100/sm_90
#    kernel exclusion as RFdiffusion; works on A10/A100/RTX 3090/4090, but fails on H100+).
# 4. Weight Caching: ESM-2 3B weights (~10 GB) are NOT baked into this image.
#    They download automatically on first run into /cache (redirected via HF_HOME & TORCH_HOME).
# ==============================================================================

# ==============================================================================
# BASE IMAGE & GPU RUNTIME
# ==============================================================================
# Pulls PyTorch's official development image (PyTorch 2.1.0 + CUDA 11.8 + cuDNN 8).
# The 'devel' variant is required because compiling OpenFold's custom C++/CUDA kernels
# requires nvcc and full CUDA build headers.
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# ==============================================================================
# BUILD ARGUMENTS & ENVIRONMENT VARIABLES
# ==============================================================================
# Pin OpenFold repository commit SHA (PR #199: ESMFold support) for build reproducibility
ARG OPENFOLD_COMMIT=4b41059694619831a7db195b7e0988fc4ff3a307

# PYTHONDONTWRITEBYTECODE=1 : Disables writing temporary .pyc bytecode files to disk
# PYTHONUNBUFFERED=1        : Streams stdout/stderr directly to logs without buffering
# HF_HOME=/cache/hf         : Redirects HuggingFace models/cache onto host-mounted storage
# TORCH_HOME=/cache/torch   : Redirects PyTorch Hub downloads (ESM-2 weights) onto host storage
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf \
    TORCH_HOME=/cache/torch

# ==============================================================================
# SYSTEM PACKAGES & COMPILER SETUP
# ==============================================================================
# 1. Update apt package index and install git and build-essential (gcc/g++)
# 2. Immediately purge /var/lib/apt/lists/* in the same layer to minimize image size
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# INTERPRETER VERSION GUARD
# ==============================================================================
# Fails the build, not the campaign, if the base image's interpreter leaves the
# window this stack is known to tolerate (3.9 - 3.11).
RUN python -c "import sys; \
assert (3, 9) <= sys.version_info[:2] < (3, 12), \
    f'python {sys.version_info[0]}.{sys.version_info[1]} is outside the tested 3.9-3.11 window'; \
print(f'python {sys.version.split()[0]} OK (esm README documents <=3.9; see header)')"

# ==============================================================================
# PYTHON DEPENDENCIES: STEP 1 (DLLOGGER)
# ==============================================================================
# Install NVIDIA dllogger directly from GitHub (required dependency of OpenFold;
# not published on PyPI)
RUN pip install --no-cache-dir \
        "dllogger @ git+https://github.com/NVIDIA/dllogger.git"

# ==============================================================================
# PYTHON DEPENDENCIES: STEP 2 (FAIR-ESM & SCIENTIFIC STACK)
# ==============================================================================
# Install fair-esm and supporting libraries required for ESMFold structure prediction.
# NumPy is explicitly pinned to 1.26.4 to prevent breaking API changes with NumPy 2.x releases.
RUN pip install --no-cache-dir \
        fair-esm==2.0.0 \
        omegaconf \
        einops \
        ml-collections \
        dm-tree \
        pytorch-lightning \
        biopython==1.81 \
        scipy \
        "numpy==1.26.4"

# ==============================================================================
# PYTHON DEPENDENCIES: STEP 3 (OPENFOLD CLONE, PATCH & KERNEL BUILD)
# ==============================================================================
# Clones OpenFold at commit 4b41059 and patches setup.py to use -std=c++17 instead
# of hardcoded -std=c++14. This resolves the PyTorch 2.1 C++17 requirement during
# nvcc kernel compilation without upgrading OpenFold to unverified major releases.
RUN git clone https://github.com/aqlaboratory/openfold.git /opt/openfold \
    && cd /opt/openfold \
    && git checkout --quiet ${OPENFOLD_COMMIT} \
    && if grep -q "'-std=c++14'" setup.py; then \
           sed -i "s/'-std=c++14'/'-std=c++17'/" setup.py; \
           echo "Patched openfold setup.py: -std=c++14 -> -std=c++17"; \
       elif grep -q "'-std=c++17'" setup.py; then \
           echo "openfold setup.py already requests -std=c++17; no patch needed"; \
       else \
           echo "ERROR: No -std flag found in openfold setup.py—re-audit setup.py before building." >&2; \
           exit 1; \
       fi \
    && grep -n "std=c++" setup.py \
    && pip install --no-cache-dir . \
    && cd / && rm -rf /opt/openfold

# ==============================================================================
# BUILD-TIME IMPORT SMOKE TEST
# ==============================================================================
# Exercises the entire ESMFold import chain (esm -> openfold -> dependencies)
# to ensure zero missing modules prior to runtime execution.
RUN python -c "\
from esm.esmfold.v1.esmfold import ESMFold; \
import esm, openfold, omegaconf, einops, ml_collections, tree, scipy, Bio; \
from openfold.model.structure_module import StructureModule; \
from openfold.np import residue_constants; \
from openfold.utils.loss import compute_predicted_aligned_error, compute_tm; \
from openfold.data.data_transforms import make_atom14_masks; \
from openfold.utils.feats import atom14_to_atom37; \
print('ESMFold import chain OK')"

# Separate and non-fatal check: confirms presence of compiled CUDA extension (.so).
# dlopen may fail without host GPU driver mounted, so host bootstrap script performs authoritative check.
RUN python -c "import attn_core_inplace_cuda; print('OpenFold CUDA extension present')" \
    || echo "NOTE: OpenFold CUDA extension not loadable at build time (no driver); build_all.sh will verify on host."

# ==============================================================================
# SCRIPT INJECTION & WORKSPACE CONFIGURATION
# ==============================================================================
# Copy local inference runner script directly into /opt in the container root filesystem
COPY fold.py /opt/fold.py

# Declare /cache as an external volume mount point for model weights and HuggingFace assets,
# ensuring large files sit on host /home storage rather than the root partition
VOLUME /cache

# Set /work as default working directory where input sequences and output PDBs resolve
WORKDIR /work

# Configure default container executable command to execute the fold.py runner
ENTRYPOINT ["python", "/opt/fold.py"]