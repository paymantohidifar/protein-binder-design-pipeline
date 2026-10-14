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
# Pulls PyTorch's official runtime image (PyTorch 2.6.0 + CUDA 12.6 + cuDNN 9).
# The 'runtime' variant excludes C++ compilers, saving gigabytes on the root drive;
# nothing here compiles, since cuEquivariance ships prebuilt .so files.
#
# THE CUDA FLOOR IS 12.5, AND IT IS WHAT CHOSE THIS TAG.
# The previous base was cuda12.1 and produced this at the first prediction step:
#
#   Error while loading libcue_ops.so: undefined symbol:
#   cublasGemmGroupedBatchedEx, version libcublas.so.12
#   ImportError: Error importing triangle_multiplicative_update from
#   cuequivariance_ops_torch.
#
# `cublasGemmGroupedBatchedEx` (grouped batched GEMM) is a cuBLAS >= 12.5
# symbol. The dependency is declared, not inferred:
#
#   cuequivariance-ops-torch-cu12 -> cuequivariance-ops-cu12
#                                 -> nvidia-cublas-cu12>=12.5.0
#
# so pip DID install a >= 12.5 cuBLAS wheel; the dynamic loader simply resolved
# the image's own CUDA 12.1 libcublas.so.12 first. A cuda12.6 base removes the
# conflict rather than fighting the search order with LD_LIBRARY_PATH, and 12.6
# satisfies the floor for every cuEquivariance release (0.12.0 declares the
# same >= 12.5). Do NOT move this below cuda12.5.
#
# PYTHON COMPATIBILITY: Boltz requires Python >=3.10,<3.13. Every
# pytorch/pytorch tag from 2.6.0 through 2.14.1 ships Python 3.11 (checked
# against docker.Makefile's PYTHON_VERSION at each release tag), so this
# constraint does not pin the choice. Do NOT move to a tag shipping 3.13.
#
# WHY 2.6.0 AND NOT A NEWER TORCH: 2.6.0 is the OLDEST tag with cuda >= 12.5,
# which keeps the stack closest to what boltz 2.2.1 was tested against (it pins
# pytorch-lightning==2.5.0, contemporaneous with torch 2.5-2.6). Newer tags up
# to 2.14.1 exist and would additionally permit cuequivariance 0.11.1+, whose
# torch floor is >= 2.11 -- but that trades five more minor versions of drift
# for unquantified kernel gains.
#
# torch 2.6 FLIPPED `torch.load`'s weights_only DEFAULT to True. Checked, not
# assumed: boltz makes no direct torch.load call, it uses
# Boltz2.load_from_checkpoint, and pytorch-lightning 2.5.0 passes
# `weights_only=False` explicitly (lightning_fabric/utilities/cloud_io.py,
# `_load`), so checkpoint loading is unaffected.
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

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
# THE RECURRING DEFECT CLASS IN THIS IMAGE: a library writes to `~` and there
# is no writable HOME under `--user $UID:$GID`. HOME resolves to `/`, so every
# `~/.cache/<x>` becomes `/.cache/<x>`, which is root-owned. Three instances so
# far, each a crash at a different depth of the first real prediction:
#
#   numba          RuntimeError: cannot cache function
#                  '_prepare_msa_arrays_inner': no locator available
#   cuEquivariance PermissionError: [Errno 13] Permission denied: '/.cache'
#                  (cuequivariance_ops/triton/cache_manager.py:55, at IMPORT)
#   triton         would be next -- Path.home()/".triton"/"cache"
#                  (triton/runtime/cache.py:17), reached when the kernels
#                  actually compile
#
# XDG_CACHE_HOME is the class-level fix: platformdirs and appdirs both honour
# it, which covers cuEquivariance's `user_cache_dir(appname=...)` and numba's
# `AppDirs(appname="numba")`. The three specific variables are set anyway, so
# the redirect does not depend on any library choosing to consult XDG.
#
# Each is a real, read knob, verified in source rather than guessed:
#   CUEQ_TRITON_CACHE_DIR  cuequivariance_ops/triton/cache_manager.py:45
#                          (`overridden_cache_dir()`); a pure path redirect --
#                          it only additionally suppresses a root-in-Docker
#                          warning, it changes no behaviour
#   TRITON_CACHE_DIR       triton/runtime/cache.py:65
#   NUMBA_CACHE_DIR        numba/core/caching.py:205 (_UserProvidedCacheLocator)
#
# All point under /cache, the declared VOLUME, so compiled kernels and JIT
# artifacts survive `--rm` and are paid for once per campaign, not per design.
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
# 1. git -- pip needs it for any VCS requirement.
# 2. build-essential -- a C COMPILER IS A RUNTIME DEPENDENCY here, not a build
#    one, which is why a 'runtime' base image is not sufficient on its own.
#    Triton JIT-compiles a small CPython extension (`cuda_utils`) from
#    third_party/nvidia/backend/driver.c on its FIRST kernel launch, and
#    triton/runtime/build.py:28 looks for $CC, then gcc, then clang. With none
#    present the campaign dies mid-prediction with:
#
#      RuntimeError: Failed to find C compiler. Please specify via CC
#      environment variable.
#
#    Reached through cuEquivariance's fused layer norm
#    (fused_layer_norm_torch.py:160 -> triton jit.py:568 -> driver init), i.e.
#    only once a REAL boltz forward pass runs the triangular update with actual
#    LayerNorm weights. The compile result is cached under TRITON_CACHE_DIR, so
#    the cost is paid once per campaign -- but the compiler must be present for
#    that first run and for any later cache miss (e.g. a triton version bump).
# 3. Purge apt lists in the same layer to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# LINKER PATH FOR TRITON'S JIT (-lcuda)
# ==============================================================================
# That same triton build links against the CUDA driver: `libraries = ['cuda']`
# (third_party/nvidia/backend/driver.py:16), so the linker needs a file named
# exactly `libcuda.so`. The nvidia container runtime injects `libcuda.so.1` --
# the versioned SONAME, with no `.so` development symlink -- so `-lcuda` fails
# even on a host with a working driver.
#
# Both plausible sources are handled at build time so the outcome is
# deterministic rather than dependent on the base flavour:
#   * a CUDA toolkit stub, if this base carries one ('devel' tags do);
#   * otherwise a symlink to where the runtime injects the real driver library.
#     That symlink DANGLES at build time and resolves under `--gpus`, which is
#     fine: it is consumed only by the linker, at which point the file exists.
#
# Linking against a stub and loading the real driver at run time is the
# standard CUDA arrangement -- the SONAME recorded is `libcuda.so.1`, which
# ldconfig resolves to the injected driver.
#
# LIBRARY_PATH (not LD_LIBRARY_PATH) is the correct variable: it is gcc's
# link-time search path. Using LD_LIBRARY_PATH would shadow the real driver at
# load time, which is exactly what must not happen.
RUN mkdir -p /opt/cuda-link \
    && if [ -e /usr/local/cuda/lib64/stubs/libcuda.so ]; then \
           ln -s /usr/local/cuda/lib64/stubs/libcuda.so /opt/cuda-link/libcuda.so; \
           echo "cuda-link: using the CUDA toolkit stub"; \
       else \
           ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /opt/cuda-link/libcuda.so; \
           echo "cuda-link: symlink to the driver library injected at run time (dangling until --gpus)"; \
       fi \
    && ls -l /opt/cuda-link
ENV LIBRARY_PATH=/opt/cuda-link

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
#
# Scope: torch/torchvision ONLY. The cuEquivariance kernels cannot be checked
# here -- see the non-fatal probe below for why.
RUN python -c "import torch, torchvision; \
from torchvision.ops import nms; \
assert torch.__version__.startswith('2.6.'), \
    f'torch moved to {torch.__version__}; a dependency upgraded it off the base image'; \
print(f'torch {torch.__version__} / torchvision {torchvision.__version__} -- ops register OK')"

# Separate and deliberately NON-FATAL: cuEquivariance's kernels.
#
# `libcue_ops.so` links against `libcuda.so.1`, the NVIDIA DRIVER stub, which
# the nvidia container runtime injects at RUN time and which does not exist
# during `docker build`. So this import cannot succeed here regardless of
# whether the image is correct:
#
#   Error while loading libcue_ops.so: libcuda.so.1: cannot open shared
#   object file: No such file or directory
#
# A previous revision asserted this fatally, on the mistaken belief that the
# cuBLAS mismatch was a pure link-time symbol check needing no driver. It is
# not: the driver dependency fails first, so the build broke on a healthy
# image. The authoritative check is `docker/build_all.sh`, which runs with
# `--gpus all`, where the nvidia container runtime injects the driver
# libraries -- exactly the arrangement the esmfold image already uses for
# OpenFold's CUDA extension. CONFIRMED on an A10: that probe reports
# `cuEquivariance triangle kernel executed, out=(1, 32, 32, 32)`.
#
# The import must still name cuequivariance_ops_torch, NOT cuequivariance_torch:
# the wrapper defers `from cuequivariance_ops_torch import
# triangle_multiplicative_update` to INSIDE the function
# (cuequivariance_torch/primitives/triangle.py:238), so importing the wrapper
# dlopens nothing and reports success on an image whose kernels are unusable.
#
# `import torch` comes first deliberately. torch's __init__ preloads the NVIDIA
# pip wheels it depends on (nvidia-cuda-nvrtc-cu12 and friends), which puts
# libnvrtc.so.12 in the process. Without it, this layer reported a MISLEADING
# cause -- the observed build output was
#
#   Error while loading libcue_ops.so: libnvrtc.so.12: cannot open shared
#   object file
#
# i.e. NVRTC, not the driver, because nothing had loaded torch yet. Production
# always imports torch before reaching these kernels, so importing it here
# makes the diagnostic reflect the real remaining gap. The message below names
# no specific library for the same reason: which one is missing depends on the
# base image flavour and on what has been loaded first.
RUN python -c "import torch; \
import cuequivariance_ops_torch; \
from cuequivariance_ops_torch import triangle_multiplicative_update; \
print('cuequivariance triangle kernels loadable OK')" \
    || echo "NOTE: cuEquivariance kernels not loadable at build time (expected -- needs CUDA driver/toolkit libraries the nvidia runtime provides only at run time); build_all.sh verifies on the GPU host"

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