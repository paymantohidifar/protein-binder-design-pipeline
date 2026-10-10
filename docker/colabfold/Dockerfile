# ==============================================================================
# PIPELINE COMPONENT: LocalColabFold
# ==============================================================================
# Role: Fast AlphaFold2 / AF2-Multimer v3 orthogonal structure prediction and 
#       validation engine (used as a quality filter gate for design candidates).
# Upstream Reference: https://github.com/sokrypton/ColabFold
#
# PROVENANCE & WEIGHT ARCHITECTURE:
# 1. Base Image: Built on ColabFold's official pre-compiled CUDA 12 image, which 
#    bundles Jax, OpenMM, Kalign, and MMseqs2 binaries out of the box.
# 2. Weight Caching: AlphaFold2 parameters (~4.5 GB) download on first execution. 
#    The `--data /cache/colabfold` flag explicitly directs parameter downloads onto 
#    the host-mounted `/cache` volume, keeping heavy weights off the root partition.
# 3. Model Target: Defaults to `alphafold2_multimer_v3`, optimizing complex prediction 
#    for multi-chain protein-protein binder validations.
# ==============================================================================

# ==============================================================================
# BASE IMAGE & GPU RUNTIME
# ==============================================================================
# Pulls ColabFold's official release container (v1.6.2 with CUDA 12 support).
# Bundles Python, JAX, CUDA runtimes, and sequence alignment utilities (MMseqs2).
FROM ghcr.io/sokrypton/colabfold:1.6.2-cuda12

# ==============================================================================
# ENVIRONMENT VARIABLES
# ==============================================================================
# DEBIAN_FRONTEND=noninteractive : Suppresses interactive OS prompts during system updates
# PYTHONUNBUFFERED=1             : Streams stdout/stderr directly to logs without buffering
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# ==============================================================================
# VOLUMES & WORKSPACE CONFIGURATION
# ==============================================================================
# Declare /cache as an external volume mount point for downloaded AlphaFold2 parameters 
# (~4.5 GB) and MSA caching, ensuring weights outlive container rebuilds
VOLUME /cache

# Set /work as the default workspace where input FASTA files and output PDB/PDB-PAE 
# predictions resolve relative paths
WORKDIR /work

# ==============================================================================
# EXECUTION ENTRYPOINT
# ==============================================================================
# Configures container executable command. Automatically appends required flags:
# - colabfold_batch        : Main CLI batch prediction script
# - --model-type ...       : Enforces AF2-Multimer v3 weights for complex modeling
# - --data /cache/colabfold: Redirects weight downloads/storage to host volume
ENTRYPOINT ["colabfold_batch", \
            "--model-type", "alphafold2_multimer_v3", \
            "--data", "/cache/colabfold"]