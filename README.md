# **Containerized *De Novo* Protein Binder Design Pipeline**

An end-to-end, cloud-native computational platform that enables users to design protein binders from scratch.

This pipeline orchestrates five state-of-the-art deep learning models across isolated, fully containerized GPU environments. Driven directly through clean Python wrappers, an interactive Jupyter notebook, and scalable Snakemake workflows, it aims to deliver a reproducible design funnel from target surface analysis to wet-lab-ready lead candidates.


### **Key Highlights**

* **Fully Containerized & Reproducible:** Every engine (`RFdiffusion`, `proteinMPNN`, `ESMFold`, `Boltz-2`, `ColabFold (AlphaFold Multimer)`) runs inside its own pinned, self-contained Docker environment for zero dependency friction.
* **Interactive Prototyping & Automated Scaling:** Seamlessly transition from notebook-based campaign prototyping to high-throughput Snakemake execution.
* **Tiered Quality Funnel:** Sequentially filters candidates through sequence scoring, monomer self-consistency, and dual-engine complex interface metrics—maximizing hit rates while conserving compute.


### **Pipeline Overview**

1. **Target Surface Preparation & Hotspots:** Quality-checks user-defined target hotspots or automatically identifies candidate binding pockets using geometric surface analysis.
2. **Backbone Generation (`RFdiffusion`):** Generates *de novo* binder backbones focused on selected target hotspot residues.
3. **Fixed-Backbone Sequence Design (`ProteinMPNN`):** Predicts soluble amino acid sequences tailored to pack tightly against the target interface.
4. **Monomer Self-Consistency Validation (`ESMFold` & `Boltz-2`):** Rapidly screens single-chain structures to confirm sequence-to-backbone fidelity ($\text{pLDDT}$ and $\text{scRMSD}$), filtering out misfold-prone designs early.
5. **Complex Interface Validation (`Boltz-2` & `ColabFold`):** Evaluates target-binder interactions side-by-side ($\text{ipTM}$ and $\text{pAE}_{\text{interaction}}$) to isolate top-tier leads for wet-lab synthesis.


### **Setup & Installation**

#### **Hardware & System Requirements**

* **Supported GPUs:** NVIDIA Ampere, Ada Lovelace, and Turing architectures (`sm_70`, `sm_80`, `sm_89`), including **A10**, **A100**, **L40S**, and **RTX 6000 Ada**.
  * *Note:* Hopper (H100) and Blackwell (B200) architectures are not currently supported.
* **VRAM & Storage:** Minimum **24 GB VRAM** and **~90 GB available NVMe storage** for container images, model weights, and workspace caches. For production-scale campaigns ($N > 10,000$ designs), we recommend $\ge 500\,\text{GB}$ on the primary compute volume or attached persistent block storage.
* **NVIDIA Driver:** Driver version $\ge 535$ with the NVIDIA Container Toolkit installed.
* **GPU Topology:** Single-GPU execution per container run. High-throughput scaling across multi-GPU setups is handled automatically via Snakemake.


#### **1. Provision a GPU Instance**

Deploy a cloud or local GPU instance running Ubuntu 22.04 LTS. For a step-by-step walkthrough on setting up instances on cloud providers like Lambda Labs, see **[`docs/CLOUD_SETUP.md`](docs/CLOUD_SETUP.md)**.

Once your instance is live, launch an interactive SSH session and proceed with the steps below.

#### **2. Clone the Repository**

Clone the pipeline codebase directly onto your primary workspace storage volume:

```bash
git clone https://github.com/paymantohidifar/protein-binder-design-pipeline.git --branch main binder_repo
cd binder_repo

```

#### **3. Verify Docker Privileges**

All pipeline stages execute inside isolated, pinned Docker containers. Configure non-root Docker privileges and verify container runtime access by running:

```bash
bash scripts/setup_docker.sh

```

> *Follow any terminal prompts (e.g., refreshing your user group session) to apply updated permissions.*

#### **4. Run Pre-Flight System Check**

Before building container images or fetching model weights, run our automated environment inspector to verify CUDA drivers, GPU topology, disk capacity, and container runtime prerequisites:

```bash
bash scripts/bootstrap_instance.sh --check

```

> *This pre-flight check catches hardware mismatches and missing dependencies before initiating large downloads or container builds.*

#### **5. Build Environment & Cache Weights**

Once the pre-flight check passes, bootstrap the instance environment. This step automatically installs the [Pixi](https://pixi.sh) package manager, sets up the host Python environment, builds all 5 pinned Docker images (~25 GB total), and pre-fetches model weights (~21 GB) into a host-mounted cache volume:

```bash
bash scripts/bootstrap_instance.sh --all

```

> *Build time takes approximately 20–30 minutes depending on network bandwidth and disk I/O. For advanced setup options, see **[`docs/BOOTSTRAP.md`](docs/BOOTSTRAP.md)**. Follow all post-build terminal prompts to complete setup.*

#### **6. Run Unit & Integration Tests**

Verify system setup, wrapper functionality, and container volume mounting by executing the test suite:

```bash
# Run standard test suite
pixi run test

# Run verbose per-test suite
pixi run test-verbose

```

### **Interactive Prototyping & Notebook Exploration**

We provide a complete interactive Jupyter notebook (`notebooks/01_binder_design_prototype.ipynb`) that walks through the entire pipeline step-by-step. It covers the underlying biophysical principles of each deep learning model while letting you run a complete **smoke-test campaign (5–10 binders)** to verify your instance setup, inspect metric distributions, and configure stage parameters before scaling up.

The included example campaign targets the **human insulin receptor L1/E1 domain**, using three key surface phenylalanines (**A59 / A83 / A91**) as the epitope anchor.


#### **Pre-Launch Jupyter Kernel Setup**

Before launching Jupyter on your cloud instance, complete the following environment configuration steps:

1. **Refresh your shell environment** to ensure `pixi` is available on your system `PATH`:
```bash
source ~/.bashrc

```

2. **Register the Pixi environment as a Jupyter kernel**:
```bash
pixi run register-kernel

```

*This registers the kernel environment displayed as **`Python (Pixi: default)`**.*

3. **Reboot the instance** to clear Jupyter's cached kernel registry:
```bash
sudo reboot

```

4. **Launch Jupyter and select the registered kernel**:
* Open `notebooks/01_binder_design_prototype.ipynb`.
* From the Jupyter kernel menu, select **`Python (Pixi: default)`** *(the notebook ships with a standard `python3` spec, so select the Pixi kernel manually on first launch)*.


#### **CPU-Only Pipeline Smoke Test (No GPU Required)**

To verify file paths, container mounts, and pipeline wiring without consuming GPU compute or running actual neural network inference:

```bash
pixi run notebook-dry-run

```

> **Note:** The dry-run executes the notebook with stubbed container calls to validate execution plumbing and data flow. To generate real structural models and evaluate binder metrics, run the full notebook on a GPU instance.


### **Production-Scale Runs**

> **Note:** Automated Snakemake workflow execution for multi-GPU scaling is currently under active development.

#### **Data Privacy & Remote API Calls**

All core deep learning engines run locally inside isolated containers without requiring external database lookups—with one explicit exception:

* **ColabFold & Boltz-2 (MSA Mode):** When running in MSA prediction mode, these models query the public MMseqs2 API server at `api.colabfold.com` to construct sequence alignments. As a result, target sequence data is transmitted off-host.
* **Privacy Option:** If transmitting sequence data off-host is a concern, configure Stage 4 complex predictions to run **Boltz-2 in single-sequence mode** or use a trusted server API.

---

### **Repository Layout**

```
.
├── config/
│   └── campaign.yaml            # Single source of truth — quality gates, paths, and tool pins
├── pipeline/                    # Modular Python library shared by notebook and Snakemake
│   ├── docker.py                # Container wrapper and volume/GPU allocation engine
│   ├── runlog.py                # File-backed transcript logging and event audit trails
│   ├── pdb_io.py                # PDB/mmCIF structural parsing utilities
│   ├── trb.py                   # RFdiffusion metadata extraction
│   ├── hotspots.py              # Geometric pocket analysis and hotspot validation
│   ├── metrics.py               # Statistical evaluation and quality scoring functions
│   ├── inputs.py                # Multi-engine manifest generators (YAML/FASTA)
│   ├── parsers.py               # Output parsers for RFdiffusion, MPNN, ESMFold, Boltz-2, ColabFold
│   ├── report.py                # Campaign registry class for sample tracking and gating
│   └── viz.py                   # Structural visualization and PyMOL alignment tools
├── docker/                      # Tool-specific Dockerfiles and container build scripts
│   ├── <tool>/Dockerfile        # Isolated image specification per deep learning engine
│   └── build_all.sh             # Sequential image build runner and GPU probe harness
├── scripts/                     # Instance environment setup and bootstrap utilities
│   ├── setup_docker.sh          # Non-root Docker privilege setup script
│   └── bootstrap_instance.sh    # Pre-flight checker and environment installer
├── notebooks/                   # Interactive prototyping and smoke-test notebooks
├── tests/                       # Unit and integration test suite (CPU-only, no GPU required)
├── inputs/                      # Target protein PDB structures and epitope specifications
├── results/                     # Campaign outputs, parsed metrics, and container runlogs
└── docs/                        # Architecture specs, setup guides, and pipeline documentation

```

---

### **Licensing**

This project is open-source software released under the [MIT License](https://www.google.com/search?q=LICENSE).

---

### **Acknowledgements**

This pipeline acts as a cloud-native orchestration framework. The underlying scientific models and foundational deep learning architectures belong to their respective authors and research groups:

* **RFdiffusion & SolubleMPNN:** Baker Laboratory, Institute for Protein Design, University of Washington ([Dauparas et al., 2022](https://www.science.org/doi/10.1126/science.add2187); [Watson et al., 2023](https://www.nature.com/articles/s41586-023-06415-8)).
* **ESMFold:** Meta AI FAIR Protein Team ([Lin et al., 2023](https://www.science.org/doi/10.1126/science.ade2574)).
* **Boltz-2:** Wohlwend et al., MIT & MIT-IBM Watson AI Lab ([Boltz Architecture](https://www.google.com/search?q=https://github.com/j337h/boltz)).
* **ColabFold:** Mirdita et al., Steinegger & Söding Labs ([Mirdita et al., 2022](https://www.nature.com/articles/s41592-022-01488-1)).