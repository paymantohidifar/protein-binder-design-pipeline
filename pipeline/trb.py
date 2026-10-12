"""RFdiffusion `.trb` (Trace/Trajectory) Metadata File Parameter Reference

The `.trb` file is a binary Python pickle dictionary output alongside each generated 
PDB file during an RFdiffusion trajectory. It contains run configurations, model confidence 
scores, structural/sequence inpainting masks, and index conversion arrays between 
the input target PDB, tensor space (0-indexed), and output PDB structures.

Nomenclature & Indexing Conventions:
  - `ref`: Reference / Input Target structure.
  - `hal`: Hallucinated / Generated Output structure.
  - `pdb_idx`: PDB-formatted residue identifiers, e.g., ('A', 34) or 'A34'.
  - `idx0`: Zero-based continuous integer indices (0, 1, ..., N-1) for PyTorch tensors.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import metrics, pdb_io
from .pdb_io import parse_label  # re-exported: trb.parse_label

TRB_PARAMETER_DEFINITIONS = {

    # RUNTIME & MODEL CONFIDENCE METRICS
    "config": {
        "type": "dict / Hydra DictConfig",
        "description": "Resolved runtime configuration flags (contig specifications, hotspots, model checkpoints, noise scales).",
        "developer_note": "Used for provenance tracking, logging, and audit trails."
    },
    "plddt": {
        "type": "numpy.ndarray / torch.Tensor (2D Float)",
        "description": "Per-timestep, per-residue pLDDT confidence from RFdiffusion. Shape (timesteps, residues) -- measured (50, 237) on results/01_backbones/design_0.trb -- on a 0-1 scale, NOT 0-100.",
        "developer_note": "Use TrbInfo.final_plddt(), which takes the LAST timestep and rescales to 0-100. np.mean(trb['plddt']) is wrong twice over: it averages across all 50 diffusion steps, and it returns ~0.998, so a `>= 80` test rejects every good design."
    },
    "device": {
        "type": "str",
        "description": "Compute device hardware string used during inference (e.g., 'cuda:0' or 'cpu').",
        "developer_note": "Debugging hardware allocation across multi-GPU clusters."
    },
    "time": {
        "type": "float",
        "description": "Total wall-clock time in seconds taken to complete the reverse diffusion trajectory.",
        "developer_note": "Useful for benchmarking compute overhead per design."
    },

    # CONTIG & RESIDUE MAPPING ARRAYS (GENERAL / MONOMER)
    "con_ref_pdb_idx": {
        "type": "list of tuples / str",
        "description": "PDB chain ID and residue number tuples of reference/motif residues held fixed from the input PDB.",
        "example": "[('A', 34), ('A', 35), ...]"
    },
    "con_hal_pdb_idx": {
        "type": "list of tuples / str",
        "description": "PDB chain ID and residue number assigned to the corresponding reference/motif residues in the output PDB.",
        "example": "[('A', 12), ('A', 13), ...]"
    },
    "con_ref_idx0": {
        "type": "numpy.ndarray (1D Int)",
        "description": "Zero-based tensor indices of reference/motif residues within the input target structure arrays.",
        "example": "array([33, 34, 35])"
    },
    "con_hal_idx0": {
        "type": "numpy.ndarray (1D Int)",
        "description": "Zero-based tensor indices showing where reference/motif residues land in the output design tensor.",
        "example": "array([11, 12, 13])",
        "developer_note": "POPULATED ONLY FOR MOTIF SCAFFOLDING. Empty in PPI/binder runs, where the fixed residues are in complex_con_hal_idx0 instead. Reading this unconditionally is the bug that blocked the campaign."
    },

    # COMPLEX & RECEPTOR/BINDER MAPPING ARRAYS (PPI / TARGET BINDER DESIGN)
    "complex_con_ref_pdb_idx": {
        "type": "list of tuples",
        "description": "PDB-style chain/residue mappings for all fixed residues across the full target-binder complex."
    },
    "complex_con_hal_pdb_idx": {
        "type": "list of tuples",
        "description": "Corresponding output PDB chain/residue mappings across the full target-binder complex."
    },
    "complex_con_ref_idx0": {
        "type": "numpy.ndarray (1D Int)",
        "description": "Zero-based tensor indices for all fixed residues across the full target-binder complex."
    },
    "complex_con_hal_idx0": {
        "type": "numpy.ndarray (1D Int)",
        "description": "Zero-based tensor indices for all fixed residues in the generated complex output, in the DESIGN TENSOR frame.",
        "developer_note": "The correct array for PPI/binder runs. Measured 87..236 on design_0 (the 150-residue target), matching mask_1d exactly."
    },
    "receptor_con_ref_pdb_idx": {
        "type": "list of tuples",
        "description": "PDB-style chain/residue mappings isolated strictly to the target/receptor protein."
    },
    "receptor_con_hal_pdb_idx": {
        "type": "list of tuples",
        "description": "Corresponding output PDB chain/residue mappings isolated strictly to the target/receptor protein."
    },
    "receptor_con_ref_idx0": {
        "type": "numpy.ndarray (1D Int)",
        "description": "Zero-based tensor indices isolated strictly to the target/receptor protein in the input."
    },
    "receptor_con_hal_idx0": {
        "type": "numpy.ndarray (1D Int)",
        "description": "Zero-based tensor indices isolated strictly to the target/receptor protein in the output, in the RECEPTOR'S OWN frame.",
        "developer_note": "NOT a substitute for complex_con_hal_idx0. Measured 0..149 on design_0 while the target actually sits at design positions 87..236, so using it silently addresses the binder plus part of the target. Same element count, different frame -- no error, wrong answer."
    },

    # INPAINTING & STRUCTURAL MASKS
    "inpaint_str": {
        "type": "list of bool / int",
        "description": "Structural Inpainting Mask. False/0 indicates backbone positions generated de novo; True/1 indicates positions structurally frozen from a template.",
        "developer_note": "Crucial for identifying binder vs target boundaries: np.where(trb['inpaint_str']) gives de novo positions."
    },
    "inpaint_seq": {
        "type": "list of bool / int",
        "description": "Sequence Inpainting Mask. False/0 indicates residue positions allowed to be redesigned; True/1 indicates native amino acids kept fixed.",
        "developer_note": "Used in downstream ProteinMPNN sequence generation to set redesign bias/fixed masks."
    },
    "sampled_mask": {
        "type": "list of bool / int",
        "description": "The explicit structural mask sampled by the internal contigmap module for variable-length contig specifications.",
        "example": "Sampled length variation for 'contigs=[10-30/A50-65/20-40]'"
    },
    "mask_1d": {
        "type": "numpy.ndarray / torch.Tensor (1D Bool)",
        "description": "Master 1D binary tensor fed into the RFdiffusion neural network uniting contig lengths, fixed motifs, de novo regions, and chain breaks.",
        "developer_note": "Internal architectural feature mask used during tensor self-attention."
    }
}

ResidueKey = tuple[str, int]

# Which key family describes the fixed (reference-derived) residues depends on
# what RFdiffusion was asked to do, so the family cannot be hardcoded.
_REFERENCE_FAMILIES = (
    ("complex_con_hal_idx0", "complex_con_hal_pdb_idx", "complex_con_ref_pdb_idx"),
    ("con_hal_idx0", "con_hal_pdb_idx", "con_ref_pdb_idx"),
)


class TrbError(RuntimeError):
    """A .trb was missing, unreadable, or inconsistent with its PDB."""


@dataclass
class TrbInfo:
    path: Path
    design_length: int
    reference_idx0: np.ndarray
    reference_design_keys: list[ResidueKey]
    reference_input_keys: list[ResidueKey]
    sampled_mask: list[str]
    plddt: np.ndarray
    config: dict

    @property
    def diffused_idx0(self) -> np.ndarray:
        """Design positions generated de novo (the complement of the motif)."""
        return np.setdiff1d(np.arange(self.design_length), self.reference_idx0)

    @property
    def hotspots(self) -> list[str] | None:
        """ppi.hotspot_res as RFdiffusion actually received it.

        None for runs that specified none (e.g. motif scaffolding). Reading this
        back catches a hotspot string that was silently dropped or mistyped.
        """
        ppi = self.config.get("ppi") or {}
        res = ppi.get("hotspot_res") if hasattr(ppi, "get") else None
        return [str(r) for r in res] if res else None

    @property
    def contigs(self) -> list[str] | None:
        contigmap = self.config.get("contigmap") or {}
        contigs = contigmap.get("contigs") if hasattr(contigmap, "get") else None
        return [str(c) for c in contigs] if contigs else None

    @property
    def input_pdb(self) -> str | None:
        inference = self.config.get("inference") or {}
        return inference.get("input_pdb") if hasattr(inference, "get") else None

    def final_plddt(self) -> np.ndarray:
        """Last-timestep per-residue pLDDT, normalised to 0-100.

        RFdiffusion writes 0-1 (measured: final-step mean 0.97 on the sample
        run), so this goes through the same normalisation as every other
        predictor's pLDDT.
        """
        if self.plddt.size == 0:
            raise TrbError(f"no pLDDT recorded in {self.path}")
        return metrics.normalise_plddt(self.plddt[-1])

    def input_to_design(self, labels: list[str]) -> list[str]:
        """Translate residue labels from input numbering to design numbering.

        The fix for RFdiffusion's renumbering. Raises on a label with no
        correspondence -- which means the residue was not carried into the
        design at all, and silently dropping it would understate how many
        hotspots a binder failed to engage.
        """
        mapping = dict(zip(self.reference_input_keys, self.reference_design_keys))
        out: list[str] = []
        for label in labels:
            key = parse_label(label)
            if key not in mapping:
                available = sorted(mapping)[:8]
                raise TrbError(
                    f"input residue {label} has no counterpart in the design "
                    f"({self.path.name}). It was not part of the reference motif. "
                    f"Mapped residues begin: {available}"
                )
            chain, resnum = mapping[key]
            out.append(f"{chain}{resnum}")
        return out


@dataclass
class ChainRoles:
    """Which design chain is the binder and which is the target."""

    binder_chain: str | None
    target_chain: str | None
    binder_idx0: np.ndarray
    target_idx0: np.ndarray
    is_binder_design: bool
    detail: str

    def require(self) -> tuple[str, str]:
        if not self.is_binder_design:
            raise TrbError(f"not a two-chain binder design: {self.detail}")
        return self.binder_chain, self.target_chain


def trb_path_for(design_pdb: str | Path) -> Path:
    path = Path(design_pdb).with_suffix(".trb")
    if not path.exists():
        raise TrbError(f"no .trb beside {design_pdb} (looked for {path})")
    return path


def read_trb(path: str | Path) -> TrbInfo:
    path = Path(path)
    try:
        with path.open("rb") as fh:
            raw = pickle.load(fh)
    except (OSError, pickle.UnpicklingError) as exc:
        raise TrbError(f"cannot read {path}: {exc}") from exc

    if "mask_1d" not in raw:
        raise TrbError(f"{path} lacks 'mask_1d'; keys present: {sorted(raw)}")

    design_length = len(raw["mask_1d"])
    mask_positions = np.where(np.asarray(raw["mask_1d"], dtype=bool))[0]
    idx_key, hal_key, ref_key = _select_reference_family(path, raw, mask_positions)

    return TrbInfo(
        path=path,
        design_length=design_length,
        reference_idx0=np.asarray(raw.get(idx_key, []), dtype=int),
        reference_design_keys=[(c, int(r)) for c, r in raw.get(hal_key, [])],
        reference_input_keys=[(c, int(r)) for c, r in raw.get(ref_key, [])],
        sampled_mask=[str(m) for m in raw.get("sampled_mask", [])],
        plddt=np.asarray(raw.get("plddt", np.empty((0, 0))), dtype=np.float64),
        config=raw.get("config", {}),
    )


def _select_reference_family(
    path: Path, raw: dict, mask_positions: np.ndarray
) -> tuple[str, str, str]:
    """Pick the key family describing this run's fixed residues.

    `mask_1d` marks exactly the reference-derived positions, so it doubles as
    the selector: the right family is the one whose design-frame indices match
    it. That makes the choice self-validating rather than inferred from the
    contig string, and it is why picking the wrong family (see
    `_REFERENCE_FAMILIES`) raises here instead of returning bad indices.
    """
    # A fully unconditional run fixes nothing: no reference residues, no mask.
    # Legitimate, so it resolves to the canonical family with empty arrays.
    if mask_positions.size == 0:
        return _REFERENCE_FAMILIES[-1]

    sizes = []
    for family in _REFERENCE_FAMILIES:
        idx_key = family[0]
        idx0 = np.asarray(raw.get(idx_key, []), dtype=int)
        sizes.append(f"{idx_key}={idx0.size}")
        if idx0.size and np.array_equal(np.sort(idx0), mask_positions):
            return family

    raise TrbError(
        f"{path}: no reference index family matches mask_1d "
        f"({mask_positions.size} fixed positions; {', '.join(sizes)}). "
        "The .trb format may have changed -- re-audit before trusting "
        "anything downstream. Note receptor_con_hal_idx0 is NOT a valid "
        "substitute: it is indexed in the receptor's frame, not the design's."
    )


def design_file_order(design_pdb: str | Path) -> list[ResidueKey]:
    """(chain, resnum) per residue in PDB file order.

    Position i corresponds to .trb index i
    """
    struct = pdb_io.read_pdb(design_pdb)
    return [
        (cid, r.seq_id)
        for cid in struct.chains
        for r in struct.chains[cid].residues
    ]


def classify_chains(trb: TrbInfo, design_pdb: str | Path) -> ChainRoles:
    """Decide which chain is binder and which is target, from the .trb."""
    
    order = design_file_order(design_pdb)
    if len(order) != trb.design_length:
        raise TrbError(
            f"{design_pdb} has {len(order)} residues but {trb.path.name} describes "
            f"{trb.design_length}. Mismatched design/metadata pair."
        )

    reference = set(trb.reference_idx0.tolist())
    per_chain: dict[str, dict[str, list[int]]] = {}
    for i, (chain, _) in enumerate(order):
        bucket = per_chain.setdefault(chain, {"reference": [], "diffused": []})
        bucket["reference" if i in reference else "diffused"].append(i)

    pure_reference = [c for c, b in per_chain.items() if b["reference"] and not b["diffused"]]
    pure_diffused = [c for c, b in per_chain.items() if b["diffused"] and not b["reference"]]
    mixed = [c for c, b in per_chain.items() if b["reference"] and b["diffused"]]

    if len(pure_reference) == 1 and len(pure_diffused) == 1:
        target, binder = pure_reference[0], pure_diffused[0]
        return ChainRoles(
            binder_chain=binder,
            target_chain=target,
            binder_idx0=np.asarray(per_chain[binder]["diffused"], dtype=int),
            target_idx0=np.asarray(per_chain[target]["reference"], dtype=int),
            is_binder_design=True,
            detail=(
                f"target={target} ({len(per_chain[target]['reference'])} reference res), "
                f"binder={binder} ({len(per_chain[binder]['diffused'])} diffused res)"
            ),
        )

    summary = ", ".join(
        f"{c}: {len(b['reference'])} ref / {len(b['diffused'])} diffused"
        for c, b in sorted(per_chain.items())
    )
    if mixed and len(per_chain) == 1:
        detail = (
            f"single chain mixing reference and diffused residues ({summary}); "
            "this is motif scaffolding, not binder design"
        )
    else:
        detail = f"cannot resolve binder/target roles from chains -- {summary}"

    return ChainRoles(
        binder_chain=None,
        target_chain=None,
        binder_idx0=trb.diffused_idx0,
        target_idx0=trb.reference_idx0,
        is_binder_design=False,
        detail=detail,
    )


def hotspots_on_design(design_pdb: str | Path) -> dict[str, str]:
    """Hotspot labels renumbered onto one RFdiffusion design: {'A59': 'B146'}.

    The target is emitted after the binder, so its residue numbers are shifted
    by the binder length -- which is sampled per design, so the shift differs
    between designs of one run (+87, +84, +92 ... across design_0-4). Taken
    from the .trb's own residue correspondence rather than computed from the
    shift, so it also holds if a contig ever puts the target first.
    """
    info = read_trb(trb_path_for(design_pdb))
    labels = info.hotspots or []
    return dict(zip(labels, info.input_to_design(labels)))


def describe(trb: TrbInfo, design_pdb: str | Path | None = None) -> str:
    """Human-readable summary for notebook checkpoints."""
    lines = [
        f"{trb.path.name}",
        f"  design length   : {trb.design_length}",
        f"  reference res   : {trb.reference_idx0.size}",
        f"  diffused res    : {trb.diffused_idx0.size}",
        f"  sampled mask    : {trb.sampled_mask}",
        f"  contigs         : {trb.contigs}",
        f"  hotspots        : {trb.hotspots or 'none specified'}",
        f"  input pdb       : {trb.input_pdb}",
    ]
    if trb.plddt.size:
        lines.append(f"  RFdiff pLDDT    : {trb.final_plddt().mean():.1f} (final step, 0-100)")
    if design_pdb:
        roles = classify_chains(trb, design_pdb)
        lines.append(f"  chain roles     : {roles.detail}")
    return "\n".join(lines)
