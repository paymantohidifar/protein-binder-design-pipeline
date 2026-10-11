"""Filter metrics for the design pipeline.

The definitions used here, fixed once so every stage agrees:

scRMSD
    Self-consistency RMSD. Binder backbone only, predicted monomer vs. the
    RFdiffusion design pose, after superposing the *binder alone*. Measures
    whether the designed sequence folds back to its intended backbone.

binder_rmsd_to_design
    Implemented as the standard binder-design quantity: superpose on
    *target* Ca, then RMSD over *binder* backbone. This measures whether the
    predicted complex reproduces the intended binding mode, which is what the
    filter is for.

pae_interaction
    Mean of BOTH off-diagonal inter-chain PAE blocks -- binder->target and
    target->binder, averaged together. Not the full-matrix mean, which is
    dominated by the intra-chain blocks and would make a bad complex look good.

plddt_mean
    Mean pLDDT over binder residues only, on a 0-100 scale. Predictors differ:
    ESMFold reports 0-1, ColabFold 0-100. Normalised on ingest.
"""

from __future__ import annotations

import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer

from .pdb_io import Chain


def kabsch_rmsd(mobile: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Optimal-superposition RMSD between two (N, 3) point sets."""
    
    mobile = np.asarray(mobile, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if mobile.shape != target.shape:
        raise ValueError(f"shape mismatch: {mobile.shape} vs {target.shape}")
    if mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(f"expected (N, 3) coordinates, got {mobile.shape}")
    if mobile.shape[0] < 3:
        raise ValueError(f"need >= 3 points to superpose, got {mobile.shape[0]}")

    sup = SVDSuperimposer()
    sup.set(target, mobile)
    sup.run()
    rotation, translation = sup.get_rotran()
    return float(sup.get_rms()), rotation, translation


def apply_transform(coords: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return coords @ rotation + translation


def _paired_backbone(a: Chain, b: Chain) -> tuple[np.ndarray, np.ndarray, int]:
    """Backbone coords for residues present and complete in both chains.

    Pairs positionally, not by residue number: a predicted monomer is numbered
    from 1 while the design pose keeps its original numbering, so matching on
    seq_id would pair nothing. Positional pairing is only valid because both
    chains describe the same sequence in the same order -- enforced by the
    length check.
    """
    coords_a, keys_a = a.backbone_coords()
    coords_b, keys_b = b.backbone_coords()
    if len(keys_a) != len(keys_b):
        raise ValueError(
            f"cannot pair residues positionally: {len(keys_a)} complete residues "
            f"vs {len(keys_b)}. Incomplete backbone in one structure."
        )
    return coords_a, coords_b, len(keys_a)


def scrmsd(predicted: Chain, design: Chain) -> float:
    """Binder backbone self-consistency RMSD, binder superposed on itself."""
    pred, des, _ = _paired_backbone(predicted, design)
    rmsd, _, _ = kabsch_rmsd(pred.reshape(-1, 3), des.reshape(-1, 3))
    return rmsd


def binder_rmsd_to_design(
    predicted_binder: Chain,
    predicted_target: Chain,
    design_binder: Chain,
    design_target: Chain,
) -> float:
    """Binding-mode agreement: superpose on target Ca, measure binder backbone.

    Deliberately does NOT superpose the binder -- that would discard exactly the
    rigid-body placement this metric exists to check.
    """
    pred_t, keys_pt = predicted_target.ca_coords()
    des_t, keys_dt = design_target.ca_coords()
    if len(keys_pt) != len(keys_dt):
        raise ValueError(
            f"target chain mismatch: {len(keys_pt)} Ca vs {len(keys_dt)}. "
            "Target coordinates should be fixed across the pipeline."
        )
    _, rotation, translation = kabsch_rmsd(pred_t, des_t)

    pred_b, des_b, n = _paired_backbone(predicted_binder, design_binder)
    moved = apply_transform(pred_b.reshape(-1, 3), rotation, translation)
    diff = moved - des_b.reshape(-1, 3)
    return float(np.sqrt((diff**2).sum() / len(diff)))


def pae_interaction(
    pae: np.ndarray,
    binder_idx: np.ndarray | list[int],
    target_idx: np.ndarray | list[int],
) -> float:
    """Mean inter-chain PAE, averaging both off-diagonal blocks.

    `pae[i, j]` is the expected positional error of residue j when the
    prediction is aligned on residue i, so the matrix is asymmetric and both
    blocks carry distinct information.
    """
    pae = np.asarray(pae, dtype=np.float64)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError(f"PAE must be square, got shape {pae.shape}")

    b = np.asarray(binder_idx, dtype=int)
    t = np.asarray(target_idx, dtype=int)
    if b.size == 0 or t.size == 0:
        raise ValueError("binder and target index sets must both be non-empty")
    overlap = np.intersect1d(b, t)
    if overlap.size:
        raise ValueError(f"binder and target indices overlap at {overlap[:5]}")
    n = pae.shape[0]
    for name, idx in (("binder", b), ("target", t)):
        if idx.max() >= n or idx.min() < 0:
            raise ValueError(f"{name} indices out of range for {n}x{n} PAE matrix")

    binder_to_target = pae[np.ix_(b, t)].mean()
    target_to_binder = pae[np.ix_(t, b)].mean()
    return float((binder_to_target + target_to_binder) / 2.0)


def normalise_plddt(values: np.ndarray | list[float]) -> np.ndarray:
    """Put pLDDT on a 0-100 scale regardless of the predictor's convention.

    ESMFold emits 0-1, ColabFold 0-100. Comparing the two unscaled would make
    every ESMFold design fail a `plddt >= 80` gate.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size and arr.max() <= 1.0:
        return arr * 100.0
    return arr


def plddt_mean(values: np.ndarray | list[float], idx: np.ndarray | list[int] | None = None) -> float:
    """Mean pLDDT, over `idx` (binder residues) when given."""
    arr = normalise_plddt(values)
    if idx is not None:
        arr = arr[np.asarray(idx, dtype=int)]
    if arr.size == 0:
        raise ValueError("no pLDDT values to average")
    return float(arr.mean())


def passes(record: dict, thresholds: dict) -> tuple[bool, list[str]]:
    """Evaluate a candidate against a gate block from config/campaign.yaml.

    Returns (passed, reasons_failed). A metric that is absent or NaN counts as a
    failure and is reported distinctly from a threshold breach -- at smoke-test
    scale a NaN means a broken parser, not a bad design.
    """
    checks = [
        ("plddt", "plddt_min", lambda v, t: v >= t, ">="),
        ("scrmsd", "scrmsd_max", lambda v, t: v <= t, "<="),
        ("iptm", "iptm_min", lambda v, t: v >= t, ">="),
        ("pae_interaction", "pae_interaction_max", lambda v, t: v <= t, "<="),
        ("binder_rmsd_to_design", "binder_rmsd_to_design_max", lambda v, t: v <= t, "<="),
    ]
    reasons: list[str] = []
    for metric, key, test, op in checks:
        if key not in thresholds:
            continue
        value = record.get(metric)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            reasons.append(f"{metric}=MISSING (parser failure, not a design failure)")
        elif not test(value, thresholds[key]):
            reasons.append(f"{metric}={value:.3f} fails {op} {thresholds[key]}")
    return not reasons, reasons
