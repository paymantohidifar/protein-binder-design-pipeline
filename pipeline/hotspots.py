"""Hotspot candidate selection/evalutation for target preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

from . import pdb_io
from .pdb_io import Chain

# Theoretical maximum solvent-accessible surface area per residue type, A^2,
# from Tien et al. 2013 (PLoS ONE 8:e80635, Table 1). Reference data, used to
# turn absolute SASA into a fraction so a small residue and a large one can be
# compared on the same exposure scale.
MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLU": 223.0, "GLN": 225.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}

# Relative-SASA floor for calling a residue surface-exposed.
EXPOSURE_FLOOR = 0.15

# Residues whose side chains dominate protein-protein interfaces.
INTERFACE_ENRICHED = frozenset({"TRP", "TYR", "PHE", "LEU", "ILE", "MET", "ARG", "PRO"})

NEIGHBOUR_RADIUS = 10.0 # Radius of patch formed by neighboring candidate residues
BURIAL_CUTOFF = 20  
HOTSPOT_CONTACT_DISTANCE = 10.0 # Acceptable distance between Cb of target and binder (in Angstrum)
MAX_PATCH_DISTANCE = 15.0 # in Angstrum


@dataclass
class HotspotCandidate:
    residue_id: int
    residue_name: str
    exposure: float # Relative SASA (fraction of this residue type's theoretical maximum).
    interface_enriched: bool
    # Meaning depends on the producer: rank_candidates writes the Cb burial
    # count, contacts_from_complex the partner-contact count. Both are
    # neighbour counts within HOTSPOT_CONTACT_DISTANCE, of different things.
    neighbours: int = 0
    sasa: float = float("nan") # Absolute SASA in A^2, kept alongside the fraction for reporting.
    _chain: str = "A"

    @property
    def label(self) -> str:
        return f"{self._chain}{self.residue_id}"


def _cb_array(chain: Chain) -> tuple[np.ndarray, list[int], list[str]]:
    rows = [(r.cb, r.seq_id, r.name) for r in chain.residues if r.cb is not None]
    if not rows:
        raise ValueError(f"chain {chain.chain_id} has no Cb or Ca atoms")
    coords, ids, names = zip(*rows)
    return np.stack(coords), list(ids), list(names)


def compute_sasa(
    pdb_path: str | Path, chain_id: str, level: str = "R"
) -> dict[int, float]:
    """Per-residue solvent-accessible surface area in A^2, via Shrake-Rupley.

    SASA is computed on the whole model, not the isolated
    chain, so a residue buried by a partner chain reads as buried -- which is
    the physically meaningful answer for a complex.
    """
    structure = PDBParser(QUIET=True).get_structure("target", str(pdb_path))
    model = next(iter(structure))
    if chain_id not in model:
        raise KeyError(
            f"chain {chain_id!r} not in {pdb_path}; present: {[c.id for c in model]}"
        )
    ShrakeRupley().compute(model, level=level)
    return {
        residue.id[1]: float(residue.sasa)
        for residue in model[chain_id]
        if residue.id[0] == " "  # standard residues only, skip waters/hetero
    }


def relative_sasa(
    pdb_path: str | Path, chain_id: str, names: dict[int, str] | None = None
) -> dict[int, float]:
    """SASA as a fraction of the residue type's theoretical maximum.

    Residues whose type has no reference maximum (non-standard) are omitted
    rather than given a wrong denominator.
    """
    absolute = compute_sasa(pdb_path, chain_id)
    if names is None:
        structure = PDBParser(QUIET=True).get_structure("target", str(pdb_path))
        model = next(iter(structure))
        names = {r.id[1]: r.get_resname() for r in model[chain_id] if r.id[0] == " "}
    return _as_fraction(absolute, names)


def _as_fraction(
    absolute: dict[int, float], names: dict[int, str]
) -> dict[int, float]:
    """Divide absolute SASA by the residue type's theoretical maximum.

    Residues whose type has no reference maximum (non-standard) are omitted
    rather than given a wrong denominator.
    """
    return {
        rid: value / MAX_ASA[names[rid]]
        for rid, value in absolute.items()
        if names.get(rid) in MAX_ASA
    }


def chain_sasa(chain: Chain, context: list[Chain] | None = None) -> dict[int, float]:
    """Per-residue SASA in A^2 for a Chain already in memory.

    The in-memory counterpart of compute_sasa, for a Chain that may have no
    file of its own. `context` adds further chains to the model, so a residue
    buried by a partner reads as buried; with no context the chain is treated
    as isolated, which gives *unbound* exposure.

    That default is deliberate for hotspot work. A good epitope residue is one
    a binder could reach on the free target, so the question is how exposed it
    is before anything binds -- in a complex those same residues read as
    buried, which would rank the real interface last.
    """
    model = pdb_io.to_bio_model([chain, *(context or [])])
    ShrakeRupley().compute(model, level="R")
    # Positionally: to_bio_model round-trips a blank chain id to a space, so
    # the id is not a reliable key here. `chain` was passed first.
    return {
        residue.id[1]: float(residue.sasa)
        for residue in next(iter(model))
        if residue.id[0] == " "
    }


def chain_relative_sasa(
    chain: Chain, context: list[Chain] | None = None
) -> dict[int, float]:
    """chain_sasa as a fraction of each residue type's theoretical maximum."""
    names = {r.seq_id: r.name for r in chain.residues}
    return _as_fraction(chain_sasa(chain, context), names)


def neighbour_counts(chain: Chain, radius: float = NEIGHBOUR_RADIUS) -> dict[int, int]:
    """Cb neighbours within `radius` for each residue, excluding itself."""
    coords, ids, _ = _cb_array(chain)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    counts = (d <= radius).sum(axis=1) - 1
    return dict(zip(ids, counts.tolist()))


def rank_candidates(
    chain: Chain,
    radius: float = NEIGHBOUR_RADIUS,
    burial_cutoff: int = BURIAL_CUTOFF,
    exposure_floor: float = EXPOSURE_FLOOR,
) -> list[HotspotCandidate]:
    """Shortlist non-buried residues, best hotspot candidates first.

    Sorted by interface-enriched identity first, then relative SASA within
    that class. Epitope choice is a judgement call, and the ranking only puts
    plausible residues in front of the reviewer.

    Two independent exposure filters, because they fail differently: the Cb
    neighbour count (`burial_cutoff`) measures local packing density and is
    blind to residue size and side-chain direction, while relative SASA
    (`exposure_floor`) is per-residue-type normalised and absolute -- 0.15
    means the same thing on any target, where a neighbour count does not.
    Ranking uses relative SASA alone.
    """
    coords, ids, names = _cb_array(chain)
    counts = neighbour_counts(chain, radius)
    absolute = chain_sasa(chain)
    relative = chain_relative_sasa(chain)

    candidates = [
        HotspotCandidate(
            residue_id=rid,
            residue_name=name,
            neighbours=counts[rid],
            exposure=relative[rid],
            sasa=absolute[rid],
            interface_enriched=name in INTERFACE_ENRICHED,
            _chain=chain.chain_id,
        )
        for rid, name in zip(ids, names)
        # A residue absent from `relative` is non-standard, so it has no
        # reference maximum and no meaningful exposure to rank on.
        if counts[rid] <= burial_cutoff and relative.get(rid, 0.0) >= exposure_floor
    ]
    candidates.sort(key=lambda c: (not c.interface_enriched, -c.exposure, c.residue_id))
    return candidates


def select_patch(
    chain: Chain,
    n: int = 3,
    max_cb_distance: float = MAX_PATCH_DISTANCE,
    radius: float = NEIGHBOUR_RADIUS,
    burial_cutoff: int = BURIAL_CUTOFF,
    exposure_floor: float = EXPOSURE_FLOOR,
) -> list[HotspotCandidate]:
    """Pick `n` mutually close, non-buried residues forming one epitope patch.

    Greedy: seed on the top-ranked candidate, then add the next-best candidate
    still within `max_cb_distance` of every member already chosen, so the patch
    describes one site a single binder can engage.
    """
    if not 3 <= n <= 6:
        raise ValueError(f"RFdiffusion recommends 3-6 hotspots, got {n}")

    candidates = rank_candidates(chain, radius, burial_cutoff, exposure_floor)
    if len(candidates) < n:
        raise ValueError(
            f"only {len(candidates)} exposed residues at burial_cutoff="
            f"{burial_cutoff}, exposure_floor={exposure_floor}; relax either"
        )

    cb = {r.seq_id: r.cb for r in chain.residues if r.cb is not None}
    chosen = [candidates[0]]
    for cand in candidates[1:]:
        if len(chosen) == n:
            break
        if all(
            np.linalg.norm(cb[cand.residue_id] - cb[c.residue_id]) <= max_cb_distance
            for c in chosen
        ):
            chosen.append(cand)

    if len(chosen) < n:
        raise ValueError(
            f"could not assemble {n} residues within {max_cb_distance} A of each "
            f"other (got {len(chosen)}); raise max_cb_distance"
        )
    return chosen


def _labelled_cb(chain: Chain, labels: list[str]) -> np.ndarray:
    ids = [int("".join(ch for ch in lab if ch.isdigit())) for lab in labels]
    cb = {r.seq_id: r.cb for r in chain.residues if r.cb is not None}
    missing = [i for i in ids if i not in cb]
    if missing:
        raise KeyError(f"residues absent from chain {chain.chain_id}: {missing}")
    return np.stack([cb[i] for i in ids])


def contacts_from_complex(
    target: Chain,
    partner: Chain,
    cutoff: float = HOTSPOT_CONTACT_DISTANCE,
) -> list[HotspotCandidate]:
    """Hotspots per the RFdiffusion definition: target residues within `cutoff`
    Cb distance of the partner chain.

    This is the measurable form of the definition, requiring a target that
    arrives already bound to something. Candidates are returned most-contacted
    first, since a residue contacting many partner residues sits deeper in the
    interface.

    Does not apply to a bare target (single chain, no partner) -- there the binder 
    does not exist and hotspots must come from prior knowledge. 
    Use verify_hotspot_contacts() after design instead.
    """
    t_coords, t_ids, t_names = _cb_array(target)
    p_coords, _, _ = _cb_array(partner)

    d = np.linalg.norm(t_coords[:, None, :] - p_coords[None, :, :], axis=-1)
    contact_counts = (d <= cutoff).sum(axis=1)
    # Unbound exposure: the target alone, with the partner left out of the SASA
    # model on purpose. Every residue reported here is by definition near the
    # partner, so in-complex SASA would read ~0 for exactly the residues that
    # matter and the number would carry no information.
    absolute = chain_sasa(target)
    relative = chain_relative_sasa(target)

    candidates = [
        HotspotCandidate(
            residue_id=rid,
            residue_name=name,
            neighbours=int(count),
            exposure=relative.get(rid, float("nan")),
            sasa=absolute[rid],
            interface_enriched=name in INTERFACE_ENRICHED,
            _chain=target.chain_id,
        )
        for rid, name, count in zip(t_ids, t_names, contact_counts)
        if count > 0
    ]
    # `neighbours` carries the partner-contact count here, not the burial count.
    candidates.sort(key=lambda c: (-c.neighbours, c.residue_id))
    return candidates


def verify_hotspot_contacts(
    design_target: Chain,
    design_binder: Chain,
    labels: list[str],
    cutoff: float = HOTSPOT_CONTACT_DISTANCE,
) -> dict:
    """Did the generated binder actually engage the hotspots it was given?

    Because a hotspot is defined as a target residue within `cutoff` Cb of the 
    binder, the definition becomes checkable once RFdiffusion has produced a 
    binder -- and a design whose binder never comes within 10 A of a requested 
    hotspot has ignored its conditioning, regardless of how good its scRMSD later looks.

    Cheap enough to run on every backbone, and it catches a silently
    mis-specified hotspot string (wrong chain, wrong numbering) immediately
    rather than after the whole funnel has run.
    """
    hotspot_cb = _labelled_cb(design_target, labels)
    binder_cb, _, _ = _cb_array(design_binder)

    d = np.linalg.norm(hotspot_cb[:, None, :] - binder_cb[None, :, :], axis=-1)
    min_distance = d.min(axis=1)
    engaged = min_distance <= cutoff

    return {
        "hotspots": list(labels),
        "min_cb_distance": {lab: round(float(v), 2) for lab, v in zip(labels, min_distance)},
        "engaged": {lab: bool(v) for lab, v in zip(labels, engaged)},
        "n_engaged": int(engaged.sum()),
        "all_engaged": bool(engaged.all()),
        "cutoff": cutoff,
    }


def verify_design_hotspots(
    design_pdb: str | Path,
    labels: list[str] | None = None,
    cutoff: float = HOTSPOT_CONTACT_DISTANCE,
) -> dict:
    """Step 1 QC on an RFdiffusion design, taking everything from its .trb.

    Resolves three things that must not be guessed, because each failure mode
    produces a plausible number rather than an error:

      - which chain is the binder and which the target (contig order decides,
        and for `[A1-150/0 70-100]` the target is chain A, not the binder);
      - the hotspots RFdiffusion actually received (read back from its config,
        so a dropped or mistyped hotspot string is caught);
      - the input -> design residue renumbering, since hotspots are written in
        input numbering and RFdiffusion renumbers the motif.

    `labels` overrides the hotspots recorded in the .trb, and is still mapped
    through the renumbering. Returns the verify_hotspot_contacts dict plus the
    resolved context.
    """
    from . import trb as trb_mod

    info = trb_mod.read_trb(trb_mod.trb_path_for(design_pdb))
    roles = trb_mod.classify_chains(info, design_pdb)
    binder_chain, target_chain = roles.require()

    requested = labels if labels is not None else info.hotspots
    if not requested:
        return {
            "skipped": "no hotspots recorded in .trb and none supplied",
            "chain_roles": roles.detail,
        }

    design_labels = info.input_to_design(requested)
    struct = pdb_io.read_pdb(design_pdb)
    result = verify_hotspot_contacts(
        struct[target_chain], struct[binder_chain], design_labels, cutoff
    )
    result.update(
        {
            "requested_hotspots": list(requested),
            "design_hotspots": design_labels,
            "binder_chain": binder_chain,
            "target_chain": target_chain,
            "chain_roles": roles.detail,
        }
    )
    return result


def patch_distances(chain: Chain, labels: list[str]) -> np.ndarray:
    """Pairwise Cb-Cb distances for an explicit hotspot set, e.g. ['A59','A83'].

    Used to verify a hand-specified set really forms one patch before spending
    GPU time on it. Note this is the spread among hotspots.
    """
    coords = _labelled_cb(chain, labels)
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
