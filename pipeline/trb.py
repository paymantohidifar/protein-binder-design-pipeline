"""RFdiffusion .trb metadata -- the authoritative description of a design.

Every design PDB ships a .trb pickle beside it. Rather than infer which output
residues are target and which are de novo binder, which chain is which, or what
hotspots were requested, read what RFdiffusion recorded.

Verified against outputs/motifscaffolding_0.{trb,pdb} in this repo:

  con_hal_idx0     0-based indices into the DESIGN, in PDB file order
  con_hal_pdb_idx  (chain, resnum) of those positions in the design
  con_ref_pdb_idx  (chain, resnum) of the SAME positions in the input PDB
  mask_1d          bool[L]; True exactly where con_hal_idx0 points
  plddt            float[T, L], RFdiffusion's own per-timestep pLDDT on 0-1
  config           the full hydra config, including ppi.hotspot_res

The con_ref -> con_hal pairing matters more than it looks: RFdiffusion
RENUMBERS. In the sample run, input A163-181 became design A30-48. Any residue
selection expressed in input numbering -- hotspots above all -- must be mapped
through this correspondence before it is applied to a design PDB, or it silently
addresses the wrong residues.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import metrics, pdb_io
from .pdb_io import parse_label  # re-exported: trb.parse_label

ResidueKey = tuple[str, int]


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
        predictor's pLDDT. A free Step 1 quality signal -- not a gate metric,
        since the gates are defined on ESMFold and Boltz-2.
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

    missing = [k for k in ("con_hal_idx0", "con_hal_pdb_idx", "mask_1d") if k not in raw]
    if missing:
        raise TrbError(f"{path} lacks {missing}; keys present: {sorted(raw)}")

    design_length = len(raw["mask_1d"])
    reference_idx0 = np.asarray(raw["con_hal_idx0"], dtype=int)

    # mask_1d should mark exactly the reference-derived positions. Verified true
    # on the sample run; if it ever diverges the file's semantics have changed
    # and nothing downstream should be trusted.
    mask_positions = np.where(np.asarray(raw["mask_1d"], dtype=bool))[0]
    if not np.array_equal(np.sort(reference_idx0), mask_positions):
        raise TrbError(
            f"{path}: mask_1d and con_hal_idx0 disagree "
            f"({mask_positions.size} vs {reference_idx0.size} positions). "
            "The .trb format may have changed."
        )

    return TrbInfo(
        path=path,
        design_length=design_length,
        reference_idx0=reference_idx0,
        reference_design_keys=[(c, int(r)) for c, r in raw["con_hal_pdb_idx"]],
        reference_input_keys=[(c, int(r)) for c, r in raw.get("con_ref_pdb_idx", [])],
        sampled_mask=[str(m) for m in raw.get("sampled_mask", [])],
        plddt=np.asarray(raw.get("plddt", np.empty((0, 0))), dtype=np.float64),
        config=raw.get("config", {}),
    )


def design_file_order(design_pdb: str | Path) -> list[ResidueKey]:
    """(chain, resnum) per residue in PDB file order.

    Position i corresponds to .trb index i -- verified against
    outputs/motifscaffolding_0.{pdb,trb}.
    """
    struct = pdb_io.read_pdb(design_pdb)
    return [
        (cid, r.seq_id)
        for cid in struct.chains
        for r in struct.chains[cid].residues
    ]


def classify_chains(trb: TrbInfo, design_pdb: str | Path) -> ChainRoles:
    """Decide which chain is binder and which is target, from the .trb.

    The target is the chain made of reference-derived residues; the binder is
    the chain made of diffused residues. Reported rather than assumed, because
    chain order follows contig order: `[A1-150/0 70-100]` lists the target
    first, so the target is chain A and the binder chain B -- the opposite of
    what a 'binder is chain A' convention would suggest.

    Motif scaffolding puts both in one chain. That is not a binder design, and
    this returns is_binder_design=False rather than inventing a split.
    """
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
