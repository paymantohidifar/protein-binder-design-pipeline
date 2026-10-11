"""Structure I/O for the pipeline, backed by Bio.PDB.

Parsing and writing are delegated to Biopython (PDBParser, PDBIO). What lives
here is the thin adapter plus the pipeline-specific accessors that Bio.PDB has
no opinion about: Cb with a glycine fallback, backbone-complete coordinate
arrays, per-residue pLDDT from the B-factor column, numbering-gap detection.

An earlier version parsed the fixed-column format by hand. It was measured
against Bio.PDB on every defect it claimed to need custom handling for --
missing residues, blank chain ids, trajectory MODEL records, altlocs,
B-factors, MSE as HETATM -- and matched on all of them, while its writer
emitted off-spec atom-name columns. It is kept, annotated, at
reference/pdb_io_handrolled.py as teaching material.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.Atom import Atom as BioAtom
from Bio.PDB.Chain import Chain as BioChain
from Bio.PDB.Model import Model as BioModel
from Bio.PDB.Residue import Residue as BioResidue
from Bio.PDB.Structure import Structure as BioStructure

BACKBONE_ATOMS = ("N", "CA", "C", "O")

# Chain ID used when column 22 is blank. Some RFdiffusion intermediates and
# hand-edited PDBs omit it; silently dropping those atoms would corrupt every
# downstream RMSD, so they are collected under an explicit sentinel instead.
DEFAULT_CHAIN = "_"


@dataclass
class Residue:
    seq_id: int
    name: str
    insertion: str = ""
    atoms: dict[str, np.ndarray] = field(default_factory=dict)
    # Per-atom B-factor column. Structure predictors overload this with pLDDT,
    # which makes it the one pLDDT source every tool in the funnel agrees on --
    # more reliable than each tool's own JSON schema. See parsers.plddt_from_pdb.
    bfactors: dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> tuple[int, str]:
        return (self.seq_id, self.insertion)

    @property
    def ca(self) -> np.ndarray | None:
        return self.atoms.get("CA")

    @property
    def plddt(self) -> float | None:
        """Residue pLDDT read from the B-factor column.

        Prefers Cα: predictors write one pLDDT per residue, replicated across
        that residue's atoms, so Cα is representative and always present in a
        usable model.

        CAUTION: Bio.PDB defaults a missing B-factor column to 0.0 rather than
        reporting its absence, so this returns 0.0 for a file that carries no
        pLDDT at all. Callers that need to tell "no pLDDT" from "pLDDT of zero"
        must check for an all-zero column -- see parsers.plddt_from_pdb.
        """
        if "CA" in self.bfactors:
            return self.bfactors["CA"]
        return next(iter(self.bfactors.values()), None)

    @property
    def cb(self) -> np.ndarray | None:
        """Cβ, falling back to Cα for glycine and for residues with no Cβ."""
        return self.atoms.get("CB", self.atoms.get("CA"))

    def backbone(self) -> np.ndarray | None:
        """(4, 3) N/CA/C/O array, or None if any backbone atom is absent."""
        try:
            return np.stack([self.atoms[a] for a in BACKBONE_ATOMS])
        except KeyError:
            return None


@dataclass
class Chain:
    chain_id: str
    residues: list[Residue] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.residues)

    @property
    def had_blank_id(self) -> bool:
        return self.chain_id == DEFAULT_CHAIN

    def residue_ids(self) -> list[int]:
        return [r.seq_id for r in self.residues]

    def gaps(self) -> list[tuple[int, int]]:
        """Breaks in residue numbering, as (last_present, next_present) pairs.

        Numbering gaps are the usual signal of unmodelled residues. They do not
        prove a physical chain break, so callers decide what to do about them.
        """
        ids = self.residue_ids()
        return [(a, b) for a, b in zip(ids, ids[1:]) if b != a + 1]

    def subset(self, start: int, end: int) -> "Chain":
        """Residues with seq_id in [start, end]; missing ones are simply absent."""
        return Chain(self.chain_id, [r for r in self.residues if start <= r.seq_id <= end])

    def sequence(self, unknown: str = "X") -> str:
        """One-letter sequence in residue order.

        Non-standard residues become `unknown` rather than being dropped: a
        silently shortened sequence would misalign every per-residue metric
        downstream.
        """
        return "".join(THREE_TO_ONE.get(r.name, unknown) for r in self.residues)

    def plddt_array(self) -> np.ndarray:
        """Per-residue pLDDT in chain order; NaN where the column was absent."""
        return np.array(
            [np.nan if (v := r.plddt) is None else v for r in self.residues],
            dtype=np.float64,
        )

    def ca_coords(self) -> tuple[np.ndarray, list[tuple[int, str]]]:
        """Cα coordinates and their residue keys, skipping Cα-less residues."""
        pairs = [(r.ca, r.key) for r in self.residues if r.ca is not None]
        if not pairs:
            return np.zeros((0, 3)), []
        coords, keys = zip(*pairs)
        return np.stack(coords), list(keys)

    def backbone_coords(self) -> tuple[np.ndarray, list[tuple[int, str]]]:
        """(N, 4, 3) backbone coordinates and keys, skipping incomplete residues."""
        pairs = [(bb, r.key) for r in self.residues if (bb := r.backbone()) is not None]
        if not pairs:
            return np.zeros((0, 4, 3)), []
        coords, keys = zip(*pairs)
        return np.stack(coords), list(keys)


@dataclass
class Structure:
    path: Path | None
    chains: dict[str, Chain] = field(default_factory=dict)

    def __getitem__(self, chain_id: str) -> Chain:
        if chain_id not in self.chains:
            raise KeyError(
                f"chain {chain_id!r} not in {self.path} "
                f"(present: {sorted(self.chains) or 'none'})"
            )
        return self.chains[chain_id]

    @property
    def chain_ids(self) -> list[str]:
        return sorted(self.chains)

    def only_chain(self) -> Chain:
        """The single chain, for files whose chain ID cannot be relied upon."""
        if len(self.chains) != 1:
            raise ValueError(f"{self.path} has {len(self.chains)} chains, expected 1")
        return next(iter(self.chains.values()))


# Three-letter to one-letter residue codes. MSE (selenomethionine) maps to M
# because it is a methionine analogue and predictors expect M.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}


def parse_label(label: str) -> tuple[str, int]:
    """'A59' -> ('A', 59).

    The residue-label format RFdiffusion uses for ppi.hotspot_res, and the one
    the notebook and PyMOL scripts pass around. Lives here rather than in any
    single consumer because trb, hotspots and viz all need it.
    """
    label = label.strip()
    chain = "".join(ch for ch in label if ch.isalpha())
    digits = "".join(ch for ch in label if ch.isdigit())
    if not chain or not digits:
        raise ValueError(f"cannot parse residue label {label!r}; expected e.g. 'A59'")
    return chain, int(digits)


# Non-standard residues carried through despite arriving as HETATM. MSE is a
# methionine analogue with real backbone geometry; dropping it would leave a
# hole in the chain and silently shift every per-residue metric after it.
KEEP_HETERO = frozenset({"MSE"})


class StructureError(ValueError):
    """A structure file could not be parsed.

    Subclasses ValueError so the per-candidate handlers in pipeline.parsers
    already catch it: one corrupt design should cost that candidate, not abort
    the campaign.
    """


def read_pdb(path: str | Path, model: int = 1) -> Structure:
    """Parse a PDB file into the pipeline's Structure/Chain/Residue types.

    `model` is 1-based. RFdiffusion trajectory files concatenate every
    denoising step as a separate MODEL, so taking all of them would stack
    timesteps into one chain.

    Raises StructureError on a file Bio.PDB cannot parse. Note that PERMISSIVE
    mode does NOT cover malformed coordinates -- Biopython aborts on those by
    design, rather than inventing a position, so a truncated or corrupt file
    costs the whole design rather than one atom. That is the right call; the
    wrapper exists so it costs one *candidate* and not the campaign.
    """
    path = Path(path)
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    try:
        bio_structure = parser.get_structure(path.stem, str(path))
    except PDBConstructionException as exc:
        raise StructureError(f"cannot parse {path}: {exc}") from exc

    models = list(bio_structure)
    if not models:
        return Structure(path=path)
    if not 1 <= model <= len(models):
        raise ValueError(
            f"{path} has {len(models)} model(s); model={model} requested"
        )
    bio_model = models[model - 1]

    struct = Structure(path=path)
    for bio_chain in bio_model:
        # Bio.PDB represents an absent chain id as a single space; the pipeline
        # uses an explicit sentinel so it can be addressed and reported.
        chain_id = bio_chain.id.strip() or DEFAULT_CHAIN
        chain = struct.chains.setdefault(chain_id, Chain(chain_id))

        for bio_residue in bio_chain:
            hetflag, seq_id, icode = bio_residue.id
            if hetflag.strip() and bio_residue.get_resname() not in KEEP_HETERO:
                continue  # waters, ions, ligands

            residue = Residue(
                seq_id=int(seq_id),
                name=bio_residue.get_resname(),
                insertion=icode.strip(),
            )
            # Iterating a residue yields the selected conformer of a disordered
            # atom, so alternate locations resolve to one position without
            # extra handling.
            for atom in bio_residue:
                name = atom.get_name()
                residue.atoms.setdefault(name, np.asarray(atom.get_coord(), dtype=np.float64))
                bfactor = atom.get_bfactor()
                if bfactor is not None:
                    residue.bfactors.setdefault(name, float(bfactor))

            if residue.atoms:
                chain.residues.append(residue)

    for chain in struct.chains.values():
        chain.residues.sort(key=lambda r: r.key)
    return struct


def _atom_fullname(name: str) -> str:
    """Atom name padded into PDB columns 13-16.

    The format right-justifies the element symbol in columns 13-14, so a name
    of three characters or fewer needs a leading space; a four-character name
    fills the field. Getting this wrong produces files that lenient readers
    accept via the element column and strict readers misparse -- the bug in
    reference/pdb_io_handrolled.py.
    """
    return name if len(name) >= 4 else f" {name:<3s}"


def write_pdb(chains: list[Chain], path: str | Path) -> Path:
    """Write chains out through Bio.PDB's PDBIO.

    Delegated rather than formatted by hand so the column layout -- atom-name
    justification above all -- is the library's problem.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bio_structure = BioStructure("out")
    bio_model = BioModel(0)
    bio_structure.add(bio_model)

    serial = 1
    for chain in chains:
        # Round-trip the sentinel back to the blank id the file had.
        bio_chain = BioChain(" " if chain.had_blank_id else chain.chain_id)
        bio_model.add(bio_chain)
        for res in chain.residues:
            bio_residue = BioResidue(
                (" ", res.seq_id, res.insertion or " "), res.name, ""
            )
            bio_chain.add(bio_residue)
            for name, xyz in res.atoms.items():
                bio_residue.add(
                    BioAtom(
                        name=name,
                        coord=np.asarray(xyz, dtype=np.float32),
                        bfactor=res.bfactors.get(name, 0.0),
                        occupancy=1.0,
                        altloc=" ",
                        fullname=_atom_fullname(name),
                        serial_number=serial,
                        # First character of the atom name. Correct for protein
                        # atoms (N, CA, C, O, CB -> N, C, C, O, C); it would be
                        # wrong for metals such as CA (calcium), which this
                        # pipeline never writes.
                        element=name[0],
                    )
                )
                serial += 1

    io = PDBIO()
    io.set_structure(bio_structure)
    io.save(str(path))
    return path


def renumber(chain: Chain, start: int = 1) -> Chain:
    """Contiguously renumber from `start`, collapsing numbering gaps."""
    out = Chain(chain.chain_id)
    for i, res in enumerate(chain.residues):
        out.residues.append(
            Residue(
                seq_id=start + i,
                name=res.name,
                insertion="",
                atoms=dict(res.atoms),
                bfactors=dict(res.bfactors),
            )
        )
    return out


def describe(struct: Structure) -> str:
    """One-line-per-chain summary for notebook sanity checks."""
    lines = [f"{struct.path}"]
    for cid in struct.chain_ids:
        chain = struct.chains[cid]
        ids = chain.residue_ids()
        gaps = chain.gaps()
        label = "(blank in file)" if chain.had_blank_id else ""
        span = f"{ids[0]}-{ids[-1]}" if ids else "empty"
        lines.append(
            f"  chain {cid}{label}: {len(chain)} res, span {span}, "
            f"{len(gaps)} numbering gap(s)"
            + (f" {gaps}" if gaps else "")
        )
    return "\n".join(lines)
