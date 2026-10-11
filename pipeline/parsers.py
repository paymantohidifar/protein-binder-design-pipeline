"""Predictor output -> one flat per-candidate record.

IMPORTANT: the file layouts and JSON keys below are written from documented
upstream formats, NOT verified against real output on this machine -- none of
these tools are installed here. Output schemas drift between versions, so every
parser:

  - globs over several candidate filename patterns rather than one,
  - reports what it DID find when a lookup fails, so a schema change is a
    legible error rather than a silent NaN,
  - prefers the B-factor column for pLDDT, which is the one convention all
    three predictors share, over each tool's own JSON schema.

Verifying these against real output is the primary purpose of the Phase A smoke
test. A NaN column in the summary CSV means a parser needs fixing here; it does
not mean the designs were bad.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import metrics, pdb_io


class ParseError(RuntimeError):
    """An expected output file or field was missing or malformed."""


def _find(root: Path, patterns: list[str], what: str) -> Path:
    """First file matching any pattern, with a diagnostic listing on failure."""
    root = Path(root)
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    present = sorted(p.name for p in root.rglob("*") if p.is_file())[:25]
    raise ParseError(
        f"no {what} in {root}\n  tried patterns: {patterns}\n  files present: {present or 'none'}"
    )


def _load_json(path: Path) -> dict:
    try:
        with path.open() as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ParseError(f"cannot read JSON {path}: {exc}") from exc


def _first_key(data: dict, keys: list[str], what: str, source: Path) -> object:
    for key in keys:
        if key in data:
            return data[key]
    raise ParseError(
        f"no {what} in {source}\n  tried keys: {keys}\n  keys present: {sorted(data)[:25]}"
    )


# --- pLDDT, the cross-tool path -------------------------------------------


def plddt_from_pdb(path: str | Path, chain_id: str | None = None) -> np.ndarray:
    """Per-residue pLDDT from the B-factor column, normalised to 0-100.

    The most portable pLDDT source: ESMFold, ColabFold and Boltz all write it
    here, whatever their JSON schema does.
    """
    struct = pdb_io.read_pdb(path)
    chain = struct[chain_id] if chain_id else struct.chains[struct.chain_ids[0]]
    values = chain.plddt_array()

    # Two ways the column can be absent. Bio.PDB defaults a missing B-factor to
    # 0.0 rather than None, so a file carrying no pLDDT yields all zeros, not
    # all NaN. Both must raise: a silent 0.0 would be read as a confident
    # prediction that failed the gate, when in fact nothing was predicted --
    # exactly the parser-failure-as-design-failure confusion the funnel exists
    # to separate.
    #
    # All-zero is safe to treat as absent: pLDDT is strictly positive for any
    # real prediction, however poor.
    if values.size == 0 or np.isnan(values).all():
        raise ParseError(f"no B-factor/pLDDT column in {path}")
    if np.nanmax(values) == 0.0:
        raise ParseError(
            f"B-factor column in {path} is entirely zero -- no pLDDT was "
            "written. This is a missing-output problem, not a low-confidence "
            "prediction."
        )
    return metrics.normalise_plddt(values)


# --- records --------------------------------------------------------------


@dataclass
class MonomerRecord:
    """Step 3 output for one sequence."""

    design_id: str
    sequence_id: str
    engine: str
    plddt: float = float("nan")
    scrmsd: float = float("nan")
    ptm: float = float("nan")
    predicted_pdb: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "design_id": self.design_id,
            "sequence_id": self.sequence_id,
            f"{self.engine}_plddt": self.plddt,
            f"{self.engine}_scrmsd": self.scrmsd,
            f"{self.engine}_ptm": self.ptm,
            f"{self.engine}_pdb": self.predicted_pdb,
        }


@dataclass
class ComplexRecord:
    """Step 4 output for one binder-target complex."""

    design_id: str
    sequence_id: str
    engine: str
    iptm: float = float("nan")
    ptm: float = float("nan")
    plddt: float = float("nan")
    pae_interaction: float = float("nan")
    binder_rmsd_to_design: float = float("nan")
    predicted_pdb: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "design_id": self.design_id,
            "sequence_id": self.sequence_id,
            f"{self.engine}_iptm": self.iptm,
            f"{self.engine}_ptm": self.ptm,
            f"{self.engine}_plddt": self.plddt,
            f"{self.engine}_pae_interaction": self.pae_interaction,
            f"{self.engine}_binder_rmsd": self.binder_rmsd_to_design,
            f"{self.engine}_pdb": self.predicted_pdb,
        }


# --- ESMFold (Step 3A) -----------------------------------------------------


def parse_esmfold(
    out_dir: str | Path,
    design_id: str,
    sequence_id: str,
    design_pdb: str | Path | None = None,
    binder_chain: str = "A",
) -> MonomerRecord:
    """ESMFold monomer prediction.

    Run through the esm package, output is a single PDB with pLDDT in the
    B-factor column; a sidecar JSON with ptm is optional, so its absence is a
    note rather than an error.
    """
    out_dir = Path(out_dir)
    record = MonomerRecord(design_id, sequence_id, engine="esmfold")

    pdb = _find(out_dir, [f"*{sequence_id}*.pdb", "*.pdb"], "ESMFold PDB")
    record.predicted_pdb = str(pdb)
    record.plddt = float(np.nanmean(plddt_from_pdb(pdb, binder_chain)))

    for pattern in (f"*{sequence_id}*.json", "*.json"):
        matches = sorted(out_dir.glob(pattern))
        if matches:
            data = _load_json(matches[0])
            ptm = data.get("ptm", data.get("pTM"))
            if ptm is not None:
                record.ptm = float(ptm)
            break
    else:
        record.notes.append("no sidecar JSON; ptm unavailable")

    if design_pdb:
        record.scrmsd = _scrmsd_against_design(pdb, design_pdb, binder_chain, record.notes)
    return record


# --- Boltz-2 ---------------------------------------------------------------


def parse_boltz2_monomer(
    out_dir: str | Path,
    design_id: str,
    sequence_id: str,
    design_pdb: str | Path | None = None,
    binder_chain: str = "A",
) -> MonomerRecord:
    """Boltz-2 monomer prediction (Step 3B)."""
    out_dir = Path(out_dir)
    record = MonomerRecord(design_id, sequence_id, engine="boltz2")

    structure = _find(
        out_dir,
        [
            f"**/*{sequence_id}*model_0.pdb",
            f"**/*{sequence_id}*.pdb",
            "**/*model_0.pdb",
            "**/*.pdb",
            f"**/*{sequence_id}*model_0.cif",
            "**/*.cif",
        ],
        "Boltz-2 structure",
    )
    record.predicted_pdb = str(structure)

    conf_path = _find(
        out_dir,
        [f"**/confidence_*{sequence_id}*.json", "**/confidence_*.json", "**/*confidence*.json"],
        "Boltz-2 confidence JSON",
    )
    conf = _load_json(conf_path)
    record.plddt = float(
        metrics.normalise_plddt(
            [_first_key(conf, ["complex_plddt", "plddt", "confidence_score"], "pLDDT", conf_path)]
        )[0]
    )
    if (ptm := conf.get("ptm")) is not None:
        record.ptm = float(ptm)

    if structure.suffix == ".cif":
        # The CIF reader is not implemented; pLDDT came from JSON above, but
        # scRMSD needs coordinates. Boltz can emit PDB via --output_format pdb.
        record.notes.append(
            "structure is CIF; scRMSD skipped. Run boltz with --output_format pdb."
        )
        return record

    if design_pdb:
        record.scrmsd = _scrmsd_against_design(structure, design_pdb, binder_chain, record.notes)
    return record


def parse_boltz2_complex(
    out_dir: str | Path,
    design_id: str,
    sequence_id: str,
    design_pdb: str | Path | None = None,
    binder_chain: str = "A",
    target_chain: str = "B",
) -> ComplexRecord:
    """Boltz-2 complex prediction (Step 4)."""
    out_dir = Path(out_dir)
    record = ComplexRecord(design_id, sequence_id, engine="boltz2")

    conf_path = _find(
        out_dir,
        [f"**/confidence_*{sequence_id}*.json", "**/confidence_*.json", "**/*confidence*.json"],
        "Boltz-2 confidence JSON",
    )
    conf = _load_json(conf_path)
    record.iptm = float(_first_key(conf, ["iptm", "complex_iptm"], "ipTM", conf_path))
    if (ptm := conf.get("ptm")) is not None:
        record.ptm = float(ptm)
    if (plddt := conf.get("complex_plddt", conf.get("plddt"))) is not None:
        record.plddt = float(metrics.normalise_plddt([plddt])[0])

    structure = _find(
        out_dir,
        [f"**/*{sequence_id}*model_0.pdb", "**/*model_0.pdb", "**/*.pdb"],
        "Boltz-2 complex structure",
    )
    record.predicted_pdb = str(structure)

    try:
        pae = _load_boltz_pae(out_dir, sequence_id)
        record.pae_interaction = _pae_interaction_from_structure(
            structure, pae, binder_chain, target_chain
        )
    except ParseError as exc:
        record.notes.append(f"pae_interaction unavailable: {exc}")

    if design_pdb:
        record.binder_rmsd_to_design = _binder_rmsd(
            structure, design_pdb, binder_chain, target_chain, record.notes
        )
    return record


def _load_boltz_pae(out_dir: Path, sequence_id: str) -> np.ndarray:
    path = _find(
        out_dir,
        [f"**/pae_*{sequence_id}*.npz", "**/pae_*.npz", "**/*pae*.npz"],
        "Boltz-2 PAE npz",
    )
    with np.load(path) as data:
        key = next((k for k in ("pae", "arr_0") if k in data), None)
        if key is None:
            raise ParseError(f"no PAE array in {path}; keys: {list(data.keys())}")
        return np.asarray(data[key], dtype=np.float64)


# --- LocalColabFold --------------------------------------------------------


def parse_colabfold_complex(
    out_dir: str | Path,
    design_id: str,
    sequence_id: str,
    design_pdb: str | Path | None = None,
    binder_chain: str = "A",
    target_chain: str = "B",
) -> ComplexRecord:
    """LocalColabFold AF2-Multimer prediction (Step 4).

    colabfold_batch writes
    `{job}_scores_rank_001_alphafold2_multimer_v3_model_*_seed_*.json` with
    plddt, pae, ptm and iptm, alongside rank-ordered PDBs. Rank 001 is the
    top-scoring model and the one used.
    """
    out_dir = Path(out_dir)
    record = ComplexRecord(design_id, sequence_id, engine="colabfold")

    scores_path = _find(
        out_dir,
        [
            f"*{sequence_id}*scores_rank_001*.json",
            "*scores_rank_001*.json",
            f"*{sequence_id}*scores*.json",
            "*scores*.json",
        ],
        "ColabFold scores JSON",
    )
    scores = _load_json(scores_path)

    record.iptm = float(_first_key(scores, ["iptm", "iPTM"], "ipTM", scores_path))
    if (ptm := scores.get("ptm")) is not None:
        record.ptm = float(ptm)

    structure = _find(
        out_dir,
        [
            f"*{sequence_id}*rank_001*.pdb",
            "*rank_001*.pdb",
            f"*{sequence_id}*unrelaxed*.pdb",
            "*.pdb",
        ],
        "ColabFold PDB",
    )
    record.predicted_pdb = str(structure)

    if "plddt" in scores:
        plddt = metrics.normalise_plddt(scores["plddt"])
        binder_idx, _ = _chain_indices(structure, binder_chain, target_chain)
        if binder_idx.max() < plddt.size:
            record.plddt = float(plddt[binder_idx].mean())
        else:
            record.plddt = float(plddt.mean())
            record.notes.append("pLDDT length != residue count; used whole-complex mean")
    else:
        record.plddt = float(np.nanmean(plddt_from_pdb(structure, binder_chain)))

    if "pae" in scores:
        try:
            record.pae_interaction = _pae_interaction_from_structure(
                structure, np.asarray(scores["pae"], dtype=np.float64), binder_chain, target_chain
            )
        except (ParseError, ValueError) as exc:
            record.notes.append(f"pae_interaction unavailable: {exc}")
    else:
        # Not a flag problem. batch.py writes the pae entry whenever
        # "predicted_aligned_error" is in the model result -- i.e. whenever the
        # model has a PAE head -- independent of --save-all. Its absence means
        # the prediction did not come from a PAE-bearing model, so the usual
        # cause is a --model-type that is not a multimer (a monomer
        # alphafold2_ptm run), or a run that died before writing scores.
        record.notes.append(
            "no PAE in scores JSON; check --model-type is alphafold2_multimer_v3 "
            "(monomer models have no PAE head)"
        )

    if design_pdb:
        record.binder_rmsd_to_design = _binder_rmsd(
            structure, design_pdb, binder_chain, target_chain, record.notes
        )
    return record


# --- shared helpers --------------------------------------------------------


def _chain_indices(
    structure: str | Path, binder_chain: str, target_chain: str
) -> tuple[np.ndarray, np.ndarray]:
    """Row indices into a PAE matrix for each chain, in file order.

    PAE is indexed by concatenated complex position, so the split must follow
    the order the chains appear in the structure, not alphabetical order.
    """
    struct = pdb_io.read_pdb(structure)
    for cid in (binder_chain, target_chain):
        if cid not in struct.chains:
            raise ParseError(
                f"chain {cid!r} not in {structure}; present: {struct.chain_ids}"
            )
    offset = 0
    spans: dict[str, np.ndarray] = {}
    for cid in struct.chains:  # insertion order == file order
        n = len(struct.chains[cid])
        spans[cid] = np.arange(offset, offset + n)
        offset += n
    return spans[binder_chain], spans[target_chain]


def _pae_interaction_from_structure(
    structure: str | Path, pae: np.ndarray, binder_chain: str, target_chain: str
) -> float:
    binder_idx, target_idx = _chain_indices(structure, binder_chain, target_chain)
    total = binder_idx.size + target_idx.size
    if pae.shape[0] != total:
        raise ParseError(
            f"PAE is {pae.shape[0]}x{pae.shape[0]} but complex has {total} residues "
            f"(binder {binder_idx.size} + target {target_idx.size}). "
            "Chain order or residue count mismatch."
        )
    return metrics.pae_interaction(pae, binder_idx, target_idx)


def _scrmsd_against_design(
    predicted: str | Path, design_pdb: str | Path, binder_chain: str, notes: list[str]
) -> float:
    try:
        pred = pdb_io.read_pdb(predicted)
        pred_chain = (
            pred.chains[binder_chain] if binder_chain in pred.chains else pred.only_chain()
        )
        design_binder = pdb_io.read_pdb(design_pdb)[binder_chain]
        return metrics.scrmsd(pred_chain, design_binder)
    except (ParseError, ValueError, KeyError) as exc:
        notes.append(f"scrmsd unavailable: {exc}")
        return float("nan")


def _binder_rmsd(
    predicted: str | Path,
    design_pdb: str | Path,
    binder_chain: str,
    target_chain: str,
    notes: list[str],
) -> float:
    try:
        pred = pdb_io.read_pdb(predicted)
        design = pdb_io.read_pdb(design_pdb)
        return metrics.binder_rmsd_to_design(
            pred[binder_chain], pred[target_chain],
            design[binder_chain], design[target_chain],
        )
    except (ValueError, KeyError) as exc:
        notes.append(f"binder_rmsd_to_design unavailable: {exc}")
        return float("nan")


# --- ProteinMPNN (Step 2) --------------------------------------------------


@dataclass
class MpnnSequence:
    """One sequence from a ProteinMPNN run.

    ProteinMPNN reports its own scores in the FASTA headers, so they are read
    rather than recomputed:

      score         mean negative log-likelihood over DESIGNED positions
      global_score  same over all positions, designed and fixed
      seq_recovery  identity to the input sequence

    Lower score is better -- it is a negative log-likelihood, not a similarity.
    A filter written as `score > x` would select the worst sequences.
    """

    design_id: str
    sample: int
    sequence: str
    chains: list[str]
    score: float = float("nan")
    global_score: float = float("nan")
    seq_recovery: float = float("nan")
    temperature: float = float("nan")
    is_native: bool = False

    @property
    def sequence_id(self) -> str:
        return f"{self.design_id}_seq{self.sample:03d}"

    def chain(self, index: int) -> str:
        """One chain's sequence. ProteinMPNN joins chains with '/'."""
        if index >= len(self.chains):
            raise ParseError(
                f"{self.sequence_id} has {len(self.chains)} chain(s), index {index} requested"
            )
        return self.chains[index]

    def as_row(self) -> dict:
        return {
            "design_id": self.design_id,
            "sequence_id": self.sequence_id,
            "sequence": self.sequence,
            "length": len(self.sequence.replace("/", "")),
            "mpnn_score": self.score,
            "mpnn_global_score": self.global_score,
            "mpnn_seq_recovery": self.seq_recovery,
            "mpnn_temperature": self.temperature,
        }


def _parse_mpnn_header(header: str) -> dict:
    """Pull key=value pairs out of a ProteinMPNN FASTA header.

    Headers look like:
      >design, score=1.1066, global_score=1.1066, fixed_chains=['B'], ...
      >T=0.1, sample=1, score=0.7073, global_score=0.9001, seq_recovery=0.5170

    Bracketed list values are kept as raw strings; only the numeric fields are
    coerced, and a field that will not parse is omitted rather than guessed at.
    """
    fields: dict = {}
    depth = 0
    token = ""
    for ch in header:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            fields.update(_mpnn_token(token))
            token = ""
        else:
            token += ch
    fields.update(_mpnn_token(token))
    return fields


def _mpnn_token(token: str) -> dict:
    token = token.strip()
    if "=" not in token:
        return {"_name": token} if token else {}
    key, _, value = token.partition("=")
    key, value = key.strip(), value.strip()
    try:
        return {key: float(value)}
    except ValueError:
        return {key: value}


def parse_proteinmpnn_fasta(
    path: str | Path, design_id: str, binder_chain_index: int = 0
) -> tuple[MpnnSequence | None, list[MpnnSequence]]:
    """Read a ProteinMPNN output FASTA.

    Returns (native, samples). The first record is the input sequence, echoed
    back with its own score; the rest are the sampled designs. The native is
    returned separately so it cannot be mistaken for a design and carried into
    Step 3.
    """
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as exc:
        raise ParseError(f"cannot read {path}: {exc}") from exc

    records: list[tuple[str, str]] = []
    header: str | None = None
    seq_parts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_parts)))
            header, seq_parts = line[1:], []
        else:
            seq_parts.append(line)
    if header is not None:
        records.append((header, "".join(seq_parts)))

    if not records:
        raise ParseError(f"no FASTA records in {path}")

    native: MpnnSequence | None = None
    samples: list[MpnnSequence] = []
    for position, (head, sequence) in enumerate(records):
        fields = _parse_mpnn_header(head)
        chains = sequence.split("/")
        entry = MpnnSequence(
            design_id=design_id,
            sample=int(fields.get("sample", 0)),
            sequence=sequence,
            chains=chains,
            score=float(fields.get("score", float("nan"))),
            global_score=float(fields.get("global_score", float("nan"))),
            seq_recovery=float(fields.get("seq_recovery", float("nan"))),
            temperature=float(fields.get("T", float("nan"))),
            is_native=position == 0 and "sample" not in fields,
        )
        if entry.is_native:
            native = entry
        else:
            samples.append(entry)

    if not samples:
        raise ParseError(
            f"{path} holds only the native sequence; ProteinMPNN produced no samples"
        )
    return native, samples


def binder_fasta(
    samples: list[MpnnSequence], out_path: str | Path, binder_chain_index: int = 0
) -> Path:
    """Write just the binder chain of each sample, for Step 3 monomer folding.

    Step 3 validates the binder in isolation, so the target chain must be
    dropped here. Passing the full multi-chain string to ESMFold would fold the
    complex as one concatenated sequence and every scRMSD downstream would be
    meaningless.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for sample in samples:
            fh.write(f">{sample.sequence_id}\n{sample.chain(binder_chain_index)}\n")
    return out_path
