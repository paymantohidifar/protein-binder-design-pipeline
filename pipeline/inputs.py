"""Build the input files each tool expects (Step 4 complex prediction).

Separate from parsers.py: that reads tool output, this writes tool input.

BOLTZ-2 USES YAML, NOT FASTA. Upstream `docs/prediction.md` states: *"YAML is
preferred; FASTA is deprecated"* and *"FASTA format is still supported but is
deprecated and only supports a limited subset of features compared to YAML."*
FASTA cannot express modified residues, covalent bonds, pocket conditioning, or
affinity. None of those are used today, but the campaign should not sit on a
deprecated path, and pocket conditioning is a decision we may want later (see
`write_boltz_complex_yamls`).

The schema, one file per prediction:

    version: 1
    sequences:
      - protein:
          id: A
          sequence: <binder>
          msa: empty
      - protein:
          id: B
          sequence: <target>
          msa: empty

Three rules survive the format change, each of which fails differently when
broken:

1. **`msa` must be the literal `empty` for single-sequence mode.** Upstream:
   *"To force single-sequence mode (not recommended, as it reduces accuracy),
   set `msa: empty`."* Omitting the key means `msa=0`, i.e. *auto-generate*,
   and boltz then raises `Missing MSA's in input and --use_msa_server flag not
   set.` (main.py:582). `empty` maps to -1 (schema.py:1127) -- the
   database-free path this campaign requires by default.
2. **One file per prediction.** Every entry under `sequences:` becomes a CHAIN
   of ONE target, and the job name is the file stem. A file holding ten
   candidates is a ten-chain complex, not ten monomers -- a wrong answer with
   no error. This was an observed defect, originally via FASTA.
3. **`version: 1` is required** by `parse_boltz_schema`.

ColabFold is unaffected and still takes a single FASTA record with chains
joined by ':'. The two engines have never shared an input file and must not:
ColabFold reading a Boltz input would see one concatenated monomer and predict
the wrong thing confidently.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from . import pdb_io


def target_sequence(target_pdb: str | Path, chain_id: str) -> str:
    """One-letter sequence of the target chain, read from its structure."""
    chain = pdb_io.read_pdb(target_pdb)[chain_id]
    sequence = chain.sequence()
    if "X" in sequence:
        n = sequence.count("X")
        print(f"  note: {n} non-standard residue(s) in chain {chain_id} written as X")
    return sequence


def write_colabfold_complex_fasta(
    samples: list,
    target_seq: str,
    out_path: str | Path,
    binder_chain_index: int = 0,
) -> Path:
    """One record per candidate, chains joined by ':' (AF2-multimer convention).

    Binder first, then target, so chain A is the binder and chain B the target
    in ColabFold's output -- which is the OPPOSITE of RFdiffusion's ordering.
    Step 4 parsing must therefore be told which is which explicitly; it is not
    the same as the design PDB's layout.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for sample in samples:
            binder = sample.chain(binder_chain_index)
            fh.write(f">{sample.sequence_id}\n{binder}:{target_seq}\n")
    return out_path


BOLTZ_SCHEMA_VERSION = 1
BOLTZ_INPUT_SUFFIXES = (".yaml", ".yml", ".fasta", ".fa", ".fas")

# `msa: empty` forces single-sequence mode. Omitting the key entirely means
# "auto-generate", which requires --use_msa_server; anything else is read as a
# path to a custom .a3m/.csv alignment.
MSA_SINGLE_SEQUENCE = "empty"


def _boltz_yaml(chains: list[tuple[str, str]], msa: str | None) -> str:
    """Render one Boltz-2 input document.

    `chains` is [(chain_id, sequence), ...] in the order they should appear.
    `msa` is `"empty"` for single-sequence mode, a path for a custom
    alignment, or None to omit the key so boltz generates one (which then
    requires `--use_msa_server`).
    """
    sequences = []
    for chain_id, sequence in chains:
        protein: dict[str, object] = {"id": chain_id, "sequence": sequence}
        if msa is not None:
            protein["msa"] = msa
        sequences.append({"protein": protein})
    # sort_keys=False keeps id/sequence/msa in reading order; default_flow_style
    # False keeps it block-style, which is what the documented schema shows.
    return yaml.safe_dump(
        {"version": BOLTZ_SCHEMA_VERSION, "sequences": sequences},
        sort_keys=False,
        default_flow_style=False,
    )


def _clear_boltz_inputs(out_dir: Path) -> int:
    """Remove stale input files, returning how many went.

    Boltz is pointed at a DIRECTORY and globs every `*` in it, so leftovers
    from an earlier run are silently predicted again. Worse, migrating from
    FASTA to YAML would leave `X.fasta` beside `X.yaml` -- two files with the
    same stem, hence the same job name, so one prediction would be skipped as
    "existing" and which one won is undefined. Only input-shaped files are
    touched, and only in a directory these writers own.
    """
    removed = 0
    for path in sorted(out_dir.glob("*")):
        if path.is_file() and path.suffix.lower() in BOLTZ_INPUT_SUFFIXES:
            path.unlink()
            removed += 1
    return removed


def write_boltz_monomer_yamls(
    samples: list,
    out_dir: str | Path,
    binder_chain_index: int = 0,
    msa: str | None = MSA_SINGLE_SEQUENCE,
) -> list[Path]:
    """One single-chain YAML per candidate, for the Step 3B monomer gate.

    The Boltz-2 counterpart of `parsers.binder_fasta`, which writes one
    multi-record FASTA for ESMFold. That file cannot be reused here: ESMFold's
    `>design_0_seq001` headers are not a Boltz input format at all, and even
    in Boltz's own FASTA dialect all records in one file would be parsed as a
    single multi-chain complex rather than separate monomers.

    Files are named `<sequence_id>.yaml` because boltz derives the job name
    from the file stem and writes to `<out_dir>/predictions/<stem>/`, which is
    what lets `parsers.parse_boltz2_monomer` locate a candidate's structure by
    globbing on its sequence_id.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_boltz_inputs(out_dir)
    written = []
    for sample in samples:
        path = out_dir / f"{sample.sequence_id}.yaml"
        path.write_text(_boltz_yaml([("A", sample.chain(binder_chain_index))], msa))
        written.append(path)
    return written


def write_boltz_complex_yamls(
    samples: list,
    target_seq: str,
    out_dir: str | Path,
    binder_chain_index: int = 0,
    msa: str | None = MSA_SINGLE_SEQUENCE,
) -> list[Path]:
    """One two-chain YAML per candidate: binder as A, target as B.

    Chain order is binder-first to match what ColabFold receives, so both Step
    4 engines share one chain convention -- which is NOT the design PDB's
    layout, and is why `parsers` is passed chain ids explicitly.

    POCKET CONDITIONING IS DELIBERATELY NOT USED, although YAML is the only
    format that offers it:

        constraints:
          - pocket:
              binder: A
              contacts: [[B, 146], [B, 170], [B, 178]]
              max_distance: 6

    Our hotspots are known, so this is tempting. It would be a mistake here.
    Step 4 exists to ask *independently* whether the binder docks at the
    epitope it was designed against; conditioning the validator on that answer
    turns the check into a restatement of the design intent and would inflate
    pass rates. It would also make the two Step 4 engines non-comparable, since
    ColabFold has no equivalent -- and the engine comparison is the whole
    reason both run. Worth revisiting only as a separate, labelled experiment.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_boltz_inputs(out_dir)
    written = []
    for sample in samples:
        path = out_dir / f"{sample.sequence_id}.yaml"
        path.write_text(_boltz_yaml(
            [("A", sample.chain(binder_chain_index)), ("B", target_seq)], msa
        ))
        written.append(path)
    return written


def write_controls(
    lead_sequence: str,
    target_seq: str,
    out_dir: str | Path,
    positive_sequence: str | None = None,
    seed: int = 0,
    msa: str | None = MSA_SINGLE_SEQUENCE,
) -> dict[str, Path]:
    """Positive and negative controls for Step 4.

    Without these a zero-lead campaign cannot be interpreted: it looks identical
    whether the gates are working, or `pae_interaction` has its sign or its
    index blocks wrong.

    `msa` must match what the real candidates get: a control predicted in a
    different MSA regime is not a control, since MSA depth shifts the
    `pae_interaction` distribution it is meant to calibrate.

    The negative is a length-matched shuffle of a real lead, which preserves
    amino-acid composition while destroying the fold -- so it isolates structure
    from composition. It must score clearly WORSE than real designs; if it does
    not, the metric is broken, not the designs.
    """
    import random

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scrambled = list(lead_sequence)
    random.Random(seed).shuffle(scrambled)
    paths = {}

    _clear_boltz_inputs(out_dir)

    negative = out_dir / "control_negative_scramble.yaml"
    negative.write_text(_boltz_yaml(
        [("A", "".join(scrambled)), ("B", target_seq)], msa
    ))
    paths["negative"] = negative

    if positive_sequence:
        positive = out_dir / "control_positive.yaml"
        positive.write_text(_boltz_yaml(
            [("A", positive_sequence), ("B", target_seq)], msa
        ))
        paths["positive"] = positive
    return paths
