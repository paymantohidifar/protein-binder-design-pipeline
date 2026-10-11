"""Build the input files each tool expects (Step 4 complex prediction).

Separate from parsers.py: that reads tool output, this writes tool input.

Boltz-2 and ColabFold take DIFFERENT complex formats, so one shared FASTA
cannot serve both -- ColabFold joins chains with ':' inside a single record,
while Boltz-2 uses one header per chain with a `>CHAIN|protein` convention.
Writing a single `complexes.fasta` for both would fail for at least one engine,
and probably silently: ColabFold reading a Boltz file sees one long
concatenated monomer, which would yield a confident-looking prediction of the
wrong thing.

FORMAT CAVEAT: the Boltz-2 header convention here is written from its
documented format and has NOT been verified against a real run. Confirm during
the Phase A smoke test.
"""

from __future__ import annotations

from pathlib import Path

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


def write_boltz_complex_fastas(
    samples: list,
    target_seq: str,
    out_dir: str | Path,
    binder_chain_index: int = 0,
) -> list[Path]:
    """One file per candidate, one header per chain.

    Boltz-2 predicts a single complex per input file, so a batch is a directory
    of files rather than one multi-record FASTA.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for sample in samples:
        binder = sample.chain(binder_chain_index)
        path = out_dir / f"{sample.sequence_id}.fasta"
        path.write_text(
            f">A|protein|empty\n{binder}\n"
            f">B|protein|empty\n{target_seq}\n"
        )
        written.append(path)
    return written


def write_controls(
    lead_sequence: str,
    target_seq: str,
    out_dir: str | Path,
    positive_sequence: str | None = None,
    seed: int = 0,
) -> dict[str, Path]:
    """Positive and negative controls for Step 4.

    Without these a zero-lead campaign cannot be interpreted: it looks identical
    whether the gates are working, or `pae_interaction` has its sign or its
    index blocks wrong.

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

    negative = out_dir / "control_negative_scramble.fasta"
    negative.write_text(
        f">A|protein|empty\n{''.join(scrambled)}\n>B|protein|empty\n{target_seq}\n"
    )
    paths["negative"] = negative

    if positive_sequence:
        positive = out_dir / "control_positive.fasta"
        positive.write_text(
            f">A|protein|empty\n{positive_sequence}\n>B|protein|empty\n{target_seq}\n"
        )
        paths["positive"] = positive
    return paths
