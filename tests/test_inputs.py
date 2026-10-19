"""Step 4 input construction.

The failure these guard against is silent: ColabFold and Boltz-2 take different
complex formats, and feeding one the other's file yields a confident prediction
of the wrong molecule rather than an error.
"""

from pathlib import Path

import pytest

from pipeline import inputs, parsers, pdb_io

PDB_DIR = Path(__file__).resolve().parents[1] / "examples/input_pdbs"
INSULIN = PDB_DIR / "insulin_target.pdb"

FASTA = """\
>design_0, score=1.1, global_score=1.1, fixed_chains=['B'], designed_chains=['A']
MKTAYIAKQRQ/TARGETSEQ
>T=0.1, sample=1, score=0.7, global_score=0.9, seq_recovery=0.5
MKTWYIAKQRQ/TARGETSEQ
"""


@pytest.fixture
def samples(tmp_path):
    path = tmp_path / "seqs.fa"
    path.write_text(FASTA)
    _, samples = parsers.parse_proteinmpnn_fasta(path, "design_0")
    return samples


class TestSequenceExtraction:
    def test_reads_target_sequence_from_structure(self):
        seq = inputs.target_sequence(INSULIN, "A")
        assert len(seq) == 150
        assert seq.startswith("EVCPGMDIRNN")

    def test_three_to_one_covers_standard_residues(self):
        assert len(set(pdb_io.THREE_TO_ONE.values())) == 20

    def test_mse_maps_to_methionine(self):
        # Selenomethionine is a methionine analogue; predictors expect M.
        assert pdb_io.THREE_TO_ONE["MSE"] == "M"

    def test_unknown_residue_becomes_x_not_dropped(self):
        # A silently shortened sequence would misalign every per-residue metric.
        chain = pdb_io.Chain("A", [
            pdb_io.Residue(seq_id=1, name="ALA"),
            pdb_io.Residue(seq_id=2, name="UNK"),
            pdb_io.Residue(seq_id=3, name="GLY"),
        ])
        assert chain.sequence() == "AXG"


class TestColabfoldFormat:
    def test_chains_joined_with_colon(self, samples, tmp_path):
        out = inputs.write_colabfold_complex_fasta(samples, "TTTTT", tmp_path / "c.fa")
        text = out.read_text()
        assert "MKTWYIAKQRQ:TTTTT" in text

    def test_one_record_per_candidate(self, samples, tmp_path):
        out = inputs.write_colabfold_complex_fasta(samples, "TTTTT", tmp_path / "c.fa")
        assert out.read_text().count(">") == len(samples)

    def test_binder_comes_before_target(self, samples, tmp_path):
        # Determines which chain is A in ColabFold's output; Step 4 parsing
        # depends on knowing this.
        out = inputs.write_colabfold_complex_fasta(samples, "TTTTT", tmp_path / "c.fa")
        line = [l for l in out.read_text().splitlines() if ":" in l][0]
        binder, target = line.split(":")
        assert binder == "MKTWYIAKQRQ" and target == "TTTTT"

    def test_mpnn_target_chain_is_not_reused(self, samples, tmp_path):
        # The target sequence comes from the structure, not from ProteinMPNN's
        # echoed target chain, which may have been redesigned.
        out = inputs.write_colabfold_complex_fasta(samples, "REALTARGET", tmp_path / "c.fa")
        assert "TARGETSEQ" not in out.read_text()


class TestBoltzFormat:
    """Boltz-2 takes YAML, one document per prediction. Upstream
    docs/prediction.md: "YAML is preferred; FASTA is deprecated"."""

    def _doc(self, written):
        import yaml
        return yaml.safe_load(written.read_text())

    def test_one_file_per_candidate(self, samples, tmp_path):
        # Every entry under `sequences:` becomes a CHAIN of one target, so a
        # single multi-candidate file would be one N-chain complex rather than
        # N complexes -- a wrong answer with no error. Observed defect.
        written = inputs.write_boltz_complex_yamls(samples, "TTTTT", tmp_path / "b")
        assert len(written) == len(samples)
        assert all(p.exists() for p in written)

    def test_writes_yaml_not_fasta(self, samples, tmp_path):
        written = inputs.write_boltz_complex_yamls(samples, "TTTTT", tmp_path / "b")
        assert all(p.suffix == ".yaml" for p in written)
        assert ">" not in written[0].read_text(), "FASTA headers are deprecated here"

    def test_schema_version_is_declared(self, samples, tmp_path):
        # parse_boltz_schema requires it.
        written = inputs.write_boltz_complex_yamls(samples, "TTTTT", tmp_path / "b")
        assert self._doc(written[0])["version"] == 1

    def test_two_protein_chains_binder_first(self, samples, tmp_path):
        written = inputs.write_boltz_complex_yamls(samples, "TTTTT", tmp_path / "b")
        seqs = self._doc(written[0])["sequences"]
        assert len(seqs) == 2
        assert [list(s)[0] for s in seqs] == ["protein", "protein"]
        assert [s["protein"]["id"] for s in seqs] == ["A", "B"]
        # Binder first, matching what ColabFold receives -- NOT the design
        # PDB's layout, which is why parsers take chain ids explicitly.
        assert seqs[1]["protein"]["sequence"] == "TTTTT"

    def test_msa_is_the_literal_empty_by_default(self, samples, tmp_path):
        # Omitting the key means msa=0 == auto-generate, and boltz then raises
        # "Missing MSA's in input and --use_msa_server flag not set."
        written = inputs.write_boltz_complex_yamls(samples, "TTTTT", tmp_path / "b")
        for entry in self._doc(written[0])["sequences"]:
            assert entry["protein"]["msa"] == "empty"

    def test_msa_none_omits_the_key_for_server_mode(self, samples, tmp_path):
        written = inputs.write_boltz_complex_yamls(
            samples, "TTTTT", tmp_path / "b", msa=None
        )
        for entry in self._doc(written[0])["sequences"]:
            assert "msa" not in entry["protein"]

    def test_msa_path_passes_through(self, samples, tmp_path):
        written = inputs.write_boltz_complex_yamls(
            samples, "TTTTT", tmp_path / "b", msa="/cache/msas/x.a3m"
        )
        assert self._doc(written[0])["sequences"][0]["protein"]["msa"] == "/cache/msas/x.a3m"

    def test_does_not_use_colon_separator(self, samples, tmp_path):
        # A ColabFold-style ':' joined record would misparse as one chain.
        written = inputs.write_boltz_complex_yamls(samples, "TTTTT", tmp_path / "b")
        for entry in self._doc(written[0])["sequences"]:
            assert ":" not in entry["protein"]["sequence"]

    def test_filenames_match_sequence_ids(self, samples, tmp_path):
        # boltz names the job after the file stem and writes to
        # <out>/predictions/<stem>/, which is how parsers locate a candidate.
        written = inputs.write_boltz_complex_yamls(samples, "TTTTT", tmp_path / "b")
        assert {p.stem for p in written} == {s.sequence_id for s in samples}

    def test_no_pocket_conditioning(self, samples, tmp_path):
        # YAML is the only format that offers it, and our hotspots are known --
        # but conditioning the Step 4 validator on the designed epitope would
        # restate the design intent instead of testing it, and ColabFold has no
        # equivalent so the engine comparison would break.
        written = inputs.write_boltz_complex_yamls(samples, "TTTTT", tmp_path / "b")
        assert "constraints" not in self._doc(written[0])

    def test_stale_inputs_are_cleared(self, samples, tmp_path):
        # Boltz globs the whole directory. A leftover X.fasta beside X.yaml is
        # the same job name twice, so one is skipped as "existing" and which
        # one wins is undefined.
        out = tmp_path / "b"
        out.mkdir()
        (out / f"{samples[0].sequence_id}.fasta").write_text(">A|protein|empty\nOLD\n")
        (out / "orphan_from_a_bigger_run.fasta").write_text(">A|protein|empty\nX\n")
        (out / "notes.txt").write_text("keep")

        inputs.write_boltz_complex_yamls(samples, "TTTTT", out)
        names = {p.name for p in out.iterdir()}
        assert f"{samples[0].sequence_id}.fasta" not in names
        assert "orphan_from_a_bigger_run.fasta" not in names
        assert "notes.txt" in names, "only input-shaped files may be removed"


class TestBoltzMonomerYamls:
    """Step 3B folds the binder alone; including the target would make it a
    complex prediction and every scRMSD meaningless."""

    def _doc(self, path):
        import yaml
        return yaml.safe_load(path.read_text())

    def _written(self, tmp_path, n=3):
        class S:
            def __init__(self, sid, seq):
                self.sequence_id, self.design_id, self._seq = sid, "design_0", seq
            def chain(self, i):
                return self._seq
        samples = [S(f"design_0_seq{i:03d}", "MTSKELL") for i in range(1, n + 1)]
        return inputs.write_boltz_monomer_yamls(samples, tmp_path / "mono")

    def test_one_file_per_candidate(self, tmp_path):
        written = self._written(tmp_path, n=3)
        assert len(written) == 3
        for path in written:
            assert len(self._doc(path)["sequences"]) == 1

    def test_single_chain_named_a(self, tmp_path):
        entry = self._doc(self._written(tmp_path)[0])["sequences"][0]["protein"]
        assert entry["id"] == "A"
        assert entry["msa"] == "empty"

    def test_filename_carries_the_sequence_id(self, tmp_path):
        written = self._written(tmp_path, n=2)
        assert {p.stem for p in written} == {"design_0_seq001", "design_0_seq002"}
        assert all(p.suffix == ".yaml" for p in written)


class TestControls:
    """Controls go through the same Boltz YAML writer as real candidates, so
    they exercise the identical path. These read the binder out of the parsed
    document rather than by line index -- an earlier version took
    `splitlines()[1]`, which silently became the literal "sequences:" line
    after the YAML migration and made one assertion vacuously true."""

    @staticmethod
    def _binder(path):
        import yaml
        doc = yaml.safe_load(path.read_text())
        return doc["sequences"][0]["protein"]["sequence"]

    def test_negative_preserves_composition(self, tmp_path):
        # A shuffle isolates structure from composition: same residues, no fold.
        lead = "MKTWYIAKQRQISFVK"
        paths = inputs.write_controls(lead, "TTTTT", tmp_path)
        scrambled = self._binder(paths["negative"])
        assert sorted(scrambled) == sorted(lead)
        assert len(scrambled) == len(lead)

    def test_negative_is_actually_shuffled(self, tmp_path):
        lead = "MKTWYIAKQRQISFVKAAAGGGLLL"
        paths = inputs.write_controls(lead, "T", tmp_path)
        assert self._binder(paths["negative"]) != lead

    def test_controls_carry_the_target_as_chain_b(self, tmp_path):
        import yaml
        paths = inputs.write_controls("MKT", "REALTARGET", tmp_path)
        doc = yaml.safe_load(paths["negative"].read_text())
        assert doc["sequences"][1]["protein"]["id"] == "B"
        assert doc["sequences"][1]["protein"]["sequence"] == "REALTARGET"

    def test_shuffle_is_deterministic(self, tmp_path):
        lead = "MKTWYIAKQRQISFVK"
        a = inputs.write_controls(lead, "T", tmp_path / "a", seed=7)["negative"].read_text()
        b = inputs.write_controls(lead, "T", tmp_path / "b", seed=7)["negative"].read_text()
        assert a == b

    def test_positive_written_only_when_supplied(self, tmp_path):
        assert "positive" not in inputs.write_controls("MKT", "T", tmp_path / "a")
        assert "positive" in inputs.write_controls(
            "MKT", "T", tmp_path / "b", positive_sequence="WWW"
        )


class TestMsaMode:
    """`boltz2.msa_mode` -> (yaml `msa` value, --use_msa_server).

    Both are derived from the one mode string because they must agree and a
    mismatch does not error: `msa: empty` with the flag runs single-sequence
    anyway, discarding the MSA the user asked for.
    """

    def test_single_sequence_writes_empty_and_no_flag(self):
        assert inputs.msa_spec("single_sequence") == inputs.MSA_SINGLE_SEQUENCE
        assert inputs.use_msa_server("single_sequence") is False

    def test_msa_server_omits_the_key_and_takes_the_flag(self):
        assert inputs.msa_spec("msa_server") is None
        assert inputs.use_msa_server("msa_server") is True

    def test_precomputed_alignment_passes_through_without_the_flag(self):
        # Read from disk, so a public API lookup would be pure waste.
        assert inputs.msa_spec("/cache/msas/x.a3m") == "/cache/msas/x.a3m"
        assert inputs.use_msa_server("/cache/msas/x.a3m") is False

    @pytest.mark.parametrize("mode", ["", "none", "None", "null", "empty"])
    def test_ambiguous_modes_are_refused(self, mode):
        # "empty" as a MODE would silently read as single-sequence while the
        # author may have meant "no mode set"; guessing picks a regime.
        with pytest.raises(ValueError):
            inputs.msa_spec(mode)

    def test_config_declares_a_valid_mode(self):
        import yaml

        root = Path(__file__).resolve().parents[1]
        cfg = yaml.safe_load((root / "config/campaign.yaml").read_text())
        mode = cfg["boltz2"]["msa_mode"]
        # Must not raise: the notebook resolves this at import time.
        inputs.msa_spec(mode)

    def test_both_boltz2_stages_gate_the_flag_on_the_same_knob(self):
        # The 3B monomer gate and the Step 4 complex must run in ONE MSA
        # regime, or scRMSD and pae_interaction describe different models.
        import json

        root = Path(__file__).resolve().parents[1]
        nb = json.loads(
            (root / "notebooks/01_binder_design_prototype.ipynb").read_text()
        )
        cells = [
            "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
        ]
        # The append call, not a bare mention: the flag is also named in the
        # comments that explain it.
        flagged = [c for c in cells if 'command.append("--use_msa_server")' in c]
        assert len(flagged) == 2, "expected the monomer and complex runs"
        for cell in flagged:
            assert "BOLTZ2_USE_MSA_SERVER" in cell
