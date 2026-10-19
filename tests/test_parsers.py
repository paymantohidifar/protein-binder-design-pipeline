"""Parser tests against synthetic fixtures.

None of the predictors are installed here, so these do not prove the real
schemas are right -- Phase A does that. What they DO prove is the logic that
sits between the file and the metric: chain-to-PAE index mapping, scale
normalisation, and that a schema mismatch produces a legible error instead of a
silent NaN.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline import parsers


def write_pdb(path, chains, plddt=None):
    """Minimal multi-chain PDB with pLDDT in the B-factor column.

    `chains` is [(chain_id, n_residues), ...] and is written in the given order,
    which is what the PAE index mapping must follow.
    """
    lines = []
    serial = 1
    for chain_id, n in chains:
        for i in range(1, n + 1):
            b = 50.0 if plddt is None else plddt[chain_id][i - 1]
            for atom, dx in (("N", 0.0), ("CA", 1.0), ("C", 2.0), ("O", 3.0)):
                lines.append(
                    f"ATOM  {serial:5d} {atom:<4s} ALA {chain_id}{i:4d}    "
                    f"{i * 3.0 + dx:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{b:6.2f}"
                    f"          {atom[0]:>2s}"
                )
                serial += 1
        lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


class TestPlddtFromPdb:
    def test_reads_bfactor_column(self, tmp_path):
        pdb = write_pdb(tmp_path / "m.pdb", [("A", 3)], plddt={"A": [90.0, 80.0, 70.0]})
        assert parsers.plddt_from_pdb(pdb, "A") == pytest.approx([90.0, 80.0, 70.0])

    def test_normalises_fractional_scale(self, tmp_path):
        # ESMFold writes 0-1 in some configurations; a 0.9 pLDDT must not fail a
        # ">= 80" gate because of units.
        pdb = write_pdb(tmp_path / "m.pdb", [("A", 2)], plddt={"A": [0.9, 0.8]})
        assert parsers.plddt_from_pdb(pdb, "A") == pytest.approx([90.0, 80.0])

    def test_selects_requested_chain(self, tmp_path):
        pdb = write_pdb(
            tmp_path / "c.pdb", [("A", 2), ("B", 2)], plddt={"A": [90.0, 90.0], "B": [10.0, 10.0]}
        )
        assert parsers.plddt_from_pdb(pdb, "A").mean() == pytest.approx(90.0)
        assert parsers.plddt_from_pdb(pdb, "B").mean() == pytest.approx(10.0)

    def test_missing_column_raises(self, tmp_path):
        # Bio.PDB defaults an absent B-factor to 0.0, not None, so this file
        # parses to pLDDT 0.0 everywhere. Without the all-zero check it would
        # look like a confident prediction that failed the gate, rather than
        # output that was never written.
        pdb = tmp_path / "nob.pdb"
        pdb.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n" "END\n"
        )
        with pytest.raises(parsers.ParseError, match="entirely zero"):
            parsers.plddt_from_pdb(pdb)

    def test_all_zero_bfactor_is_rejected_not_reported_as_zero_confidence(self, tmp_path):
        # Regression guard for the swap to Bio.PDB. pLDDT is strictly positive
        # for any real prediction, so an all-zero column means missing output.
        pdb = write_pdb(tmp_path / "z.pdb", [("A", 4)], plddt={"A": [0.0] * 4})
        with pytest.raises(parsers.ParseError, match="missing-output problem"):
            parsers.plddt_from_pdb(pdb, "A")

    def test_genuinely_low_plddt_is_still_returned(self, tmp_path):
        # The complement: low-but-nonzero confidence is a real measurement and
        # must reach the gate as a threshold failure, not a parse error.
        pdb = write_pdb(tmp_path / "low.pdb", [("A", 3)], plddt={"A": [12.0, 8.0, 15.0]})
        assert parsers.plddt_from_pdb(pdb, "A").mean() < 20.0


class TestChainIndices:
    def test_follows_file_order_not_alphabetical(self, tmp_path):
        # The bug this guards: PAE is indexed by concatenated complex position.
        # If the target is written FIRST, the binder's rows start at the target's
        # length -- mapping alphabetically would silently read the wrong block
        # and produce a plausible but meaningless pae_interaction.
        pdb = write_pdb(tmp_path / "tb.pdb", [("B", 5), ("A", 3)])
        binder_idx, target_idx = parsers._chain_indices(pdb, "A", "B")
        assert target_idx.tolist() == [0, 1, 2, 3, 4]
        assert binder_idx.tolist() == [5, 6, 7]

    def test_binder_first_layout(self, tmp_path):
        pdb = write_pdb(tmp_path / "bt.pdb", [("A", 3), ("B", 5)])
        binder_idx, target_idx = parsers._chain_indices(pdb, "A", "B")
        assert binder_idx.tolist() == [0, 1, 2]
        assert target_idx.tolist() == [3, 4, 5, 6, 7]

    def test_missing_chain_lists_what_is_present(self, tmp_path):
        pdb = write_pdb(tmp_path / "a.pdb", [("A", 2)])
        with pytest.raises(parsers.ParseError, match=r"present: \['A'\]"):
            parsers._chain_indices(pdb, "A", "B")


class TestPaeInteractionFromStructure:
    def test_uses_correct_block_for_target_first_layout(self, tmp_path):
        pdb = write_pdb(tmp_path / "tb.pdb", [("B", 4), ("A", 2)])
        # Intra-chain blocks confident, inter-chain blocks bad. Correct answer
        # is 20.0; reading the wrong block would give ~1.0.
        pae = np.ones((6, 6))
        pae[np.ix_([4, 5], [0, 1, 2, 3])] = 20.0
        pae[np.ix_([0, 1, 2, 3], [4, 5])] = 20.0
        assert parsers._pae_interaction_from_structure(pdb, pae, "A", "B") == pytest.approx(20.0)

    def test_size_mismatch_is_reported_clearly(self, tmp_path):
        # A truncated or padded PAE must not be silently indexed.
        pdb = write_pdb(tmp_path / "c.pdb", [("A", 3), ("B", 4)])
        with pytest.raises(parsers.ParseError, match="PAE is 5x5 but complex has 7"):
            parsers._pae_interaction_from_structure(pdb, np.ones((5, 5)), "A", "B")


class TestFindDiagnostics:
    def test_failure_lists_patterns_and_files(self, tmp_path):
        (tmp_path / "something_else.txt").write_text("x")
        with pytest.raises(parsers.ParseError) as exc:
            parsers._find(tmp_path, ["*.pdb", "*.cif"], "structure")
        message = str(exc.value)
        assert "tried patterns" in message
        assert "something_else.txt" in message

    def test_prefers_earlier_pattern(self, tmp_path):
        (tmp_path / "b.cif").write_text("x")
        (tmp_path / "a.pdb").write_text("x")
        assert parsers._find(tmp_path, ["*.pdb", "*.cif"], "s").name == "a.pdb"


class TestFirstKeyDiagnostics:
    def test_missing_key_lists_present_keys(self, tmp_path):
        path = tmp_path / "c.json"
        with pytest.raises(parsers.ParseError, match="keys present"):
            parsers._first_key({"confidence": 1}, ["iptm"], "ipTM", path)

    def test_returns_first_available_alias(self, tmp_path):
        assert parsers._first_key({"complex_iptm": 0.9}, ["iptm", "complex_iptm"], "x",
                                  tmp_path / "c.json") == 0.9


class TestEsmfold:
    def test_parses_plddt_and_records_missing_ptm(self, tmp_path):
        write_pdb(tmp_path / "seq_0001.pdb", [("A", 3)], plddt={"A": [88.0, 90.0, 92.0]})
        rec = parsers.parse_esmfold(tmp_path, "design_0", "seq_0001")
        assert rec.plddt == pytest.approx(90.0)
        assert np.isnan(rec.ptm)
        assert any("ptm unavailable" in n for n in rec.notes)

    def test_reads_ptm_from_sidecar(self, tmp_path):
        write_pdb(tmp_path / "seq_0001.pdb", [("A", 2)], plddt={"A": [80.0, 80.0]})
        (tmp_path / "seq_0001.json").write_text(json.dumps({"ptm": 0.77}))
        assert parsers.parse_esmfold(tmp_path, "d", "seq_0001").ptm == pytest.approx(0.77)

    def test_scrmsd_zero_against_itself(self, tmp_path):
        pdb = write_pdb(tmp_path / "seq_0001.pdb", [("A", 6)], plddt={"A": [90.0] * 6})
        rec = parsers.parse_esmfold(tmp_path, "d", "seq_0001", design_pdb=pdb)
        assert rec.scrmsd == pytest.approx(0.0, abs=1e-9)

    def test_missing_output_raises_with_listing(self, tmp_path):
        with pytest.raises(parsers.ParseError, match="ESMFold PDB"):
            parsers.parse_esmfold(tmp_path, "d", "seq_0001")


class TestBoltz2Complex:
    def _setup(self, tmp_path, iptm=0.85):
        pred = tmp_path / "predictions" / "seq_0001"
        pred.mkdir(parents=True)
        write_pdb(pred / "seq_0001_model_0.pdb", [("A", 3), ("B", 4)])
        (pred / "confidence_seq_0001_model_0.json").write_text(
            json.dumps({"iptm": iptm, "ptm": 0.8, "complex_plddt": 0.91})
        )
        pae = np.ones((7, 7))
        pae[np.ix_([0, 1, 2], [3, 4, 5, 6])] = 7.0
        pae[np.ix_([3, 4, 5, 6], [0, 1, 2])] = 9.0
        np.savez(pred / "pae_seq_0001_model_0.npz", pae=pae)
        return tmp_path

    def test_parses_all_metrics(self, tmp_path):
        root = self._setup(tmp_path)
        rec = parsers.parse_boltz2_complex(root, "d", "seq_0001")
        assert rec.iptm == pytest.approx(0.85)
        assert rec.plddt == pytest.approx(91.0)  # 0.91 normalised
        assert rec.pae_interaction == pytest.approx(8.0)  # (7+9)/2
        assert rec.notes == []

    def test_missing_pae_is_a_note_not_a_crash(self, tmp_path):
        root = self._setup(tmp_path)
        next(root.glob("**/pae_*.npz")).unlink()
        rec = parsers.parse_boltz2_complex(root, "d", "seq_0001")
        assert rec.iptm == pytest.approx(0.85)
        assert np.isnan(rec.pae_interaction)
        assert any("pae_interaction unavailable" in n for n in rec.notes)

    def test_cif_only_output_notes_the_flag_to_use(self, tmp_path):
        pred = tmp_path / "predictions" / "seq_0001"
        pred.mkdir(parents=True)
        (pred / "seq_0001_model_0.cif").write_text("data_x\n")
        (pred / "confidence_seq_0001_model_0.json").write_text(json.dumps({"complex_plddt": 0.9}))
        rec = parsers.parse_boltz2_monomer(tmp_path, "d", "seq_0001", design_pdb="unused.pdb")
        assert any("--output_format pdb" in n for n in rec.notes)
        assert np.isnan(rec.scrmsd)


class TestColabfoldComplex:
    def _setup(self, tmp_path, with_pae=True):
        write_pdb(tmp_path / "seq_0001_rank_001_model_3.pdb", [("A", 3), ("B", 4)])
        scores = {"iptm": 0.82, "ptm": 0.75, "plddt": [90.0] * 3 + [40.0] * 4}
        if with_pae:
            pae = np.ones((7, 7))
            pae[np.ix_([0, 1, 2], [3, 4, 5, 6])] = 5.0
            pae[np.ix_([3, 4, 5, 6], [0, 1, 2])] = 5.0
            scores["pae"] = pae.tolist()
        (tmp_path / "seq_0001_scores_rank_001_model_3.json").write_text(json.dumps(scores))
        return tmp_path

    def test_plddt_is_binder_only(self, tmp_path):
        # Binder 90, target 40. A whole-complex mean would be ~61 and would fail
        # a ">= 80" gate on an excellent binder.
        rec = parsers.parse_colabfold_complex(self._setup(tmp_path), "d", "seq_0001")
        assert rec.plddt == pytest.approx(90.0)

    def test_parses_iptm_and_pae(self, tmp_path):
        rec = parsers.parse_colabfold_complex(self._setup(tmp_path), "d", "seq_0001")
        assert rec.iptm == pytest.approx(0.82)
        assert rec.pae_interaction == pytest.approx(5.0)

    def test_absent_pae_suggests_the_fix(self, tmp_path):
        # The note must NOT blame --save-all: batch.py gates the pae entry on
        # the model having a PAE head, not on that flag.
        rec = parsers.parse_colabfold_complex(
            self._setup(tmp_path, with_pae=False), "d", "seq_0001"
        )
        assert np.isnan(rec.pae_interaction)
        assert any("alphafold2_multimer_v3" in n for n in rec.notes)
        assert not any("--save-all" in n for n in rec.notes)


class TestRecordRows:
    def test_monomer_row_is_engine_prefixed(self):
        row = parsers.MonomerRecord("d0", "s1", "esmfold", plddt=90.0, scrmsd=1.2).as_row()
        assert row["esmfold_plddt"] == 90.0
        assert row["esmfold_scrmsd"] == 1.2

    def test_complex_rows_from_two_engines_do_not_collide(self):
        # Step 4 runs both engines on the same candidate; their columns must sit
        # side by side so the comparison the MANIFEST asks for is possible.
        a = parsers.ComplexRecord("d0", "s1", "boltz2", iptm=0.9).as_row()
        b = parsers.ComplexRecord("d0", "s1", "colabfold", iptm=0.7).as_row()
        assert set(a) & set(b) == {"design_id", "sequence_id"}
        merged = {**a, **b}
        assert merged["boltz2_iptm"] == 0.9
        assert merged["colabfold_iptm"] == 0.7


MPNN_FASTA = """\
>design_0, score=1.1066, global_score=1.1066, fixed_chains=['B'], designed_chains=['A'], model_name=v_48_020, seed=37
MKTAYIAKQRQISFVK/TARGETSEQ
>T=0.1, sample=1, score=0.7073, global_score=0.9001, seq_recovery=0.5170
MKTAYIAKQRQIAAVK/TARGETSEQ
>T=0.1, sample=2, score=0.8120, global_score=0.9500, seq_recovery=0.4800
MKTWYIAKQRQISFVK/TARGETSEQ
"""


class TestProteinMpnnFasta:
    def _write(self, tmp_path, text=MPNN_FASTA):
        path = tmp_path / "seqs.fa"
        path.write_text(text)
        return path

    def test_native_separated_from_samples(self, tmp_path):
        # The native must not be carried into Step 3 as if it were a design.
        native, samples = parsers.parse_proteinmpnn_fasta(self._write(tmp_path), "design_0")
        assert native is not None and native.is_native
        assert len(samples) == 2
        assert all(not s.is_native for s in samples)

    def test_reads_scores_from_headers(self, tmp_path):
        _, samples = parsers.parse_proteinmpnn_fasta(self._write(tmp_path), "design_0")
        assert samples[0].score == pytest.approx(0.7073)
        assert samples[0].global_score == pytest.approx(0.9001)
        assert samples[0].seq_recovery == pytest.approx(0.5170)
        assert samples[0].temperature == pytest.approx(0.1)
        assert samples[0].sample == 1

    def test_bracketed_header_values_do_not_break_splitting(self, tmp_path):
        # fixed_chains=['A', 'C'] contains a comma inside brackets; splitting
        # naively on ',' would corrupt every field after it.
        text = MPNN_FASTA.replace("fixed_chains=['B']", "fixed_chains=['B', 'C']")
        native, _ = parsers.parse_proteinmpnn_fasta(self._write(tmp_path, text), "design_0")
        assert native.score == pytest.approx(1.1066)

    def test_chains_split_on_slash(self, tmp_path):
        _, samples = parsers.parse_proteinmpnn_fasta(self._write(tmp_path), "design_0")
        assert samples[0].chains == ["MKTAYIAKQRQIAAVK", "TARGETSEQ"]
        assert samples[0].chain(0) == "MKTAYIAKQRQIAAVK"

    def test_sequence_ids_are_unique(self, tmp_path):
        _, samples = parsers.parse_proteinmpnn_fasta(self._write(tmp_path), "design_0")
        assert len({s.sequence_id for s in samples}) == len(samples)

    def test_native_only_file_raises(self, tmp_path):
        text = "\n".join(MPNN_FASTA.splitlines()[:2]) + "\n"
        with pytest.raises(parsers.ParseError, match="no samples"):
            parsers.parse_proteinmpnn_fasta(self._write(tmp_path, text), "design_0")

    def test_empty_file_raises(self, tmp_path):
        with pytest.raises(parsers.ParseError, match="no FASTA records"):
            parsers.parse_proteinmpnn_fasta(self._write(tmp_path, ""), "design_0")

    def test_row_length_excludes_chain_separator(self, tmp_path):
        _, samples = parsers.parse_proteinmpnn_fasta(self._write(tmp_path), "design_0")
        row = samples[0].as_row()
        assert row["length"] == len("MKTAYIAKQRQIAAVK") + len("TARGETSEQ")


class TestBinderFasta:
    def test_writes_binder_chain_only(self, tmp_path):
        # Feeding the full multi-chain string to ESMFold would fold target and
        # binder as one concatenated sequence, making every scRMSD meaningless.
        path = tmp_path / "seqs.fa"
        path.write_text(MPNN_FASTA)
        _, samples = parsers.parse_proteinmpnn_fasta(path, "design_0")
        out = parsers.binder_fasta(samples, tmp_path / "binders.fa")
        text = out.read_text()
        assert "TARGETSEQ" not in text
        assert "MKTAYIAKQRQIAAVK" in text
        assert text.count(">") == 2

    def test_out_of_range_chain_index_raises(self, tmp_path):
        path = tmp_path / "seqs.fa"
        path.write_text(MPNN_FASTA)
        _, samples = parsers.parse_proteinmpnn_fasta(path, "design_0")
        with pytest.raises(parsers.ParseError, match="index 5 requested"):
            parsers.binder_fasta(samples, tmp_path / "b.fa", binder_chain_index=5)


class TestMpnnFilter:
    THRESHOLDS = {
        "mpnn_score_max": 1.25,
        "mpnn_score_preferred_max": 1.10,
        "mpnn_score_marginal_max": 1.35,
    }

    def _samples(self, tmp_path, text=MPNN_FASTA):
        path = tmp_path / "seqs.fa"
        path.write_text(text)
        _, samples = parsers.parse_proteinmpnn_fasta(path, "design_0")
        return samples

    def test_keeps_sequences_at_or_below_the_gate(self, tmp_path):
        kept, dropped = parsers.filter_mpnn_samples(self._samples(tmp_path), self.THRESHOLDS)
        assert len(kept) == 2 and dropped == []
        assert [s.tier for s in kept] == ["preferred", "preferred"]

    def test_drops_above_the_gate_with_a_reason(self, tmp_path):
        text = MPNN_FASTA.replace("score=0.7073", "score=1.3000")
        kept, dropped = parsers.filter_mpnn_samples(self._samples(tmp_path, text), self.THRESHOLDS)
        assert [s.sample for s in kept] == [2]
        (sample, reason) = dropped[0]
        assert sample.tier == "marginal"
        assert "<= 1.25" in reason

    def test_unparsed_score_is_dropped_as_parser_failure(self, tmp_path):
        text = MPNN_FASTA.replace("score=0.7073", "score=n/a")
        kept, dropped = parsers.filter_mpnn_samples(self._samples(tmp_path, text), self.THRESHOLDS)
        assert len(kept) == 1
        (sample, reason) = dropped[0]
        assert sample.tier == "unknown"
        assert "parser failure" in reason

    def test_tier_reaches_the_summary_row(self, tmp_path):
        kept, _ = parsers.filter_mpnn_samples(self._samples(tmp_path), self.THRESHOLDS)
        assert kept[0].as_row()["mpnn_tier"] == "preferred"

    def test_unfiltered_sample_has_no_tier(self, tmp_path):
        assert self._samples(tmp_path)[0].as_row()["mpnn_tier"] is None

    def test_binder_fasta_writes_only_survivors(self, tmp_path):
        text = MPNN_FASTA.replace("score=0.7073", "score=1.4000")
        kept, _ = parsers.filter_mpnn_samples(self._samples(tmp_path, text), self.THRESHOLDS)
        out = parsers.binder_fasta(kept, tmp_path / "binders.fa")
        assert out.read_text().count(">") == 1


class TestStep2GateConfig:
    def test_campaign_yaml_defines_the_step2_gate(self):
        import yaml

        root = Path(__file__).resolve().parents[1]
        gates = yaml.safe_load((root / "config/campaign.yaml").read_text())["gates"]
        block = gates["step2_proteinmpnn"]
        # Thresholds are read from config, never hardcoded in the pipeline.
        assert block["mpnn_score_preferred_max"] <= block["mpnn_score_max"]
        assert block["mpnn_score_max"] <= block["mpnn_score_marginal_max"]
