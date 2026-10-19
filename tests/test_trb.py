"""RFdiffusion .trb parsing, against this project's own real output.

Two fixtures, because the .trb key layout is conditional on run mode:

  motifscaffolding_0  contig 10-40/A163-181/10-40 on 5TPN -- fixed residues in
                      `con_hal_idx0`
  ppi_binder_0        contig A1-150/0 70-100 on insulin_target -- fixed
                      residues in `complex_con_hal_idx0`, `con_hal_idx0` EMPTY

Both are real runs (NVIDIA A10), so these tests exercise the actual file format
rather than a mock of it. Covering only the first is what let read_trb() ship
raising on every PPI design while this suite stayed green; see
tests/fixtures/README.md.
"""

from pathlib import Path

import numpy as np
import pytest

from pipeline import trb

ROOT = Path(__file__).resolve().parents[1]
# Committed fixtures, so the suite works after a fresh clone;
# outputs/ is gitignored. Falls back to outputs/ if fixtures are absent.
FIXTURES = Path(__file__).resolve().parent / "fixtures"
TRB = FIXTURES / "motifscaffolding_0.trb"
PDB = FIXTURES / "motifscaffolding_0.pdb"
if not TRB.exists():
    TRB = ROOT / "outputs/motifscaffolding_0.trb"
    PDB = ROOT / "outputs/motifscaffolding_0.pdb"

pytestmark = pytest.mark.skipif(
    not TRB.exists(), reason="prior RFdiffusion output not present"
)


@pytest.fixture(scope="module")
def info():
    return trb.read_trb(TRB)


class TestReadTrb:
    def test_design_length_matches_pdb(self, info):
        assert info.design_length == 67
        assert len(trb.design_file_order(PDB)) == 67

    def test_reference_and_diffused_partition_the_design(self, info):
        assert info.reference_idx0.size == 19
        assert info.diffused_idx0.size == 48
        overlap = np.intersect1d(info.reference_idx0, info.diffused_idx0)
        assert overlap.size == 0
        assert info.reference_idx0.size + info.diffused_idx0.size == info.design_length

    def test_idx0_agrees_with_pdb_file_order(self, info):
        # The assumption everything else rests on: .trb index i is the i-th
        # residue in PDB file order. If this breaks, every chain role and every
        # residue mapping is wrong while still looking plausible.
        order = trb.design_file_order(PDB)
        for idx, expected in zip(info.reference_idx0, info.reference_design_keys):
            assert order[int(idx)] == expected

    def test_config_is_recovered(self, info):
        assert info.contigs == ["10-40/A163-181/10-40"]
        assert info.input_pdb.endswith("5TPN.pdb")
        assert info.sampled_mask == ["29-29/A163-181/19-19"]

    def test_motif_scaffolding_run_has_no_hotspots(self, info):
        # This run specified none; reading None back (rather than inventing a
        # default) is what makes a silently-dropped hotspot string detectable.
        assert info.hotspots is None

    def test_plddt_is_normalised_to_0_100(self, info):
        # RFdiffusion writes 0-1; the measured final-step mean is 0.97.
        plddt = info.final_plddt()
        assert plddt.size == info.design_length
        assert 90.0 < plddt.mean() < 100.0

    def test_missing_file_raises_trberror(self, tmp_path):
        with pytest.raises(trb.TrbError, match="cannot read"):
            trb.read_trb(tmp_path / "absent.trb")

    def test_trb_path_for_locates_sibling(self):
        assert trb.trb_path_for(PDB) == TRB

    def test_trb_path_for_missing_sibling_raises(self, tmp_path):
        orphan = tmp_path / "x.pdb"
        orphan.write_text("END\n")
        with pytest.raises(trb.TrbError, match="no .trb beside"):
            trb.trb_path_for(orphan)


class TestRenumbering:
    def test_input_numbering_differs_from_design_numbering(self, info):
        # The bug this module exists to prevent: input A163-181 became design
        # A30-48, so a hotspot given in input numbering addresses the wrong
        # residue if applied directly to the design PDB.
        assert info.reference_input_keys[0] == ("A", 163)
        assert info.reference_design_keys[0] == ("A", 30)

    def test_maps_input_labels_to_design_labels(self, info):
        assert info.input_to_design(["A163", "A170", "A181"]) == ["A30", "A37", "A48"]

    def test_unmapped_residue_raises_rather_than_dropping(self, info):
        # Silently skipping would understate how many hotspots went unengaged.
        with pytest.raises(trb.TrbError, match="no counterpart in the design"):
            info.input_to_design(["A59"])


class TestClassifyChains:
    def test_motif_scaffolding_is_not_a_binder_design(self, info):
        # Reference and diffused residues share chain A here, so there is no
        # binder/target split to find. Reporting that beats inventing one.
        roles = trb.classify_chains(info, PDB)
        assert not roles.is_binder_design
        assert roles.binder_chain is None
        assert "motif scaffolding, not binder design" in roles.detail

    def test_require_raises_for_non_binder_design(self, info):
        with pytest.raises(trb.TrbError, match="not a two-chain binder design"):
            trb.classify_chains(info, PDB).require()

    def test_mismatched_pdb_raises(self, info, tmp_path):
        short = tmp_path / "short.pdb"
        short.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "END\n"
        )
        with pytest.raises(trb.TrbError, match="Mismatched design/metadata"):
            trb.classify_chains(info, short)

    def test_two_chain_binder_design_resolves_roles(self, info, tmp_path, monkeypatch):
        # Synthesises the PPI case this repo has no output for yet: target chain
        # A built from reference residues, binder chain B from diffused ones.
        # Asserts the role assignment is target=A / binder=B -- the opposite of
        # a 'binder is chain A' assumption.
        n_ref, n_diff = 4, 3
        lines, serial = [], 1
        for cid, count in (("A", n_ref), ("B", n_diff)):
            for i in range(1, count + 1):
                for atom in ("N", "CA", "C", "O"):
                    lines.append(
                        f"ATOM  {serial:5d} {atom:<4s} ALA {cid}{i:4d}    "
                        f"{i*3.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 50.00           {atom[0]:>2s}"
                    )
                    serial += 1
            lines.append("TER")
        pdb = tmp_path / "ppi.pdb"
        pdb.write_text("\n".join(lines) + "\nEND\n")

        fake = trb.TrbInfo(
            path=tmp_path / "ppi.trb",
            design_length=n_ref + n_diff,
            reference_idx0=np.arange(n_ref),
            reference_design_keys=[("A", i) for i in range(1, n_ref + 1)],
            reference_input_keys=[("A", 100 + i) for i in range(1, n_ref + 1)],
            sampled_mask=[],
            plddt=np.empty((0, 0)),
            config={},
        )
        roles = trb.classify_chains(fake, pdb)
        assert roles.is_binder_design
        assert roles.require() == ("B", "A")
        assert roles.binder_idx0.tolist() == [4, 5, 6]
        assert roles.target_idx0.tolist() == [0, 1, 2, 3]


class TestDescribe:
    def test_summary_mentions_key_facts(self, info):
        text = trb.describe(info, PDB)
        assert "design length" in text
        assert "67" in text
        assert "hotspots" in text


# --------------------------------------------------------------------------
# PPI binder design
#
# The campaign's own run mode, and a DIFFERENT .trb key layout from motif
# scaffolding: `con_hal_idx0` is empty and the fixed residues live in
# `complex_con_hal_idx0`. Having only the motif fixture above is what let
# read_trb() ship raising on every real design while all tests passed.
# --------------------------------------------------------------------------

PPI_TRB = FIXTURES / "ppi_binder_0.trb"
PPI_PDB = FIXTURES / "ppi_binder_0.pdb"

ppi_only = pytest.mark.skipif(
    not PPI_TRB.exists(), reason="PPI binder fixture not present"
)


@pytest.fixture(scope="module")
def ppi():
    return trb.read_trb(PPI_TRB)


@ppi_only
class TestPpiBinderTrb:
    def test_reads_without_raising(self, ppi):
        # The regression: read_trb() validated mask_1d against con_hal_idx0,
        # which is empty for a PPI run, so every real design raised.
        assert ppi.design_length == 237

    def test_con_hal_idx0_really_is_empty(self):
        # Guards the premise of this whole class. If upstream ever starts
        # populating con_hal_idx0 for PPI runs, this fixture stops testing
        # the fallback path and the selector needs re-auditing.
        import pickle
        with PPI_TRB.open("rb") as fh:
            raw = pickle.load(fh)
        assert len(raw["con_hal_idx0"]) == 0
        assert len(raw["complex_con_hal_idx0"]) == 150

    def test_reference_is_the_target_chain(self, ppi):
        assert ppi.reference_idx0.size == 150
        assert ppi.diffused_idx0.size == 87

    def test_reference_indices_are_in_design_frame(self, ppi):
        # The receptor_* trap: receptor_con_hal_idx0 reads 0..149, but the
        # target actually occupies design positions 87..236. Using it would
        # silently address the binder plus part of the target, with no error.
        assert ppi.reference_idx0.min() == 87
        assert ppi.reference_idx0.max() == 236
        assert ppi.diffused_idx0.min() == 0
        assert ppi.diffused_idx0.max() == 86

    def test_binder_is_chain_a_target_is_chain_b(self, ppi):
        # Contig order lists the target first (`A1-150/0 70-100`) but the
        # BINDER is emitted first, so it takes chain A. config/campaign.yaml
        # had this inverted.
        roles = trb.classify_chains(ppi, PPI_PDB)
        assert roles.is_binder_design
        assert roles.require() == ("A", "B")
        assert roles.binder_idx0.size == 87
        assert roles.target_idx0.size == 150

    def test_hotspots_map_through_renumbering(self, ppi):
        # Input A1-150 becomes design B88-237, so the +87 offset must be
        # applied before a hotspot label touches a design PDB.
        assert ppi.hotspots == ["A59", "A83", "A91"]
        assert ppi.input_to_design(ppi.hotspots) == ["B146", "B170", "B178"]

    def test_contigs_and_sampled_mask(self, ppi):
        assert ppi.contigs == ["A1-150/0 70-100"]
        assert ppi.sampled_mask == ["A1-150/0", "87-87"]

    def test_final_plddt_is_normalised(self, ppi):
        # Stored 0-1; final_plddt() rescales. The TRB_PARAMETER_DEFINITIONS
        # note suggesting `np.mean(trb['plddt']) >= 80` is wrong on both counts
        # -- the array is 2-D (timesteps, residues) and on a 0-1 scale.
        assert ppi.plddt.ndim == 2
        assert ppi.plddt.shape == (50, 237)
        assert ppi.plddt.max() <= 1.0
        final = ppi.final_plddt()
        assert final.size == 237
        assert 99.0 < final.mean() < 100.0


@ppi_only
class TestReferenceFamilySelection:
    """The selector must fail loudly rather than fall back to bad indices."""

    def _raw(self):
        import pickle
        with PPI_TRB.open("rb") as fh:
            return pickle.load(fh)

    def test_raises_when_no_family_matches_the_mask(self, tmp_path):
        import pickle
        raw = self._raw()
        raw["complex_con_hal_idx0"] = raw["receptor_con_hal_idx0"]  # the trap
        path = tmp_path / "wrong_frame.trb"
        with path.open("wb") as fh:
            pickle.dump(raw, fh)

        with pytest.raises(trb.TrbError, match="no reference index family"):
            trb.read_trb(path)

    def test_error_names_the_receptor_trap(self, tmp_path):
        import pickle
        raw = self._raw()
        raw["complex_con_hal_idx0"] = []
        path = tmp_path / "empty.trb"
        with path.open("wb") as fh:
            pickle.dump(raw, fh)

        with pytest.raises(trb.TrbError, match="receptor_con_hal_idx0"):
            trb.read_trb(path)

    def test_unconditional_run_with_no_fixed_residues_is_valid(self, tmp_path):
        # Nothing fixed, nothing masked -- a legitimate unconditional design,
        # not a format error.
        import pickle
        import numpy as np
        raw = self._raw()
        raw["mask_1d"] = np.zeros(100, dtype=bool)
        for key in ("con_hal_idx0", "complex_con_hal_idx0",
                    "con_hal_pdb_idx", "complex_con_hal_pdb_idx"):
            raw[key] = []
        path = tmp_path / "unconditional.trb"
        with path.open("wb") as fh:
            pickle.dump(raw, fh)

        info = trb.read_trb(path)
        assert info.reference_idx0.size == 0
        assert info.diffused_idx0.size == 100
