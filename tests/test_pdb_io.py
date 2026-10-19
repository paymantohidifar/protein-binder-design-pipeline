"""Parser robustness against the real PDB files in this repo.

MANIFEST Section 5 instruction 2 requires graceful handling of missing residues
and missing chain identifiers. These tests use actual files rather than
synthetic ones so the defects are the defects that will really arrive.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from pipeline import pdb_io

PDB_DIR = Path(__file__).resolve().parents[1] / "examples/input_pdbs"
INSULIN = PDB_DIR / "insulin_target.pdb"
TPN = PDB_DIR / "5TPN.pdb"
YCR = PDB_DIR / "1YCR.pdb"


class TestInsulinTarget:
    """The campaign target. Verified: chain A, 150 contiguous residues."""

    def test_single_chain_150_residues(self):
        chain = pdb_io.read_pdb(INSULIN).only_chain()
        assert chain.chain_id == "A"
        assert len(chain) == 150

    def test_numbering_is_contiguous_1_to_150(self):
        chain = pdb_io.read_pdb(INSULIN)["A"]
        assert chain.residue_ids() == list(range(1, 151))
        assert chain.gaps() == []

    def test_glycine_cb_falls_back_to_ca(self):
        # 144 CB atoms for 150 residues: 6 residues have no CB. Returning None
        # for them would silently drop them from every hotspot calculation.
        chain = pdb_io.read_pdb(INSULIN)["A"]
        no_cb = [r for r in chain.residues if "CB" not in r.atoms]
        assert len(no_cb) == 6
        for res in no_cb:
            assert res.cb is not None
            assert np.allclose(res.cb, res.ca)

    def test_every_residue_has_complete_backbone(self):
        chain = pdb_io.read_pdb(INSULIN)["A"]
        coords, keys = chain.backbone_coords()
        assert coords.shape == (150, 4, 3)
        assert len(keys) == 150


class TestMissingResidues:
    """5TPN has unmodelled regions -- the gap case."""

    def test_gaps_are_detected_not_fatal(self):
        chain = pdb_io.read_pdb(TPN).chains["A"]
        gaps = chain.gaps()
        assert gaps, "5TPN is expected to have numbering gaps"
        for prev, nxt in gaps:
            assert nxt > prev + 1

    def test_subset_tolerates_absent_residues(self):
        # The contig A163-181 used in the prior run of this project.
        chain = pdb_io.read_pdb(TPN).chains["A"]
        sub = chain.subset(163, 181)
        assert 0 < len(sub) <= 19
        assert all(163 <= r.seq_id <= 181 for r in sub.residues)

    def test_describe_mentions_gaps(self):
        text = pdb_io.describe(pdb_io.read_pdb(TPN))
        assert "numbering gap" in text


class TestMultiChain:
    def test_two_chains_resolved_separately(self):
        struct = pdb_io.read_pdb(YCR)
        assert struct.chain_ids == ["A", "B"]
        assert len(struct["A"]) == 85
        assert len(struct["B"]) == 13

    def test_missing_chain_raises_with_available_listed(self):
        struct = pdb_io.read_pdb(YCR)
        with pytest.raises(KeyError, match="present"):
            struct["Z"]

    def test_only_chain_refuses_ambiguity(self):
        with pytest.raises(ValueError, match="2 chains"):
            pdb_io.read_pdb(YCR).only_chain()


class TestBlankChainId:
    def test_blank_chain_id_is_collected_under_sentinel(self, tmp_path):
        # Dropping these atoms would corrupt downstream RMSDs rather than fail
        # loudly, so they must land somewhere addressable.
        pdb = tmp_path / "blank.pdb"
        pdb.write_text(
            "ATOM      1  N   ALA     1       0.000   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA     1       1.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA     1       2.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      4  O   ALA     1       2.000   1.000   0.000  1.00  0.00           O\n"
            "END\n"
        )
        struct = pdb_io.read_pdb(pdb)
        assert struct.chain_ids == [pdb_io.DEFAULT_CHAIN]
        chain = struct.only_chain()
        assert chain.had_blank_id and len(chain) == 1

    def test_round_trip_preserves_blank_id(self, tmp_path):
        pdb = tmp_path / "blank.pdb"
        pdb.write_text(
            "ATOM      1  CA  GLY     7      1.000   2.000   3.000  1.00  0.00           C\n"
            "END\n"
        )
        chain = pdb_io.read_pdb(pdb).only_chain()
        out = pdb_io.write_pdb([chain], tmp_path / "out.pdb")
        assert pdb_io.read_pdb(out).only_chain().had_blank_id


class TestMalformedInput:
    def test_bad_coordinate_raises_structure_error(self, tmp_path):
        # Bio.PDB aborts on an unparseable coordinate rather than inventing a
        # position ("Easier to abort!" in its source), so unlike the previous
        # hand-rolled parser this costs the whole file, not one atom. Wrapped in
        # StructureError -- a ValueError subclass -- so the per-candidate
        # handlers in parsers/ catch it and the campaign continues.
        pdb = tmp_path / "bad.pdb"
        pdb.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      2  CA  ALA A   2         NaNx   0.000   0.000  1.00  0.00           C\n"
            "END\n"
        )
        with pytest.raises(pdb_io.StructureError, match="cannot parse"):
            pdb_io.read_pdb(pdb)

    def test_structure_error_is_a_valueerror(self):
        # The property that keeps one corrupt design from aborting a campaign.
        assert issubclass(pdb_io.StructureError, ValueError)

    def test_altloc_b_is_ignored(self, tmp_path):
        pdb = tmp_path / "alt.pdb"
        pdb.write_text(
            "ATOM      1  CA AALA A   1       0.000   0.000   0.000  0.60  0.00           C\n"
            "ATOM      2  CA BALA A   1       9.000   9.000   9.000  0.40  0.00           C\n"
            "END\n"
        )
        chain = pdb_io.read_pdb(pdb)["A"]
        assert len(chain) == 1
        assert np.allclose(chain.residues[0].ca, [0.0, 0.0, 0.0])


class TestTrajectoryModels:
    def test_only_first_model_is_read(self, tmp_path):
        # RFdiffusion trajectory files concatenate every denoising step as a
        # model; reading all of them would stack timesteps into one chain.
        pdb = tmp_path / "traj.pdb"
        pdb.write_text(
            "MODEL        1\n"
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "ENDMDL\n"
            "MODEL        2\n"
            "ATOM      2  CA  ALA A   1       9.000   9.000   9.000  1.00  0.00           C\n"
            "ENDMDL\n"
        )
        chain = pdb_io.read_pdb(pdb)["A"]
        assert len(chain) == 1
        assert np.allclose(chain.residues[0].ca, [0.0, 0.0, 0.0])


class TestRenumber:
    def test_collapses_gaps(self):
        chain = pdb_io.read_pdb(TPN).chains["A"]
        out = pdb_io.renumber(chain)
        assert out.residue_ids() == list(range(1, len(chain) + 1))
        assert out.gaps() == []


class TestRealRfdiffusionOutput:
    """The project's own prior RFdiffusion output must parse."""

    def test_prior_design_parses(self):
        out = Path(__file__).resolve().parent / "fixtures/motifscaffolding_0.pdb"
        if not out.exists():
            pytest.skip("prior RFdiffusion output not present")
        struct = pdb_io.read_pdb(out)
        chain = struct.chains[struct.chain_ids[0]]
        assert len(chain) > 0
        coords, _ = chain.backbone_coords()
        assert coords.shape[1:] == (4, 3)


class TestElementInference:
    """The element column written for each atom.

    Bio.PDB falls back to guessing -- with a warning -- when the column holds a
    non-element, and the guess feeds atomic radii, so a wrong column silently
    changes any SASA computed on a re-read file.
    """

    def test_backbone_and_sidechain_atoms(self):
        for name, element in [
            ("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"),
            ("CB", "C"), ("OG1", "O"), ("SD", "S"), ("NZ", "N"),
        ]:
            assert pdb_io._element_for(name) == element, name

    def test_hydrogens_with_leading_branch_digit(self):
        # The defect: name[0] gives "3", not an element.
        for name in ("3HB", "1HD2", "2HG1"):
            assert pdb_io._element_for(name) == "H", name

    def test_round_trip_emits_no_construction_warning(self, tmp_path):
        chain = pdb_io.read_pdb(INSULIN)["A"]
        out = pdb_io.write_pdb([chain], tmp_path / "rt.pdb")
        with warnings.catch_warnings():
            warnings.simplefilter("error", PDBConstructionWarning)
            pdb_io.read_pdb(out)


class TestToBioModel:
    def test_preserves_chain_order_not_id_order(self):
        s = pdb_io.read_pdb(YCR)
        model = pdb_io.to_bio_model([s["B"], s["A"]])
        assert [c.id for c in model] == ["B", "A"]

    def test_carries_every_residue_and_atom(self):
        chain = pdb_io.read_pdb(INSULIN)["A"]
        model = pdb_io.to_bio_model([chain])
        bio_chain = next(iter(model))
        assert len(list(bio_chain)) == len(chain)
        assert sum(len(list(r)) for r in bio_chain) == sum(
            len(r.atoms) for r in chain.residues
        )
