"""Visualization tests.

py3Dmol views cannot be meaningfully asserted on headlessly, so these cover the
.pml generators (durable artefacts, and the thing a reviewer actually opens)
plus the selector arithmetic that both paths share.
"""

from pathlib import Path

import pytest

from pipeline import viz
from .test_docker import strip_comments

PDB_DIR = Path(__file__).resolve().parents[1] / "examples/input_pdbs"
INSULIN = PDB_DIR / "insulin_target.pdb"
YCR = PDB_DIR / "1YCR.pdb"


class TestSelectors:
    """The two residue-selector vocabularies.

    Both fail silently on the other's syntax, so each is pinned separately.
    3Dmol.js has no special case for `resi`: it compares each value against the
    numeric atom.resi with loose equality, so a LIST of numeric strings matches
    while a comma-joined string is compared as one value and matches nothing.
    PyMOL wants '+' instead.
    """

    def test_resi_selector_strips_chain_letters(self):
        assert viz._resi_selector(["A59", "A83", "A91"]) == ["59", "83", "91"]

    def test_resi_selector_returns_a_list_not_a_joined_string(self):
        # Pins the 3Dmol semantics above: a str here would select nothing.
        selector = viz._resi_selector(["A59", "A83"])
        assert isinstance(selector, list)
        assert all(isinstance(item, str) for item in selector)

    def test_pymol_resi_selector_joins_with_plus(self):
        assert viz._pymol_resi_selector(["A59", "A83", "A91"]) == "59+83+91"

    def test_selectors_do_not_share_syntax(self):
        labels = ["A59", "A83", "A91"]
        assert "," not in viz._pymol_resi_selector(labels)

    @pytest.mark.parametrize("selector", [viz._resi_selector, viz._pymol_resi_selector])
    def test_rejects_malformed_label(self, selector):
        with pytest.raises(ValueError, match="cannot parse residue label"):
            selector(["notaresidue"])


class TestPmlHotspots:
    def test_emits_pymol_resi_syntax(self, tmp_path):
        # PyMOL uses '+' between residue numbers; a comma would start a new
        # argument to `select`, yielding a selection that matches nothing.
        out = viz.write_pml_hotspots(INSULIN, ["A59", "A83", "A91"], tmp_path / "h.pml")
        text = out.read_text()
        assert "resi 59+83+91" in text
        assert "chain A" in text
        assert f"color {viz.PYMOL_HOTSPOT_COLOUR}, hotspots" in text

    def test_uses_only_colours_pymol_knows(self, tmp_path):
        # The defect this guards: "darkred" is a CSS name py3Dmol accepts but
        # PyMOL has no such colour, and `color` on an unknown name errors and
        # leaves the selection its previous colour -- an uncoloured scene
        # rather than a visible failure.
        text = viz.write_pml_hotspots(INSULIN, ["A59"], tmp_path / "h.pml").read_text()
        # strip_comments per the repo convention: a .pml that documents why
        # darkred is wrong would otherwise match its own explanation.
        body = strip_comments(text)
        for css_only in ("darkred", "lightgrey", "lightgray"):
            assert css_only not in body

    def test_embeds_absolute_path(self, tmp_path):
        # .pml files get opened from arbitrary working directories.
        out = viz.write_pml_hotspots(INSULIN, ["A59"], tmp_path / "h.pml")
        assert str(INSULIN.resolve()) in out.read_text()

    def test_creates_parent_directory(self, tmp_path):
        out = viz.write_pml_hotspots(INSULIN, ["A59"], tmp_path / "deep/nested/h.pml")
        assert out.exists()


class TestPmlSuperposition:
    def test_disables_outlier_rejection_for_comparability(self, tmp_path):
        # PyMOL's align defaults to refinement cycles that drop outliers, which
        # would report a better RMSD than metrics.scrmsd computes over all atoms.
        out = viz.write_pml_superposition(YCR, YCR, tmp_path / "s.pml")
        text = out.read_text()
        assert "cycles=0" in text
        assert "color cyan, design" in text
        assert "color magenta, predicted" in text


class TestPmlInterface:
    def test_colours_binder_by_bfactor(self, tmp_path):
        out = viz.write_pml_interface(YCR, tmp_path / "i.pml", binder_chain="B",
                                      target_chain="A")
        text = out.read_text()
        assert "spectrum b," in text
        assert "chain B" in text

    def test_emits_interface_contacts(self, tmp_path):
        text = viz.write_pml_interface(YCR, tmp_path / "i.pml").read_text()
        assert "distance polar_contacts" in text
        assert "within 5 of" in text

    def test_chain_arguments_are_respected(self, tmp_path):
        # Chain roles come from the .trb and are NOT always binder=A; the script
        # must honour whatever is passed.
        text = viz.write_pml_interface(
            YCR, tmp_path / "i.pml", binder_chain="A", target_chain="B"
        ).read_text()
        assert "color grey80, chain B" in text
        assert "spectrum b, red_yellow_green_blue, chain A" in text


class TestChainSummary:
    def test_reports_chains(self):
        summary = viz.chain_summary(YCR)
        assert "chain A" in summary and "chain B" in summary


class TestMonomerSuperposition:
    """The prediction must be moved onto the design before rendering.

    py3Dmol does not superpose. A design pose and a prediction of it are in
    unrelated frames -- RFdiffusion recentres, predictors emit their own -- so
    rendered as loaded they sit tens of angstroms apart and the view says
    nothing about the fold. The fit is `metrics.backbone_superposition`, so the
    separation on screen is the scRMSD in the table.
    """

    @staticmethod
    def _rotated_copy(chain, path, degrees=30.0, shift=(25.0, -10.0, 7.0)):
        """`chain` in a different rigid-body frame, as a prediction would be."""
        import numpy as np
        from pipeline import pdb_io, viz

        theta = np.radians(degrees)
        rotation = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ])
        moved = viz._transform_chain(chain, rotation, np.array(shift))
        return pdb_io.write_pdb([moved], path)

    @pytest.fixture
    def pair(self, tmp_path):
        from pipeline import pdb_io

        design = pdb_io.read_pdb(INSULIN)["A"]
        predicted_pdb = self._rotated_copy(design, tmp_path / "pred.pdb")
        return INSULIN, predicted_pdb, design

    def test_removes_the_frame_offset(self, pair):
        import numpy as np
        from pipeline import pdb_io

        design_pdb, predicted_pdb, design = pair
        before = np.linalg.norm(
            pdb_io.read_pdb(predicted_pdb).only_chain().ca_coords()[0].mean(0)
            - design.ca_coords()[0].mean(0)
        )
        assert before > 10.0, "fixture should start in a different frame"

        text = viz._superpose_prediction(predicted_pdb, design_pdb, "A")
        out = pdb_io.read_pdb(_as_file(text, predicted_pdb.parent)).only_chain()
        after = np.linalg.norm(
            out.ca_coords()[0].mean(0) - design.ca_coords()[0].mean(0)
        )
        assert after < 0.1, f"still {after:.2f} A apart after superposition"

    def test_residual_on_screen_equals_the_reported_scrmsd(self, pair):
        # A rigid-body copy: the gate reads ~0, so the moved coordinates must
        # land on the design rather than merely nearby.
        import numpy as np
        from pipeline import metrics, pdb_io

        design_pdb, predicted_pdb, design = pair
        predicted = pdb_io.read_pdb(predicted_pdb).only_chain()
        reported = metrics.scrmsd(predicted, design)

        text = viz._superpose_prediction(predicted_pdb, design_pdb, "A")
        moved = pdb_io.read_pdb(_as_file(text, predicted_pdb.parent)).only_chain()
        # No further fitting: both are in one frame now, so a plain per-atom
        # RMS must reproduce the metric.
        a, _ = moved.backbone_coords()
        b, _ = design.backbone_coords()
        diff = (a - b).reshape(-1, 3)
        in_frame = float(np.sqrt((diff**2).sum() / len(diff)))
        assert in_frame == pytest.approx(reported, abs=1e-3)

    def test_carries_the_plddt_column_through(self, pair):
        # The B-factor column is the funnel's one agreed pLDDT source; a
        # transform that dropped it would silently zero every pLDDT read off
        # the rendered file.
        from pipeline import pdb_io

        design_pdb, predicted_pdb, _ = pair
        text = viz._superpose_prediction(predicted_pdb, design_pdb, "A")
        moved = pdb_io.read_pdb(_as_file(text, predicted_pdb.parent)).only_chain()
        original = pdb_io.read_pdb(predicted_pdb).only_chain()
        assert moved.residues[0].bfactors == original.residues[0].bfactors

    def test_mismatched_lengths_are_refused(self, tmp_path):
        from pipeline import pdb_io

        design = pdb_io.read_pdb(INSULIN)["A"]
        truncated = pdb_io.Chain(chain_id="A", residues=design.residues[:40])
        short = pdb_io.write_pdb([truncated], tmp_path / "short.pdb")
        with pytest.raises(ValueError, match="pair residues positionally"):
            viz._superpose_prediction(short, INSULIN, "A")

    def test_view_falls_back_unaligned_and_says_so(self, tmp_path, capsys):
        # A failed fit must not pass for a catastrophic prediction: the note
        # goes to stdout AND the caption, which is the only thing a reader of
        # the rendered cell sees.
        pytest.importorskip("py3Dmol")
        from pipeline import pdb_io

        design = pdb_io.read_pdb(INSULIN)["A"]
        truncated = pdb_io.Chain(chain_id="A", residues=design.residues[:40])
        short = pdb_io.write_pdb([truncated], tmp_path / "short.pdb")

        view = viz.show_monomer_superposition(INSULIN, short, scrmsd=1.2)
        assert "UNALIGNED" in capsys.readouterr().out
        assert "UNALIGNED" in view.startjs + view.endjs

    def test_view_renders_both_models(self, pair):
        pytest.importorskip("py3Dmol")
        design_pdb, predicted_pdb, _ = pair
        view = viz.show_monomer_superposition(design_pdb, predicted_pdb, scrmsd=0.0)
        js = view.startjs + view.endjs
        assert js.count("addModel") == 2
        assert "UNALIGNED" not in js


def _as_file(pdb_text: str, directory: Path) -> Path:
    path = directory / "superposed.pdb"
    path.write_text(pdb_text)
    return path
