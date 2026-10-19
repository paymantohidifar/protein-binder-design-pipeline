"""Hotspot selection, validated against the known-good answer.

examples/design_ppi.sh targets A59/A83/A91 on the insulin receptor. Geometry
alone should agree with that choice, which makes this a real check on the
selection code with no GPU involved.
"""

from pathlib import Path

import numpy as np
import pytest

from pipeline import hotspots, pdb_io

PDB_DIR = Path(__file__).resolve().parents[1] / "examples/input_pdbs"
INSULIN = PDB_DIR / "insulin_target.pdb"
YCR = PDB_DIR / "1YCR.pdb"
KNOWN_HOTSPOTS = ["A59", "A83", "A91"]


@pytest.fixture(scope="module")
def target():
    return pdb_io.read_pdb(INSULIN)["A"]


@pytest.fixture(scope="module")
def mdm2_p53():
    """1YCR: MDM2 (chain A, 85 res) bound to the p53 transactivation helix
    (chain B, 13 res). A complex with a literature-known interface, so the
    contact code can be checked against an answer it was not tuned on."""
    s = pdb_io.read_pdb(YCR)
    return s["A"], s["B"]


class TestContactsFromComplex:
    """The RFdiffusion definition, measured: target residues within 10 A Cb of
    the binder."""

    def test_recovers_the_p53_f19_w23_l26_triad(self, mdm2_p53):
        # The headline validation. F19/W23/L26 are the three p53 side chains
        # that insert into the MDM2 cleft -- the canonical example of a
        # three-residue hotspot patch. Ranking p53 residues by contact count
        # must put exactly these on top, and nothing in the code was fitted to
        # this answer.
        mdm2, p53 = mdm2_p53
        top3 = hotspots.contacts_from_complex(p53, mdm2)[:3]
        assert {c.residue_id for c in top3} == {19, 23, 26}
        assert [c.residue_name for c in top3[:2]] == ["PHE", "TRP"]

    def test_recovers_known_mdm2_pocket_residues(self, mdm2_p53):
        # The MDM2 side of the same interface, from the literature.
        mdm2, p53 = mdm2_p53
        found = {c.residue_id for c in hotspots.contacts_from_complex(mdm2, p53)}
        known_pocket = {54, 58, 61, 62, 67, 72, 75, 93, 96, 99}
        assert known_pocket <= found, f"missed: {known_pocket - found}"

    def test_ranks_by_contact_count_descending(self, mdm2_p53):
        mdm2, p53 = mdm2_p53
        counts = [c.neighbours for c in hotspots.contacts_from_complex(mdm2, p53)]
        assert counts == sorted(counts, reverse=True)
        assert all(c > 0 for c in counts)

    def test_excludes_non_contacting_residues(self, mdm2_p53):
        # 10 A Cb is a generous cutoff, so this is a shortlist to take 3-6 from,
        # not a final answer -- but it must still exclude most of the protein.
        mdm2, p53 = mdm2_p53
        found = hotspots.contacts_from_complex(mdm2, p53)
        assert len(found) < len(mdm2)

    def test_cutoff_is_monotonic(self, mdm2_p53):
        mdm2, p53 = mdm2_p53
        tight = hotspots.contacts_from_complex(mdm2, p53, cutoff=6.0)
        loose = hotspots.contacts_from_complex(mdm2, p53, cutoff=12.0)
        assert {c.residue_id for c in tight} <= {c.residue_id for c in loose}

    def test_labels_use_target_chain_id(self, mdm2_p53):
        mdm2, p53 = mdm2_p53
        assert all(c.label.startswith("A") for c in hotspots.contacts_from_complex(mdm2, p53))


class TestVerifyHotspotContacts:
    """Step 1 QC: did the generated binder actually engage its hotspots?"""

    def test_real_interface_residues_are_engaged(self, mdm2_p53):
        mdm2, p53 = mdm2_p53
        labels = [c.label for c in hotspots.contacts_from_complex(mdm2, p53)[:3]]
        result = hotspots.verify_hotspot_contacts(mdm2, p53, labels)
        assert result["all_engaged"]
        assert result["n_engaged"] == 3
        assert all(v <= 10.0 for v in result["min_cb_distance"].values())

    def test_buried_residue_on_far_face_is_not_engaged(self, mdm2_p53):
        # The failure this QC exists to catch: a hotspot the binder never
        # reaches. MDM2 A30 is the most buried residue, 15.19 A from the helix.
        mdm2, p53 = mdm2_p53
        result = hotspots.verify_hotspot_contacts(mdm2, p53, ["A30"])
        assert not result["all_engaged"]
        assert result["n_engaged"] == 0
        assert result["min_cb_distance"]["A30"] == pytest.approx(15.19, abs=0.05)

    def test_reports_per_hotspot_detail(self, mdm2_p53):
        # Partial engagement must be visible per-residue, not collapsed into a
        # single boolean -- 2-of-3 engaged is diagnostic of a usable design.
        mdm2, p53 = mdm2_p53
        result = hotspots.verify_hotspot_contacts(mdm2, p53, ["A93", "A30"])
        assert result["engaged"] == {"A93": True, "A30": False}
        assert result["n_engaged"] == 1
        assert not result["all_engaged"]

    def test_unknown_residue_raises(self, mdm2_p53):
        mdm2, p53 = mdm2_p53
        with pytest.raises(KeyError, match="absent from chain"):
            hotspots.verify_hotspot_contacts(mdm2, p53, ["A9999"])


class TestKnownHotspots:
    def test_violate_the_manifest_10A_constraint(self, target):
        # Pins Finding A. MANIFEST Section 1 states Cb-Cb <= 10 A, but the
        # known-good set from examples/design_ppi.sh spans 10.44 A. The spec is
        # a misreading: 10 A is the interface *contact* distance, not the patch
        # diameter. If this test ever fails, the measurement changed and the
        # reasoning behind MAX_PATCH_DISTANCE needs revisiting.
        d = hotspots.patch_distances(target, KNOWN_HOTSPOTS)
        off_diagonal = d[~np.eye(len(KNOWN_HOTSPOTS), dtype=bool)]
        assert off_diagonal.max() == pytest.approx(10.44, abs=0.05)
        assert off_diagonal.max() <= hotspots.MAX_PATCH_DISTANCE

    def test_are_candidates_under_our_filter(self, target):
        # Pins Finding B. All three are PHE at 16/18/18 neighbours against a
        # median of 16, so a cutoff of 16 would discard two of them. They must
        # all survive, or select_patch could never propose the known-good site.
        exposed = {c.residue_id for c in hotspots.rank_candidates(target)}
        ids = {59, 83, 91}
        assert ids <= exposed, f"excluded by burial filter: {ids - exposed}"

    def test_rank_above_median_on_identity(self, target):
        # Exposed hydrophobics should surface near the front of the shortlist.
        # Asserting the exact published triple would be overfitting -- epitope
        # choice is a judgement call -- so this only checks they are not buried
        # in the tail of the ranking.
        ranked = [c.residue_id for c in hotspots.rank_candidates(target)]
        positions = [ranked.index(r) for r in (59, 83, 91)]
        assert max(positions) < len(ranked) // 2, f"ranked at {positions} of {len(ranked)}"

    def test_missing_residue_raises_clearly(self, target):
        with pytest.raises(KeyError, match="absent from chain"):
            hotspots.patch_distances(target, ["A59", "A999"])


class TestNeighbourCounts:
    def test_covers_every_residue(self, target):
        counts = hotspots.neighbour_counts(target)
        assert len(counts) == 150

    def test_buried_residues_have_more_neighbours_than_exposed(self, target):
        counts = hotspots.neighbour_counts(target)
        values = np.array(list(counts.values()))
        # A real globular domain must show a spread; a uniform count would mean
        # the distance matrix is wrong.
        assert values.max() - values.min() > 10

    def test_excludes_self(self, target):
        # Self-distance is 0, which is <= radius; failing to subtract it would
        # bias every count by exactly 1.
        single = pdb_io.Chain("A", [target.residues[0]])
        assert hotspots.neighbour_counts(single) == {target.residues[0].seq_id: 0}


class TestSelectPatch:
    def test_returns_requested_count(self, target):
        assert len(hotspots.select_patch(target, n=3)) == 3

    def test_members_are_mutually_close(self, target):
        chosen = hotspots.select_patch(target, n=4, max_cb_distance=10.0)
        labels = [c.label for c in chosen]
        assert len(labels) == 4
        d = hotspots.patch_distances(target, labels)
        off_diagonal = d[~np.eye(len(labels), dtype=bool)]
        assert off_diagonal.max() <= 10.0

    def test_labels_match_rfdiffusion_format(self, target):
        # ppi.hotspot_res expects e.g. A59, not 59 or A.59.
        for cand in hotspots.select_patch(target, n=3):
            assert cand.label.startswith("A")
            assert cand.label[1:].isdigit()

    def test_rejects_count_outside_manifest_range(self, target):
        for n in (2, 7):
            with pytest.raises(ValueError, match="3-6"):
                hotspots.select_patch(target, n=n)

    def test_impossible_distance_constraint_raises(self, target):
        with pytest.raises(ValueError, match="could not assemble"):
            hotspots.select_patch(target, n=6, max_cb_distance=0.5)

    def test_over_strict_burial_cutoff_raises(self, target):
        with pytest.raises(ValueError, match="exposed residues"):
            hotspots.select_patch(target, n=3, burial_cutoff=0)


class TestChainSasa:
    """SASA computed from a Chain in memory, which is what the ranking uses.

    rank_candidates receives a Chain, not a path, so the exposure it ranks on
    has to be computable without a file -- a `subset()` or `renumber()` result
    has none.
    """

    def test_matches_the_path_based_function(self, target):
        # The in-memory model must be the same model Bio.PDB would parse, or
        # the ranking and the notebook's reported SASA would disagree.
        from_path = hotspots.compute_sasa(INSULIN, "A")
        from_chain = hotspots.chain_sasa(target)
        assert set(from_chain) == set(from_path)
        for rid, value in from_path.items():
            assert from_chain[rid] == pytest.approx(value, abs=1e-6)

    def test_relative_matches_the_path_based_function(self, target):
        from_path = hotspots.relative_sasa(INSULIN, "A")
        from_chain = hotspots.chain_relative_sasa(target)
        for rid, value in from_path.items():
            assert from_chain[rid] == pytest.approx(value, abs=1e-6)

    def test_is_a_fraction_of_the_reference_maximum(self, target):
        for rid, value in hotspots.chain_relative_sasa(target).items():
            assert 0.0 <= value <= 1.5, f"A{rid} relative SASA {value}"

    def test_context_chain_buries_the_interface(self, mdm2_p53):
        # A partner chain must reduce exposure; the p53 helix is mostly
        # interface, so the drop is large.
        mdm2, p53 = mdm2_p53
        alone = hotspots.chain_sasa(p53)
        bound = hotspots.chain_sasa(p53, context=[mdm2])
        assert sum(bound.values()) < sum(alone.values())
        assert all(bound[rid] <= alone[rid] + 1e-6 for rid in alone)


class TestRankCandidatesExposure:
    """Ranking is on relative SASA, not the Cb neighbour-count proxy.

    The proxy it replaced was size-blind (Gly and Trp weighted alike) and
    normalised by the most-crowded residue in the chain, so its values were
    not comparable between targets.
    """

    def test_exposure_is_relative_sasa(self, target):
        rel = hotspots.relative_sasa(INSULIN, "A")
        for cand in hotspots.rank_candidates(target):
            assert cand.exposure == pytest.approx(rel[cand.residue_id], abs=1e-6)

    def test_absolute_sasa_is_populated(self, target):
        # Was left at the NaN default by the proxy implementation.
        absolute = hotspots.compute_sasa(INSULIN, "A")
        for cand in hotspots.rank_candidates(target):
            assert not np.isnan(cand.sasa)
            assert cand.sasa == pytest.approx(absolute[cand.residue_id], abs=1e-6)

    def test_neighbours_still_carries_the_burial_count(self, target):
        counts = hotspots.neighbour_counts(target)
        for cand in hotspots.rank_candidates(target):
            assert cand.neighbours == counts[cand.residue_id]

    def test_sorted_by_exposure_within_identity_class(self, target):
        for enriched in (True, False):
            group = [
                c.exposure
                for c in hotspots.rank_candidates(target)
                if c.interface_enriched is enriched
            ]
            assert group == sorted(group, reverse=True)
        # and every enriched residue precedes every non-enriched one
        flags = [c.interface_enriched for c in hotspots.rank_candidates(target)]
        assert flags == sorted(flags, reverse=True)

    def test_exposure_floor_is_applied(self, target):
        # EXPOSURE_FLOOR was defined but unread while the proxy was in place.
        for floor in (0.0, 0.15, 0.4):
            kept = hotspots.rank_candidates(target, exposure_floor=floor)
            assert all(c.exposure >= floor for c in kept)
        assert len(hotspots.rank_candidates(target, exposure_floor=0.4)) < len(
            hotspots.rank_candidates(target, exposure_floor=0.0)
        )

    def test_floor_is_monotonic(self, target):
        tight = {c.residue_id for c in hotspots.rank_candidates(target, exposure_floor=0.4)}
        loose = {c.residue_id for c in hotspots.rank_candidates(target, exposure_floor=0.1)}
        assert tight <= loose

    def test_known_hotspots_clear_the_default_floor(self, target):
        # Companion to TestKnownHotspots: the published set must survive the
        # SASA filter too, and A59 has the least margin of the three.
        rel = hotspots.chain_relative_sasa(target)
        for rid in (59, 83, 91):
            assert rel[rid] >= hotspots.EXPOSURE_FLOOR, f"A{rid} at {rel[rid]:.3f}"

    def test_over_strict_floor_raises_in_select_patch(self, target):
        with pytest.raises(ValueError, match="exposed residues"):
            hotspots.select_patch(target, n=3, exposure_floor=0.99)


class TestContactsFromComplexExposure:
    def test_exposure_is_unbound_relative_sasa(self, mdm2_p53):
        # Deliberately the free target: every residue reported here is next to
        # the partner, so in-complex SASA would be ~0 for all of them.
        mdm2, p53 = mdm2_p53
        rel = hotspots.chain_relative_sasa(p53)
        for cand in hotspots.contacts_from_complex(p53, mdm2):
            assert cand.exposure == pytest.approx(rel[cand.residue_id], abs=1e-6)

    def test_triad_reads_as_exposed_on_the_free_helix(self, mdm2_p53):
        mdm2, p53 = mdm2_p53
        top3 = hotspots.contacts_from_complex(p53, mdm2)[:3]
        assert all(c.exposure > hotspots.EXPOSURE_FLOOR for c in top3)
        assert all(not np.isnan(c.sasa) for c in top3)

    def test_neighbours_still_carries_the_contact_count(self, mdm2_p53):
        mdm2, p53 = mdm2_p53
        found = hotspots.contacts_from_complex(p53, mdm2)
        assert all(c.neighbours > 0 for c in found)
        assert [c.neighbours for c in found] == sorted(
            [c.neighbours for c in found], reverse=True
        )
