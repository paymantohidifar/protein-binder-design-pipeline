"""Metric correctness tests.

These run without a GPU and must pass before any GPU time is spent: a sign or
indexing error in metrics.py produces plausible-looking numbers that silently
invalidate the whole campaign.
"""

import numpy as np
import pytest

from pipeline import metrics
from pipeline.pdb_io import Chain, Residue


def make_chain(coords, chain_id="A", names=None, start=1):
    """Chain whose residues carry a full N/CA/C/O backbone at given offsets."""
    chain = Chain(chain_id)
    for i, xyz in enumerate(coords):
        xyz = np.asarray(xyz, dtype=float)
        name = (names[i] if names else "ALA")
        chain.residues.append(
            Residue(
                seq_id=start + i,
                name=name,
                atoms={
                    "N": xyz + [0.0, 0.0, 0.0],
                    "CA": xyz + [1.0, 0.0, 0.0],
                    "C": xyz + [2.0, 0.0, 0.0],
                    "O": xyz + [2.0, 1.0, 0.0],
                },
            )
        )
    return chain


def transform_chain(chain, rotation, translation):
    """Rigidly move every atom of a chain.

    Transforming the coordinates *before* make_chain would be wrong: make_chain
    adds its backbone offsets in the unrotated frame, so the result would not be
    a rigid motion of the original and a correct metric would report a non-zero
    RMSD for what looks like the same structure.
    """
    out = Chain(chain.chain_id)
    for res in chain.residues:
        out.residues.append(
            Residue(
                seq_id=res.seq_id,
                name=res.name,
                atoms={k: v @ rotation + translation for k, v in res.atoms.items()},
            )
        )
    return out


HELIX = np.array([[i * 1.5, np.sin(i), np.cos(i)] for i in range(20)])


class TestKabsch:
    def test_identical_is_zero(self):
        rmsd, _, _ = metrics.kabsch_rmsd(HELIX, HELIX)
        assert rmsd == pytest.approx(0.0, abs=1e-9)

    def test_translation_is_removed(self):
        moved = HELIX + [10.0, -5.0, 3.0]
        rmsd, _, _ = metrics.kabsch_rmsd(moved, HELIX)
        assert rmsd == pytest.approx(0.0, abs=1e-9)

    def test_rotation_is_removed(self):
        theta = 0.7
        rot = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rmsd, _, _ = metrics.kabsch_rmsd(HELIX @ rot, HELIX)
        assert rmsd == pytest.approx(0.0, abs=1e-9)

    def test_uniform_offset_gives_that_offset(self):
        # Alternating +-0.5 A displacement cannot be removed by any rigid
        # motion, so RMSD lands on 0.5. Not exactly 0.5: Kabsch still finds a
        # marginal rotation that shaves ~3e-6 off, which is correct behaviour
        # for an optimal superposition, so the tolerance is loose enough to
        # allow it while still catching a real scale error.
        noise = np.zeros_like(HELIX)
        noise[:, 0] = np.where(np.arange(len(HELIX)) % 2 == 0, 0.5, -0.5)
        rmsd, _, _ = metrics.kabsch_rmsd(HELIX + noise, HELIX)
        assert rmsd == pytest.approx(0.5, abs=1e-3)

    def test_reflection_is_rejected(self):
        # A mirror image must NOT superpose to 0. Without the det<0 correction
        # a physically impossible design would score as perfect.
        mirrored = HELIX * [1.0, 1.0, -1.0]
        rmsd, rotation, _ = metrics.kabsch_rmsd(mirrored, HELIX)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-9)
        assert rmsd > 0.5

    def test_transform_round_trip(self):
        moved = HELIX @ np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0] + [3, 1, 4]
        _, rot, trans = metrics.kabsch_rmsd(moved, HELIX)
        assert np.allclose(metrics.apply_transform(moved, rot, trans), HELIX, atol=1e-8)

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least|>= 3|need"):
            metrics.kabsch_rmsd(HELIX[:2], HELIX[:2])


class TestScrmsd:
    def test_identical_design_is_zero(self):
        chain = make_chain(HELIX)
        assert metrics.scrmsd(chain, chain) == pytest.approx(0.0, abs=1e-9)

    def test_rigid_motion_is_zero(self):
        # scRMSD superposes the binder on itself, so placement is irrelevant.
        assert metrics.scrmsd(make_chain(HELIX + [50, 0, 0]), make_chain(HELIX)) == pytest.approx(
            0.0, abs=1e-9
        )

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="pair residues positionally"):
            metrics.scrmsd(make_chain(HELIX[:10]), make_chain(HELIX))


class TestBinderRmsdToDesign:
    def test_identical_complex_is_zero(self):
        binder, target = make_chain(HELIX), make_chain(HELIX + [30, 0, 0], chain_id="B")
        assert metrics.binder_rmsd_to_design(binder, target, binder, target) == pytest.approx(
            0.0, abs=1e-9
        )

    def test_whole_complex_rigid_motion_is_zero(self):
        # Superposing on the target removes global placement: the same complex
        # in a different frame must score 0.
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        trans = np.array([7.0, 7.0, 7.0])
        binder = make_chain(HELIX)
        target = make_chain(HELIX + [30, 0, 0], chain_id="B")
        assert metrics.binder_rmsd_to_design(
            transform_chain(binder, rot, trans),
            transform_chain(target, rot, trans),
            binder,
            target,
        ) == pytest.approx(0.0, abs=1e-8)

    def test_binder_displacement_is_detected(self):
        # THE test that separates this metric from scRMSD: binder internally
        # perfect but docked in the wrong place. scRMSD sees 0; this must not.
        binder = make_chain(HELIX)
        target = make_chain(HELIX + [30, 0, 0], chain_id="B")
        shifted = make_chain(HELIX + [5.0, 0.0, 0.0])
        assert metrics.scrmsd(shifted, binder) == pytest.approx(0.0, abs=1e-9)
        assert metrics.binder_rmsd_to_design(shifted, target, binder, target) == pytest.approx(
            5.0, abs=1e-6
        )

    def test_target_mismatch_raises(self):
        binder = make_chain(HELIX)
        with pytest.raises(ValueError, match="target chain mismatch"):
            metrics.binder_rmsd_to_design(
                binder, make_chain(HELIX[:5], chain_id="B"), binder,
                make_chain(HELIX, chain_id="B"),
            )


class TestPaeInteraction:
    def test_averages_both_offdiagonal_blocks(self):
        # Asymmetric by construction: binder->target block is 2, target->binder
        # is 8, intra-chain blocks are 1. Correct answer is (2+8)/2 = 5.
        n = 6
        pae = np.ones((n, n))
        b, t = [0, 1, 2], [3, 4, 5]
        pae[np.ix_(b, t)] = 2.0
        pae[np.ix_(t, b)] = 8.0
        assert metrics.pae_interaction(pae, b, t) == pytest.approx(5.0)

    def test_differs_from_full_matrix_mean(self):
        # Guards the specific error the docstring warns about: the full-matrix
        # mean is dragged down by confident intra-chain blocks, making a bad
        # interface look good.
        n = 20
        pae = np.full((n, n), 1.0)
        b, t = list(range(10)), list(range(10, 20))
        pae[np.ix_(b, t)] = 25.0
        pae[np.ix_(t, b)] = 25.0
        assert metrics.pae_interaction(pae, b, t) == pytest.approx(25.0)
        assert pae.mean() == pytest.approx(13.0)

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            metrics.pae_interaction(np.ones((4, 5)), [0, 1], [2, 3])

    def test_overlapping_indices_raise(self):
        with pytest.raises(ValueError, match="overlap"):
            metrics.pae_interaction(np.ones((4, 4)), [0, 1], [1, 2])

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            metrics.pae_interaction(np.ones((4, 4)), [0, 1], [2, 9])

    def test_empty_set_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            metrics.pae_interaction(np.ones((4, 4)), [], [1, 2])


class TestPlddt:
    def test_esmfold_scale_is_normalised(self):
        assert metrics.plddt_mean([0.9, 0.8, 0.7]) == pytest.approx(80.0)

    def test_colabfold_scale_is_left_alone(self):
        assert metrics.plddt_mean([90.0, 80.0, 70.0]) == pytest.approx(80.0)

    def test_boundary_value_of_one_treated_as_fraction(self):
        assert metrics.plddt_mean([1.0, 1.0]) == pytest.approx(100.0)

    def test_subset_indices(self):
        assert metrics.plddt_mean([10.0, 90.0, 90.0], idx=[1, 2]) == pytest.approx(90.0)


class TestPasses:
    THRESHOLDS = {"plddt_min": 80.0, "scrmsd_max": 2.0}

    def test_passing_record(self):
        ok, reasons = metrics.passes({"plddt": 85.0, "scrmsd": 1.1}, self.THRESHOLDS)
        assert ok and reasons == []

    def test_threshold_breach_is_reported(self):
        ok, reasons = metrics.passes({"plddt": 70.0, "scrmsd": 1.1}, self.THRESHOLDS)
        assert not ok
        assert "plddt=70.000 fails >= 80.0" in reasons[0]

    def test_missing_metric_flagged_as_parser_failure(self):
        # At smoke-test scale this distinction is the whole point: a missing
        # metric means a broken parser, not a bad design.
        ok, reasons = metrics.passes({"scrmsd": 1.1}, self.THRESHOLDS)
        assert not ok
        assert "parser failure" in reasons[0]

    def test_nan_flagged_as_parser_failure(self):
        ok, reasons = metrics.passes({"plddt": float("nan"), "scrmsd": 1.1}, self.THRESHOLDS)
        assert not ok
        assert "parser failure" in reasons[0]

    def test_absent_threshold_is_not_checked(self):
        ok, _ = metrics.passes({"plddt": 85.0}, {"plddt_min": 80.0})
        assert ok


class TestMpnnScoreGate:
    """ProteinMPNN score is a negative log-likelihood: lower is better.

    Bounds mirror gates.step2_proteinmpnn in config/campaign.yaml, duplicated
    here only so the tier ladder is exercised independently of the config file.
    """

    THRESHOLDS = {
        "mpnn_score_max": 1.25,
        "mpnn_score_preferred_max": 1.10,
        "mpnn_score_marginal_max": 1.35,
    }

    @pytest.mark.parametrize(
        "score,tier",
        [
            (0.71, "preferred"),
            (1.10, "preferred"),   # boundaries are inclusive
            (1.18, "acceptable"),
            (1.25, "acceptable"),
            (1.30, "marginal"),
            (1.35, "marginal"),
            (1.51, "fail"),
        ],
    )
    def test_tier_ladder(self, score, tier):
        assert metrics.mpnn_tier(score, self.THRESHOLDS) == tier

    def test_nan_score_is_unknown_not_fail(self):
        # A NaN score is a FASTA-header parse failure, not a bad sequence.
        assert metrics.mpnn_tier(float("nan"), self.THRESHOLDS) == "unknown"
        assert metrics.mpnn_tier(None, self.THRESHOLDS) == "unknown"

    def test_missing_bound_collapses_tier_rather_than_defaulting(self):
        # No number is invented when a bound is absent from config.
        assert metrics.mpnn_tier(0.5, {"mpnn_score_max": 1.25}) == "acceptable"
        assert metrics.mpnn_tier(0.5, {}) == "fail"

    def test_gate_direction_is_upper_bound(self):
        # The defect this guards: a `>=` test would keep the worst sequences.
        ok, _ = metrics.passes({"mpnn_score": 0.71}, self.THRESHOLDS)
        assert ok
        bad, reasons = metrics.passes({"mpnn_score": 1.40}, self.THRESHOLDS)
        assert not bad
        assert "<= 1.25" in reasons[0]

    def test_gate_ignores_label_only_bounds(self):
        # preferred/marginal bounds must not gate; only mpnn_score_max does.
        ok, _ = metrics.passes({"mpnn_score": 1.20}, self.THRESHOLDS)
        assert ok

    def test_other_gate_blocks_do_not_check_mpnn_score(self):
        ok, _ = metrics.passes({"plddt": 85.0}, {"plddt_min": 80.0})
        assert ok
