"""Summary accumulation and funnel accounting."""

import numpy as np
import pytest

from pipeline import parsers, report


@pytest.fixture
def campaign():
    c = report.Campaign()
    c.add_all(
        [
            {"design_id": "d0", "sequence_id": "s1", "esmfold_plddt": 90.0, "esmfold_scrmsd": 1.0},
            {"design_id": "d0", "sequence_id": "s2", "esmfold_plddt": 60.0, "esmfold_scrmsd": 1.0},
            {"design_id": "d1", "sequence_id": "s3", "esmfold_plddt": 85.0, "esmfold_scrmsd": 3.0},
        ]
    )
    return c


class TestMerge:
    def test_rows_merge_on_design_and_sequence(self, campaign):
        # Step 3 and Step 4 write separately for the same candidate; they must
        # land on one row, not three.
        campaign.add({"design_id": "d0", "sequence_id": "s1", "boltz2_iptm": 0.9})
        frame = campaign.frame()
        assert len(frame) == 3
        row = frame[frame.sequence_id == "s1"].iloc[0]
        assert row.esmfold_plddt == 90.0 and row.boltz2_iptm == 0.9

    def test_two_engines_coexist(self, campaign):
        campaign.add_records(
            [
                parsers.ComplexRecord("d0", "s1", "boltz2", iptm=0.9),
                parsers.ComplexRecord("d0", "s1", "colabfold", iptm=0.7),
            ]
        )
        row = campaign.frame().set_index("sequence_id").loc["s1"]
        assert row.boltz2_iptm == 0.9 and row.colabfold_iptm == 0.7

    def test_notes_are_accumulated(self, campaign):
        rec = parsers.ComplexRecord("d0", "s1", "boltz2")
        rec.notes.append("pae missing")
        campaign.add_records([rec])
        assert "pae missing" in campaign.frame().set_index("sequence_id").loc["s1"].notes


class TestGate:
    THRESHOLDS = {"plddt_min": 80.0, "scrmsd_max": 2.0}

    def test_enforced_gate_drops_failures(self, campaign):
        survivors = campaign.gate("gate3a", self.THRESHOLDS, "esmfold")
        assert len(survivors) == 1
        assert campaign.frame().sequence_id.tolist() == ["s1"]

    def test_unenforced_gate_keeps_rows_but_records_verdict(self, campaign):
        campaign.gate("gate3a", self.THRESHOLDS, "esmfold", enforce=False)
        frame = campaign.frame().set_index("sequence_id")
        assert len(frame) == 3
        assert frame.loc["s1"].gate3a_pass
        assert not frame.loc["s2"].gate3a_pass
        assert "plddt" in frame.loc["s2"].gate3a_reasons

    def test_funnel_separates_threshold_from_missing(self):
        # The distinction the smoke test depends on: a bad design and a broken
        # parser must not be counted the same way.
        c = report.Campaign()
        c.add({"design_id": "d0", "sequence_id": "bad", "esmfold_plddt": 10.0,
               "esmfold_scrmsd": 1.0})
        c.add({"design_id": "d0", "sequence_id": "unparsed", "esmfold_scrmsd": 1.0})
        c.gate("gate3a", self.THRESHOLDS, "esmfold", enforce=False)
        row = c.funnel().iloc[0]
        assert row.failed_threshold == 1
        assert row.failed_missing == 1

    def test_funnel_records_pass_rate(self, campaign):
        campaign.gate("gate3a", self.THRESHOLDS, "esmfold")
        # as_row() rounds to 4 dp for readability in the CSV.
        assert campaign.funnel().iloc[0].pass_rate == pytest.approx(1 / 3, abs=1e-4)

    def test_complex_binder_rmsd_column_alias_is_resolved(self):
        # ComplexRecord writes `<engine>_binder_rmsd`, but the threshold key is
        # binder_rmsd_to_design_max. A missed alias would report every candidate
        # as MISSING on an otherwise good metric.
        c = report.Campaign()
        c.add_records([parsers.ComplexRecord("d0", "s1", "boltz2", iptm=0.9,
                                             pae_interaction=5.0,
                                             binder_rmsd_to_design=1.0)])
        c.gate("step4", {"binder_rmsd_to_design_max": 2.0}, "boltz2", enforce=False)
        frame = c.frame().set_index("sequence_id")
        assert frame.loc["s1"].step4_pass
        assert frame.loc["s1"].step4_reasons == ""


class TestHealth:
    def test_flags_a_fully_missing_column(self):
        c = report.Campaign()
        c.add({"design_id": "d0", "sequence_id": "s1", "esmfold_plddt": 90.0,
               "boltz2_iptm": float("nan")})
        health = c.health()
        assert "boltz2_iptm" in health["fully_missing_columns"]
        assert health["verdict"].startswith("BROKEN")

    def test_clean_run_passes(self):
        c = report.Campaign()
        c.add({"design_id": "d0", "sequence_id": "s1", "esmfold_plddt": 90.0})
        assert c.health()["verdict"].startswith("ok")

    def test_empty_campaign_is_reported_not_crashed(self):
        assert report.Campaign().health()["rows"] == 0


class TestRankAndWrite:
    def test_rank_puts_nan_last_when_ascending(self):
        c = report.Campaign()
        c.add({"design_id": "d", "sequence_id": "good", "boltz2_pae_interaction": 5.0})
        c.add({"design_id": "d", "sequence_id": "missing",
               "boltz2_pae_interaction": float("nan")})
        c.add({"design_id": "d", "sequence_id": "bad", "boltz2_pae_interaction": 20.0})
        assert c.rank().sequence_id.tolist() == ["good", "bad", "missing"]

    def test_write_emits_both_files(self, tmp_path, campaign):
        campaign.gate("gate3a", {"plddt_min": 80.0}, "esmfold", enforce=False)
        out = campaign.write(tmp_path / "r/summary.csv", tmp_path / "r/funnel.csv")
        assert out.exists()
        assert (tmp_path / "r/funnel.csv").exists()
        assert "sequence_id" in out.read_text().splitlines()[0]

    def test_identifier_columns_come_first(self, campaign):
        assert campaign.frame().columns[:2].tolist() == ["design_id", "sequence_id"]


class TestFunnelProjection:
    def test_manifest_prototype_scale_projects_under_one_lead(self):
        # Quantifies the MANIFEST's yield problem before any GPU time: 15
        # backbones x 3 seqs through the stated pass rates.
        projection = report.funnel_projection(
            15, 3, [("gate3a", 0.17), ("gate3b", 0.07), ("step4", 0.5)]
        )
        assert projection.iloc[0].expected == 45
        assert projection.iloc[-1].expected < 1.0

    def test_campaign_scale_projects_usable_yield(self):
        projection = report.funnel_projection(
            1000, 8, [("gate3a", 0.17), ("gate3b", 0.07), ("step4", 0.5)]
        )
        assert projection.iloc[-1].expected > 40
