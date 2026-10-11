"""Accumulate per-candidate metrics into the campaign summary.

Produces results/final_leads_summary.csv, plus a pipeline table counting 
survivors at each stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics


@dataclass
class StageCount:
    stage: str
    entered: int
    passed: int
    failed_threshold: int
    failed_missing: int
    reasons: dict[str, int] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.entered if self.entered else float("nan")

    def as_row(self) -> dict:
        return {
            "stage": self.stage,
            "entered": self.entered,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 4) if self.entered else None,
            "failed_threshold": self.failed_threshold,
            "failed_missing": self.failed_missing,
            "top_reason": max(self.reasons, key=self.reasons.get) if self.reasons else None,
        }


class Campaign:
    """Rows keyed by (design_id, sequence_id), merged across stages."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], dict] = {}
        self._stages: list[StageCount] = []

    def add(self, row: dict) -> None:
        key = (str(row.get("design_id", "")), str(row.get("sequence_id", "")))
        self._rows.setdefault(key, {"design_id": key[0], "sequence_id": key[1]}).update(row)

    def add_all(self, rows: list[dict]) -> None:
        for row in rows:
            self.add(row)

    def add_records(self, records: list) -> None:
        """Accept parser record objects (anything with .as_row())."""
        for record in records:
            self.add(record.as_row())
            if getattr(record, "notes", None):
                key = (record.design_id, record.sequence_id)
                existing = self._rows[key].get("notes", "")
                joined = "; ".join(record.notes)
                self._rows[key]["notes"] = f"{existing}; {joined}" if existing else joined

    def gate(
        self,
        stage: str,
        thresholds: dict,
        metric_prefix: str,
        enforce: bool = True,
    ) -> list[tuple[str, str]]:
        """Apply a gate block from config/campaign.yaml.

        Records the verdict on every row as `<stage>_pass` and `<stage>_reasons`
        whether or not `enforce` is set, so a smoke run collects the same
        diagnostics it would enforce later. Returns the surviving keys.

        `metric_prefix` maps engine-prefixed columns (esmfold_plddt) onto the
        bare names metrics.passes expects (plddt).
        """
        survivors: list[tuple[str, str]] = []
        counts = StageCount(stage, entered=len(self._rows), passed=0,
                            failed_threshold=0, failed_missing=0)

        for key, row in self._rows.items():
            candidate = {
                name: row.get(f"{metric_prefix}_{name}")
                for name in (
                    "plddt", "scrmsd", "iptm", "pae_interaction", "binder_rmsd_to_design"
                )
            }
            # binder_rmsd is written with a shorter column name by ComplexRecord.
            if candidate["binder_rmsd_to_design"] is None:
                candidate["binder_rmsd_to_design"] = row.get(f"{metric_prefix}_binder_rmsd")

            ok, reasons = metrics.passes(candidate, thresholds)
            row[f"{stage}_pass"] = ok
            row[f"{stage}_reasons"] = "; ".join(reasons)
            if ok:
                counts.passed += 1
                survivors.append(key)
            else:
                missing = any("MISSING" in r for r in reasons)
                if missing:
                    counts.failed_missing += 1
                else:
                    counts.failed_threshold += 1
                for reason in reasons:
                    label = reason.split("=")[0]
                    counts.reasons[label] = counts.reasons.get(label, 0) + 1

        self._stages.append(counts)
        if enforce:
            self._rows = {k: v for k, v in self._rows.items() if k in set(survivors)}
        return survivors

    def frame(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame()
        df = pd.DataFrame(list(self._rows.values()))
        lead_cols = [c for c in ("design_id", "sequence_id") if c in df.columns]
        return df[lead_cols + sorted(c for c in df.columns if c not in lead_cols)]

    def funnel(self) -> pd.DataFrame:
        return pd.DataFrame([s.as_row() for s in self._stages])

    def write(self, path: str | Path, funnel_path: str | Path | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = self.frame()
        frame.to_csv(path, index=False)
        if funnel_path:
            funnel_path = Path(funnel_path)
            funnel_path.parent.mkdir(parents=True, exist_ok=True)
            self.funnel().to_csv(funnel_path, index=False)
        return path

    def rank(self, by: str = "boltz2_pae_interaction", ascending: bool = True) -> pd.DataFrame:
        """Leads ordered by a chosen metric, NaNs last whichever way we sort."""
        frame = self.frame()
        if frame.empty or by not in frame.columns:
            return frame
        return frame.sort_values(by, ascending=ascending, na_position="last")

    def health(self) -> dict:
        """Did the run produce usable metrics, independent of design quality?

        The check the smoke test exists for. A column that is entirely NaN is a
        parser or wiring failure; it should never be read as every design having
        failed.
        """
        frame = self.frame()
        if frame.empty:
            return {"rows": 0, "verdict": "no rows -- nothing ran or nothing survived"}

        numeric = frame.select_dtypes(include=[np.number])
        all_nan = [c for c in numeric.columns if numeric[c].isna().all()]
        some_nan = {
            c: int(numeric[c].isna().sum())
            for c in numeric.columns
            if 0 < numeric[c].isna().sum() < len(numeric)
        }
        return {
            "rows": len(frame),
            "fully_missing_columns": all_nan,
            "partially_missing_columns": some_nan,
            "verdict": (
                f"BROKEN: {all_nan} never populated -- fix parsers before trusting results"
                if all_nan
                else "ok: every metric column has at least one value"
            ),
        }


def funnel_projection(
    backbones: int, seqs_per_backbone: int, pass_rates: list[tuple[str, float]]
) -> pd.DataFrame:
    """Expected survivors per stage, before spending any GPU time.

    Worth running before a campaign: the MANIFEST's prototype numbers
    (10-15 backbones, 2-3 seqs, ~17% then ~7%) project to well under one lead,
    which is a scale problem rather than a design problem.
    """
    remaining = float(backbones * seqs_per_backbone)
    rows = [{"stage": "sequences", "expected": round(remaining, 2), "pass_rate": None}]
    for name, rate in pass_rates:
        remaining *= rate
        rows.append({"stage": name, "expected": round(remaining, 2), "pass_rate": rate})
    return pd.DataFrame(rows)
