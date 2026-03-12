"""Synthesis longitudinale — retrospective analysis during dream.

Read-only analysis of recent cycles. Outputs a text summary for
PromptBuilder (NOT Thinker observations). Detects trends, anomalies,
and cross-patterns across Luna's recent cognitive history.

All thresholds are phi-derived.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from luna_common.constants import INV_PHI, INV_PHI2, INV_PHI3, COMP_NAMES
from luna_common.schemas.cycle import CycleRecord

log = logging.getLogger(__name__)

# Minimum cycles for meaningful analysis.
_MIN_CYCLES: int = 10

# Moving baseline window for anomaly detection.
_BASELINE_WINDOW: int = 20

# Anomaly threshold: 2-sigma.
_ANOMALY_SIGMA: float = 2.0


# ══════════════════════════════════════════════════════════════════════════════
#  DATA MODEL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrendLine:
    """A detected trend in a metric over time."""

    metric: str
    slope: float  # positive = improving
    r_squared: float  # goodness of fit
    direction: str  # "up", "down", "stable"
    window: int  # cycles analyzed


@dataclass
class Anomaly:
    """A detected anomaly — metric deviated significantly from baseline."""

    metric: str
    cycle_index: int
    value: float
    baseline_mean: float
    baseline_std: float
    sigma_deviation: float


@dataclass
class CrossPattern:
    """A cross-metric correlation or pattern."""

    metric_a: str
    metric_b: str
    correlation: float
    description: str


@dataclass
class SynthesisReport:
    """Complete synthesis report — trends, anomalies, cross-patterns."""

    trends: list[TrendLine] = field(default_factory=list)
    anomalies: list[Anomaly] = field(default_factory=list)
    cross_patterns: list[CrossPattern] = field(default_factory=list)
    cycles_analyzed: int = 0
    summary: str = ""


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHESIS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

# Metric labels for French display.
METRIC_LABELS: dict[str, str] = {
    "psi_0": f"{COMP_NAMES[0]} (Perception)",
    "psi_1": f"{COMP_NAMES[1]} (Reflexion)",
    "psi_2": f"{COMP_NAMES[2]} (Integration)",
    "psi_3": f"{COMP_NAMES[3]} (Expression)",
    "phi_iit": "Phi_IIT",
    "thinker_confidence": "Confiance Thinker",
    "affect_valence": "Valence affective",
    "affect_arousal": "Arousal affectif",
}


class Synthesis:
    """Longitudinal analysis engine — retrospective during dream.

    Reads CycleRecords and extracts statistical patterns.
    Pure read-only: does not modify any state.
    """

    def __init__(self, cycle_store: object | None = None) -> None:
        self._cycle_store = cycle_store

    def run(
        self,
        cycles: list[CycleRecord] | None = None,
        window: int = 50,
    ) -> SynthesisReport:
        """Run synthesis on recent cycles.

        Args:
            cycles: Pre-loaded cycles (overrides cycle_store).
            window: Maximum number of recent cycles to analyze.

        Returns:
            SynthesisReport with all findings.
        """
        if cycles is None and self._cycle_store is not None:
            try:
                cycles = self._cycle_store.read_recent(window)
            except Exception:
                log.debug("Synthesis: could not read cycles", exc_info=True)
                cycles = []

        if not cycles or len(cycles) < _MIN_CYCLES:
            return SynthesisReport(
                cycles_analyzed=len(cycles) if cycles else 0,
                summary="Insufficient data for synthesis.",
            )

        cycles = cycles[-window:]
        report = SynthesisReport(cycles_analyzed=len(cycles))

        # Extract time series.
        series = self._extract_series(cycles)

        # Trend detection.
        report.trends = self._detect_trends(series)

        # Anomaly detection.
        report.anomalies = self._detect_anomalies(series)

        # Cross-pattern analysis.
        report.cross_patterns = self._detect_cross_patterns(series, cycles)

        # Generate summary.
        report.summary = self._generate_summary(report)

        return report

    def _extract_series(self, cycles: list[CycleRecord]) -> dict[str, list[float]]:
        """Extract named time series from CycleRecords."""
        series: dict[str, list[float]] = {
            "psi_0": [], "psi_1": [], "psi_2": [], "psi_3": [],
            "phi_iit": [],
            "thinker_confidence": [],
            "affect_valence": [], "affect_arousal": [],
        }

        for c in cycles:
            psi = c.psi_after
            series["psi_0"].append(psi[0])
            series["psi_1"].append(psi[1])
            series["psi_2"].append(psi[2])
            series["psi_3"].append(psi[3])
            series["phi_iit"].append(c.phi_iit_after)
            series["thinker_confidence"].append(c.thinker_confidence)

            # Affect from trace if available.
            if c.affect_trace:
                series["affect_valence"].append(
                    c.affect_trace.get("valence_after", 0.0)
                )
                series["affect_arousal"].append(
                    c.affect_trace.get("arousal_after", 0.0)
                )
            else:
                series["affect_valence"].append(0.0)
                series["affect_arousal"].append(0.0)

        return series

    def _detect_trends(self, series: dict[str, list[float]]) -> list[TrendLine]:
        """Detect linear trends using np.polyfit."""
        trends: list[TrendLine] = []

        for metric, values in series.items():
            if len(values) < _MIN_CYCLES:
                continue

            arr = np.array(values)
            x = np.arange(len(arr))

            try:
                coeffs = np.polyfit(x, arr, 1)
                slope = float(coeffs[0])

                # R-squared.
                predicted = np.polyval(coeffs, x)
                ss_res = float(np.sum((arr - predicted) ** 2))
                ss_tot = float(np.sum((arr - np.mean(arr)) ** 2))
                r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

                # Direction classification.
                if abs(slope) < 1e-4:
                    direction = "stable"
                elif slope > 0:
                    direction = "up"
                else:
                    direction = "down"

                trends.append(TrendLine(
                    metric=metric,
                    slope=round(slope, 6),
                    r_squared=round(max(0.0, r_squared), 4),
                    direction=direction,
                    window=len(values),
                ))
            except Exception:
                pass

        return trends

    def _detect_anomalies(self, series: dict[str, list[float]]) -> list[Anomaly]:
        """Detect anomalies using moving baseline + 2-sigma threshold."""
        anomalies: list[Anomaly] = []

        for metric, values in series.items():
            if len(values) < _BASELINE_WINDOW:
                continue

            arr = np.array(values)

            for i in range(_BASELINE_WINDOW, len(arr)):
                window = arr[max(0, i - _BASELINE_WINDOW):i]
                mean = float(np.mean(window))
                std = float(np.std(window))

                if std < 1e-8:
                    continue

                deviation = abs(float(arr[i]) - mean) / std
                if deviation >= _ANOMALY_SIGMA:
                    anomalies.append(Anomaly(
                        metric=metric,
                        cycle_index=i,
                        value=round(float(arr[i]), 4),
                        baseline_mean=round(mean, 4),
                        baseline_std=round(std, 4),
                        sigma_deviation=round(deviation, 2),
                    ))

        return anomalies

    def _detect_cross_patterns(
        self,
        series: dict[str, list[float]],
        cycles: list[CycleRecord],
    ) -> list[CrossPattern]:
        """Detect cross-metric correlations and notable patterns."""
        patterns: list[CrossPattern] = []

        # Correlation between phi_iit and thinker_confidence.
        if len(series["phi_iit"]) >= _MIN_CYCLES:
            phi_arr = np.array(series["phi_iit"])
            conf_arr = np.array(series["thinker_confidence"])
            if np.std(phi_arr) > 1e-8 and np.std(conf_arr) > 1e-8:
                corr = float(np.corrcoef(phi_arr, conf_arr)[0, 1])
                if abs(corr) > INV_PHI2:
                    patterns.append(CrossPattern(
                        metric_a="phi_iit",
                        metric_b="thinker_confidence",
                        correlation=round(corr, 3),
                        description=(
                            f"Phi_IIT et confiance Thinker correles ({corr:+.3f})"
                        ),
                    ))

        # LLM failures vs phi_iit (if llm_failed field exists).
        llm_failures = []
        for c in cycles:
            failed = getattr(c, "llm_failed", False)
            llm_failures.append(1.0 if failed else 0.0)

        if sum(llm_failures) > 0 and len(llm_failures) >= _MIN_CYCLES:
            fail_arr = np.array(llm_failures)
            phi_arr = np.array(series["phi_iit"])
            if np.std(fail_arr) > 1e-8 and np.std(phi_arr) > 1e-8:
                corr = float(np.corrcoef(fail_arr, phi_arr)[0, 1])
                if abs(corr) > INV_PHI3:
                    patterns.append(CrossPattern(
                        metric_a="llm_failures",
                        metric_b="phi_iit",
                        correlation=round(corr, 3),
                        description=(
                            f"Echecs LLM et Phi_IIT correles ({corr:+.3f})"
                        ),
                    ))

        # Significant episodes count vs affect arousal.
        if len(series["affect_arousal"]) >= _MIN_CYCLES:
            arousal_arr = np.array(series["affect_arousal"])
            psi2_arr = np.array(series["psi_2"])
            if np.std(arousal_arr) > 1e-8 and np.std(psi2_arr) > 1e-8:
                corr = float(np.corrcoef(arousal_arr, psi2_arr)[0, 1])
                if abs(corr) > INV_PHI2:
                    patterns.append(CrossPattern(
                        metric_a="affect_arousal",
                        metric_b="psi_2",
                        correlation=round(corr, 3),
                        description=(
                            f"Arousal et Integration correles ({corr:+.3f})"
                        ),
                    ))

        return patterns

    def _generate_summary(self, report: SynthesisReport) -> str:
        """Generate a French summary for PromptBuilder injection."""
        lines: list[str] = []
        lines.append(f"Synthese longitudinale ({report.cycles_analyzed} cycles):")

        # Trends.
        significant_trends = [
            t for t in report.trends
            if t.r_squared > INV_PHI3 and t.direction != "stable"
        ]
        if significant_trends:
            lines.append("")
            lines.append("Tendances:")
            for t in significant_trends:
                label = METRIC_LABELS.get(t.metric, t.metric)
                arrow = "^" if t.direction == "up" else "v"
                lines.append(
                    f"  {arrow} {label}: pente={t.slope:+.4f} (R2={t.r_squared:.2f})"
                )

        # Anomalies.
        if report.anomalies:
            lines.append("")
            lines.append(f"Anomalies detectees: {len(report.anomalies)}")
            for a in report.anomalies[:3]:
                label = METRIC_LABELS.get(a.metric, a.metric)
                lines.append(
                    f"  ! {label} cycle {a.cycle_index}: "
                    f"{a.value} (baseline {a.baseline_mean}, {a.sigma_deviation:.1f}sigma)"
                )

        # Cross-patterns.
        if report.cross_patterns:
            lines.append("")
            lines.append("Correlations:")
            for cp in report.cross_patterns:
                lines.append(f"  ~ {cp.description}")

        if not significant_trends and not report.anomalies and not report.cross_patterns:
            lines.append("Aucune tendance significative detectee.")

        return "\n".join(lines)
