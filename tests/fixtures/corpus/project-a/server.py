"""
project-a: Coherence and Entrainment Analysis Server

This module implements the core analysis pipeline for detecting entrainment
signatures in interaction data. Entrainment — the synchronisation of rhythms
across loosely coupled agents — is the primary mechanism under investigation.

The server exposes a set of tools for:
- Ingesting time-series interaction sequences
- Computing cross-correlation and phase-locking metrics
- Detecting entrainment events above configurable thresholds
- Returning annotated coherence assessments
"""

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class EntrainmentEvent:
    """A detected moment of rhythmic synchronisation between two agents."""
    agent_a: str
    agent_b: str
    onset_index: int
    duration: int
    strength: float  # 0.0–1.0, where 1.0 is perfect phase lock
    resonance_score: float
    notes: str = ""


@dataclass
class CoherenceReport:
    """Aggregate coherence assessment for a sequence of interactions."""
    sequence_id: str
    total_length: int
    entrainment_events: list[EntrainmentEvent] = field(default_factory=list)
    mean_coherence: float = 0.0
    peak_coherence: float = 0.0
    coherence_stability: float = 0.0  # variance-based measure
    interpretation: str = ""


def compute_cross_correlation(seq_a: list[float], seq_b: list[float]) -> list[float]:
    """
    Compute normalised cross-correlation between two sequences.

    Cross-correlation measures the degree to which oscillatory patterns
    in seq_a are reflected in seq_b at varying time lags. High correlation
    at lag=0 suggests synchrony; high correlation at non-zero lag suggests
    phase offset (still entrainment, but shifted).
    """
    n = min(len(seq_a), len(seq_b))
    mean_a = sum(seq_a[:n]) / n
    mean_b = sum(seq_b[:n]) / n
    centred_a = [x - mean_a for x in seq_a[:n]]
    centred_b = [x - mean_b for x in seq_b[:n]]

    # Compute normalisation factor
    norm = math.sqrt(
        sum(x**2 for x in centred_a) * sum(x**2 for x in centred_b)
    )
    if norm == 0:
        return [0.0] * (2 * n - 1)

    result = []
    for lag in range(-(n - 1), n):
        total = 0.0
        for i in range(n):
            j = i + lag
            if 0 <= j < n:
                total += centred_a[i] * centred_b[j]
        result.append(total / norm)

    return result


def detect_entrainment_events(
    correlation: list[float],
    threshold: float = 0.7,
    min_duration: int = 3,
) -> list[tuple[int, int, float]]:
    """
    Identify contiguous regions of high cross-correlation.

    Returns a list of (onset, duration, mean_strength) tuples representing
    detected entrainment windows. Only windows meeting the minimum duration
    criterion are returned — transient correlations are noise, not coherence.

    The threshold and min_duration parameters are configurable to allow
    sensitivity adjustment based on the nature of the interaction data.
    """
    events = []
    in_event = False
    onset = 0
    window_values = []

    for i, val in enumerate(correlation):
        if val >= threshold:
            if not in_event:
                in_event = True
                onset = i
                window_values = []
            window_values.append(val)
        else:
            if in_event:
                if len(window_values) >= min_duration:
                    strength = sum(window_values) / len(window_values)
                    events.append((onset, len(window_values), strength))
                in_event = False
                window_values = []

    # Handle event still open at end of sequence
    if in_event and len(window_values) >= min_duration:
        strength = sum(window_values) / len(window_values)
        events.append((onset, len(window_values), strength))

    return events


def assess_coherence(
    sequence_id: str,
    agent_sequences: dict[str, list[float]],
    entrainment_threshold: float = 0.7,
    min_event_duration: int = 3,
) -> CoherenceReport:
    """
    Produce a full coherence report for a set of agent interaction sequences.

    Computes pairwise entrainment between all agent pairs, aggregates into
    a coherence score, and returns an annotated report. The interpretation
    field is populated with a human-readable summary of the coherence state.
    """
    report = CoherenceReport(
        sequence_id=sequence_id,
        total_length=max((len(s) for s in agent_sequences.values()), default=0),
    )

    agents = list(agent_sequences.keys())
    all_strengths = []

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            a, b = agents[i], agents[j]
            correlation = compute_cross_correlation(
                agent_sequences[a], agent_sequences[b]
            )
            raw_events = detect_entrainment_events(
                correlation, entrainment_threshold, min_event_duration
            )
            for onset, duration, strength in raw_events:
                event = EntrainmentEvent(
                    agent_a=a,
                    agent_b=b,
                    onset_index=onset,
                    duration=duration,
                    strength=strength,
                    resonance_score=strength * math.log1p(duration),
                )
                report.entrainment_events.append(event)
                all_strengths.append(strength)

    if all_strengths:
        report.mean_coherence = sum(all_strengths) / len(all_strengths)
        report.peak_coherence = max(all_strengths)
        mean = report.mean_coherence
        variance = sum((s - mean) ** 2 for s in all_strengths) / len(all_strengths)
        report.coherence_stability = 1.0 / (1.0 + variance)

    report.interpretation = _interpret_coherence(report)
    return report


def _interpret_coherence(report: CoherenceReport) -> str:
    if not report.entrainment_events:
        return "No entrainment detected. System exhibits low coherence."
    if report.mean_coherence >= 0.85:
        return "Strong, sustained entrainment. System is operating in high coherence."
    if report.mean_coherence >= 0.65:
        return "Moderate entrainment present. Coherence is established but not fully stable."
    return "Weak or intermittent entrainment. Coherence is fragile and may not persist."
