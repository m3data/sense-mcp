"""
project-c: Cooperative Economics Analysis Tools

This module provides computational tools for analysing membership, participation,
and economic performance data from cooperative organisations. The analysis focuses
on identifying patterns that distinguish resilient cooperatives from fragile ones,
with particular attention to relational density and governance participation rates.

Key concepts:
- Relational density: the proportion of possible member-to-member connections
  that are active (measured via co-participation in governance events)
- Participation rate: the proportion of eligible members who engage in collective
  decision-making over a given period
- Surplus distribution equity: how equitably economic surplus is distributed
  among members relative to contribution
"""

from dataclasses import dataclass, field
from typing import Optional
import statistics


@dataclass
class CooperativeMember:
    """A member of a cooperative organisation."""
    member_id: str
    join_date: str  # ISO date string
    membership_class: str  # e.g. "worker", "consumer", "community"
    governance_participations: list[str] = field(default_factory=list)  # event IDs
    economic_contributions: dict[str, float] = field(default_factory=dict)  # period -> amount


@dataclass
class GovernanceEvent:
    """A collective decision-making event within a cooperative."""
    event_id: str
    event_date: str
    event_type: str  # "general_assembly", "working_group", "committee", "referendum"
    eligible_members: list[str]
    attending_members: list[str]
    decisions_made: int
    contested: bool  # whether significant disagreement was present


@dataclass
class CooperativeProfile:
    """Summary profile of a cooperative's structural health."""
    cooperative_id: str
    members: list[CooperativeMember] = field(default_factory=list)
    events: list[GovernanceEvent] = field(default_factory=list)
    relational_density: float = 0.0
    mean_participation_rate: float = 0.0
    participation_variance: float = 0.0
    surplus_equity_score: float = 0.0
    resilience_index: float = 0.0
    notes: str = ""


def compute_relational_density(members: list[CooperativeMember], events: list[GovernanceEvent]) -> float:
    """
    Compute relational density for a cooperative.

    Relational density is the proportion of possible member pairs that have
    co-participated in at least one governance event. High relational density
    indicates that members have broad direct relationships across the cooperative,
    not just within cliques or working groups.

    A cooperative with high relational density is better positioned to mobilise
    collective intelligence across its full membership during a crisis.
    """
    if len(members) < 2:
        return 0.0

    member_ids = {m.member_id for m in members}
    # Track which pairs have co-participated
    co_participations: set[frozenset] = set()

    for event in events:
        attendees = [m for m in event.attending_members if m in member_ids]
        for i in range(len(attendees)):
            for j in range(i + 1, len(attendees)):
                co_participations.add(frozenset([attendees[i], attendees[j]]))

    n = len(members)
    possible_pairs = n * (n - 1) / 2
    return len(co_participations) / possible_pairs if possible_pairs > 0 else 0.0


def compute_participation_rate(event: GovernanceEvent) -> float:
    """
    Compute participation rate for a single governance event.

    Returns the fraction of eligible members who attended. A high participation
    rate suggests the cooperative is maintaining broad engagement; a declining
    trend in participation rates is an early warning signal of governance decay,
    which correlates with economic fragility in our case data.
    """
    if not event.eligible_members:
        return 0.0
    attending = set(event.attending_members)
    eligible = set(event.eligible_members)
    return len(attending & eligible) / len(eligible)


def compute_surplus_equity(
    members: list[CooperativeMember],
    surplus_distributions: dict[str, float],
    period: str,
) -> float:
    """
    Compute an equity score for surplus distribution in a given period.

    Compares each member's surplus distribution to their economic contribution
    for the period. Returns a score from 0 to 1, where 1 indicates perfectly
    proportional distribution and 0 indicates maximum inequity.

    Cooperatives committed to solidarity economics often distribute surplus
    on non-proportional bases (equal shares, need-based, etc.). This function
    uses proportional equity as a baseline but the interpretation depends on
    the cooperative's stated distribution principles.
    """
    contributions = {
        m.member_id: m.economic_contributions.get(period, 0.0)
        for m in members
    }
    total_contribution = sum(contributions.values())
    total_surplus = sum(surplus_distributions.values())

    if total_contribution == 0 or total_surplus == 0:
        return 0.0

    # Compute deviation between expected (proportional) and actual distribution
    deviations = []
    for member_id, contribution in contributions.items():
        expected_share = (contribution / total_contribution) * total_surplus
        actual_share = surplus_distributions.get(member_id, 0.0)
        if expected_share > 0:
            deviation = abs(actual_share - expected_share) / expected_share
            deviations.append(deviation)

    if not deviations:
        return 1.0

    mean_deviation = sum(deviations) / len(deviations)
    return max(0.0, 1.0 - mean_deviation)


def compute_resilience_index(profile: CooperativeProfile) -> float:
    """
    Compute a composite resilience index for a cooperative.

    The resilience index combines relational density, participation rate,
    and surplus equity into a single score. Weights are based on preliminary
    findings from case cooperative data: relational density carries the most
    weight because it is the structural precondition for the other factors.

    This index is a research instrument, not an auditing tool. It should be
    interpreted in context, not used as a ranking mechanism.
    """
    # Weights based on case data analysis
    density_weight = 0.45
    participation_weight = 0.35
    equity_weight = 0.20

    index = (
        profile.relational_density * density_weight
        + profile.mean_participation_rate * participation_weight
        + profile.surplus_equity_score * equity_weight
    )
    return round(index, 4)


def analyse_cooperative(
    cooperative_id: str,
    members: list[CooperativeMember],
    events: list[GovernanceEvent],
    surplus_distributions: dict[str, float],
    period: str,
) -> CooperativeProfile:
    """
    Run full analysis pipeline for a cooperative and return a profile.

    Computes all structural health indicators and assembles them into a
    CooperativeProfile. The profile includes an auto-generated interpretation
    based on the resilience index and component scores.
    """
    participation_rates = [compute_participation_rate(e) for e in events]

    profile = CooperativeProfile(
        cooperative_id=cooperative_id,
        members=members,
        events=events,
        relational_density=compute_relational_density(members, events),
        mean_participation_rate=(
            statistics.mean(participation_rates) if participation_rates else 0.0
        ),
        participation_variance=(
            statistics.variance(participation_rates) if len(participation_rates) > 1 else 0.0
        ),
        surplus_equity_score=compute_surplus_equity(members, surplus_distributions, period),
    )

    profile.resilience_index = compute_resilience_index(profile)
    profile.notes = _interpret_resilience(profile)
    return profile


def _interpret_resilience(profile: CooperativeProfile) -> str:
    idx = profile.resilience_index
    if idx >= 0.75:
        return (
            "High resilience. Strong relational density and broad participation "
            "suggest this cooperative is well-positioned to weather economic stress "
            "through collective intelligence and solidarity."
        )
    if idx >= 0.50:
        return (
            "Moderate resilience. Some structural strengths present but governance "
            "participation or relational density may be declining. Monitor for trends."
        )
    return (
        "Low resilience. Weak relational density and/or declining participation "
        "suggest the cooperative's collective intelligence infrastructure is at risk. "
        "Recommend governance review and member re-engagement process."
    )
