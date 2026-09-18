"""The operator can shorten an episode's SIM budget (HAR-1).

A tier's episode timeout sizes its worst case. When the expert finishes its
work well before that, the rest of the budget is spent stepping a scene
nobody is acting on: the measured S1 store run stopped commanding the base at
147 sim s and then stepped a still scene for another 453. Shortening the
budget is an operator choice rather than a default, because it changes what a
`timeout` verdict means.
"""

from __future__ import annotations

import pytest

from aisle.harness.rollout import resolve_budgets, tier_budgets

pytestmark = pytest.mark.unit


def test_tier_budget_is_unchanged_without_an_override():
    """HAR-1: the tier still decides by default, on every tier."""
    for tier in ("T0", "T2", "S1", "S3"):
        assert resolve_budgets(tier, "oracle") == tier_budgets(tier)


def test_override_shortens_the_sim_budget_and_its_wall_clamp():
    """HAR-1/ADR-23: a shorter sim budget must drag the wall clamp with it, or
    the run keeps the tier's clamp and the saving is only theoretical. The
    wall clamp stays sized by the same sim-to-wall factor the tiers use."""
    tier_sim, tier_wall = tier_budgets("S1")
    sim, wall = resolve_budgets("S1", "oracle", episode_timeout_override_s=180)
    assert sim == 180 < tier_sim
    assert wall < tier_wall
    # still enough wall time for the shortened episode at the slowest engine
    assert wall >= 180 * 2


def test_wall_override_still_wins_over_the_derived_clamp():
    """ADR-38 amendment: the lockstep VLA eval sets the wall clamp directly
    because model latency freezes sim time; that override outranks the one
    derived from a shortened sim budget."""
    sim, wall = resolve_budgets(
        "S1", "oracle", per_episode_wall_override_s=999, episode_timeout_override_s=180
    )
    assert (sim, wall) == (180, 999)
