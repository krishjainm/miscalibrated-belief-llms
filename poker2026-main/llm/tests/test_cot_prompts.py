"""Sanity checks for CoT vs direct prompts."""

from analysis.cot_prompts import format_belief_prompt_cot


def test_direct_has_json_only_language():
    p = format_belief_prompt_cot(
        hero_hole=["As", "Ah"],
        board=[],
        pot=4,
        street="PREFLOP",
        history=[],
        mode="direct",
        hero_index=0,
    )
    assert "JSON" in p
    assert "REASONING" not in p


def test_cot_has_sections():
    p = format_belief_prompt_cot(
        hero_hole=["As", "Ah"],
        board=[],
        pot=4,
        street="PREFLOP",
        history=[],
        mode="cot",
        hero_index=0,
    )
    assert "REASONING" in p
    assert "PROBABILITIES" in p
