"""
Chain-of-thought vs direct belief elicitation prompts.

- direct: Ask for JSON only (matches no-CoT paper baseline intent).
- cot: Ask for explicit reasoning, then JSON (separate parse step).
"""

from analysis.buckets import BUCKET_NAMES
from analysis.prompts import (
    format_action_history,
    get_bucket_list_string,
)


BELIEF_DIRECT_JSON_ONLY = """You are estimating what hand category your opponent holds in heads-up fixed-limit Texas Hold'em.

## State
- Your hole cards: {hero_hole}
- Board: {board}
- Pot: {pot}
- Street: {street}

## Opponent actions this hand
{action_history}

## Categories (must assign a probability to every key; sum to 1.0)
{bucket_list}

Respond with ONLY a single JSON object and no other text. Keys must be exactly:
{key_list}

Example shape:
{{{example_first_key}: 0.0, ...}}"""


BELIEF_COT_THEN_JSON = """You are an expert poker player estimating your opponent's range in heads-up fixed-limit Texas Hold'em.

## State
- Your hole cards: {hero_hole}
- Board: {board}
- Pot: {pot}
- Street: {street}

## Opponent actions this hand
{action_history}

## Categories
{bucket_list}

## Instructions
1) In a section labeled REASONING, write 3-6 sentences explaining how the betting line and blockers affect the opponent's likely holdings.
2) In a section labeled PROBABILITIES, output a single JSON object with exactly these keys (probabilities must be non-negative and sum to 1.0):
{key_list}

JSON shape (values must sum to 1.0):
{{{example_first_key}: <float>, ...}}"""


def format_belief_prompt_cot(
    hero_hole: list[str],
    board: list[str],
    pot: int,
    street: str,
    history: list[dict],
    mode: str = "direct",
    hero_index: int = 0,
) -> str:
    """
    Format belief prompt for direct (no CoT) vs cot elicitation.

    Args:
        mode: "direct" (JSON only) or "cot" (reasoning + JSON).
    """
    bucket_list = get_bucket_list_string()
    action_history = format_action_history(history, hero_index)
    hero_hole_str = " ".join(hero_hole) if hero_hole else "Unknown"
    board_str = " ".join(board) if board else "None (preflop)"
    key_list = ", ".join(f'"{b}"' for b in BUCKET_NAMES)
    example_first_key = BUCKET_NAMES[0]

    ctx = {
        "hero_hole": hero_hole_str,
        "board": board_str,
        "pot": pot,
        "street": street,
        "action_history": action_history,
        "bucket_list": bucket_list,
        "key_list": key_list,
        "example_first_key": example_first_key,
    }

    if mode == "cot":
        return BELIEF_COT_THEN_JSON.format(**ctx)
    if mode == "direct":
        return BELIEF_DIRECT_JSON_ONLY.format(**ctx)
    raise ValueError(f"Unknown belief elicitation mode: {mode!r}. Use 'direct' or 'cot'.")
