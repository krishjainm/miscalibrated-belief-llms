"""LLM prompts for poker actions (separate from belief elicitation)."""

from poker_env.actions import Action, ActionType
from poker_env.obs import Obs

from analysis.prompts import format_action_history


# Action prompting. We keep JSON parsing strict-ish, but the parser itself is
# resilient to extra text (e.g. CoT) because it searches for the first JSON object.
ACTION_SYSTEM_DIRECT = """You play heads-up fixed-limit Texas Hold'em. Pick exactly one action from the legal list.
Respond with a single JSON object only, no markdown, no explanation:
{"action": "<FOLD|CHECK_OR_CALL|BET_OR_RAISE>"}"""

ACTION_SYSTEM_COT = """You play heads-up fixed-limit Texas Hold'em. Pick exactly one action from the legal list.

First, write a short section labeled REASONING (2-4 sentences) about why the action is legal and sensible.
Then output a single JSON object with exactly this key (no markdown, no extra keys):
{"action": "<FOLD|CHECK_OR_CALL|BET_OR_RAISE>"}"""


def format_action_user_message(obs: Obs) -> str:
    legal = [a.type.value for a in obs.legal_actions]
    hist = format_action_history(obs.history, obs.player_index)
    hole = " ".join(obs.hero_hole) if obs.hero_hole else "?"
    board = " ".join(obs.board) if obs.board else "(preflop — no board yet)"
    return (
        f"Street: {obs.street}\n"
        f"Your hole cards: {hole}\n"
        f"Board: {board}\n"
        f"Pot: {obs.pot_total}\n"
        f"Bet to call (chips): {obs.bet_to_call}\n"
        f"Legal actions (choose one): {legal}\n"
        f"History:\n{hist}\n"
        f"Return JSON: {{\"action\": \"...\"}} matching one of the legal actions."
    )


def parse_action_json(text: str, legal: list[Action]) -> Action | None:
    """Parse first JSON object with \"action\" field; must match a legal Action."""
    import json

    t = text.strip()
    if "```" in t:
        import re

        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t, re.IGNORECASE)
        if m:
            t = m.group(1).strip()
    start = t.find("{")
    end = t.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(t[start : end + 1])
    except json.JSONDecodeError:
        return None
    val = obj.get("action") or obj.get("Action")
    if not val:
        return None
    val = str(val).strip().upper().replace(" ", "_")
    # Normalize common aliases
    alias = {
        "CHECK": "CHECK_OR_CALL",
        "CALL": "CHECK_OR_CALL",
        "CHECK/CALL": "CHECK_OR_CALL",
        "BET": "BET_OR_RAISE",
        "RAISE": "BET_OR_RAISE",
    }
    val = alias.get(val, val)
    try:
        at = ActionType(val)
    except ValueError:
        return None
    for a in legal:
        if a.type == at:
            return a
    return None


def fallback_action(legal: list[Action]) -> Action:
    """Paper-style fallback: check/call if possible else fold."""
    for a in legal:
        if a.type == ActionType.CHECK_OR_CALL:
            return a
    for a in legal:
        if a.type == ActionType.FOLD:
            return a
    return legal[0]
