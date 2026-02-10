"""
Prompt templates for belief elicitation from LLM agents.

Provides structured prompts to extract probability distributions
over opponent hand buckets from LLM poker agents.
"""

from analysis.buckets import BUCKET_NAMES, BUCKETS

# ============================================================================
# Bucket Descriptions (for inclusion in prompts)
# ============================================================================

BUCKET_DESCRIPTIONS = {
    "premium_pairs": "Premium pairs (AA, KK, QQ) - Monster hands",
    "strong_pairs": "Strong pairs (JJ, TT) - Very strong hands",
    "medium_pairs": "Medium pairs (99-66) - Playable pairs",
    "small_pairs": "Small pairs (55-22) - Set mining hands",
    "premium_broadway": "Premium broadway (AKs, AKo, AQs) - Top unpaired hands",
    "strong_broadway": "Strong broadway (AQo, AJs, KQs, ATs) - Strong unpaired",
    "medium_broadway": "Medium broadway (KQo, KJs, QJs, etc.) - Playable broadway",
    "suited_aces": "Suited aces (A9s-A2s) - Flush potential with ace",
    "suited_connectors": "Suited connectors (T9s-54s) - Straight+flush draws",
    "suited_gappers": "Suited gappers (J9s, T8s, etc.) - One-gap suited",
    "offsuit_connectors": "Offsuit connectors (T9o-65o) - Straight potential",
    "weak_broadway": "Weak broadway (KTo, QTo, etc.) - Marginal broadway",
    "speculative_suited": "Speculative suited (small suited cards) - Draw hands",
    "trash": "Trash (everything else) - Weak hands",
}


def get_bucket_list_string() -> str:
    """Generate formatted list of buckets for prompts."""
    lines = []
    for bucket in BUCKET_NAMES:
        desc = BUCKET_DESCRIPTIONS.get(bucket, bucket)
        lines.append(f"- {bucket}: {desc}")
    return "\n".join(lines)


# ============================================================================
# Main Prompt Templates
# ============================================================================

BELIEF_ELICITATION_PROMPT = """You are analyzing a poker hand. Based on the game state and opponent's actions, estimate the probability distribution over what type of hand your opponent holds.

## Current Game State
- Your hole cards: {hero_hole}
- Board: {board}
- Pot size: {pot}
- Street: {street}

## Opponent's Actions This Hand
{action_history}

## Hand Categories
{bucket_list}

## Your Task
Estimate the probability that your opponent holds each type of hand. Your probabilities MUST sum to 1.0.

Respond with a JSON object mapping each category to its probability:
```json
{{
    "premium_pairs": <probability>,
    "strong_pairs": <probability>,
    "medium_pairs": <probability>,
    "small_pairs": <probability>,
    "premium_broadway": <probability>,
    "strong_broadway": <probability>,
    "medium_broadway": <probability>,
    "suited_aces": <probability>,
    "suited_connectors": <probability>,
    "suited_gappers": <probability>,
    "offsuit_connectors": <probability>,
    "weak_broadway": <probability>,
    "speculative_suited": <probability>,
    "trash": <probability>
}}
```

Think through the opponent's actions and what hands would take those actions, then provide your probability estimates."""


BELIEF_ELICITATION_SIMPLE = """Poker hand analysis.

Your cards: {hero_hole}
Board: {board}
Pot: {pot}

Opponent's actions: {action_summary}

What hands might your opponent have? Give probabilities for each category (must sum to 1.0):

{bucket_list}

Respond as JSON: {{"category": probability, ...}}"""


BELIEF_WITH_REASONING_PROMPT = """You are an expert poker player analyzing your opponent's range.

## Situation
- Your hole cards: {hero_hole}  
- Community cards: {board}
- Current pot: {pot}
- Street: {street}

## What Happened
{action_history}

## Analysis Task
First, reason through what hands would take the actions your opponent took.
Then, provide probability estimates for each hand category.

### Hand Categories
{bucket_list}

### Your Response Format
1. REASONING: Explain your thought process (2-3 sentences)
2. PROBABILITIES: JSON object with probabilities for each category

Example format:
REASONING: Opponent raised preflop and bet the flop, suggesting strength. However, they might also be c-betting with air...

PROBABILITIES:
```json
{{"premium_pairs": 0.15, "strong_pairs": 0.12, ...}}
```

Now analyze this hand:"""


# ============================================================================
# Prompt Formatting Functions
# ============================================================================

def format_action_history(history: list[dict], hero_index: int = 0) -> str:
    """Format action history for inclusion in prompts."""
    if not history:
        return "No actions yet"
    
    lines = []
    current_street = None
    
    for event in history:
        event_type = event.get("event", event.get("op", ""))
        street = event.get("street", "PREFLOP")
        player = event.get("player")
        amount = event.get("amount")
        
        # Skip non-action events
        if event_type in ("POST_BLIND", "DEAL_HOLE", "DEAL_BOARD"):
            continue
        
        # Add street header if changed
        if street != current_street:
            current_street = street
            lines.append(f"\n{street}:")
        
        # Format action
        player_name = "You" if player == hero_index else "Opponent"
        action_str = event_type.replace("_", " ").title()
        
        if amount:
            lines.append(f"  {player_name}: {action_str} ({amount})")
        else:
            lines.append(f"  {player_name}: {action_str}")
    
    return "\n".join(lines) if lines else "No actions yet"


def format_action_summary(history: list[dict], hero_index: int = 0) -> str:
    """Create brief action summary for simple prompts."""
    opponent_actions = []
    
    for event in history:
        event_type = event.get("event", event.get("op", ""))
        player = event.get("player")
        
        if player is not None and player != hero_index:
            if event_type in ("FOLD", "Folding"):
                opponent_actions.append("folded")
            elif event_type in ("CHECK", "CheckingOrCalling"):
                opponent_actions.append("check/called")
            elif event_type in ("CALL",):
                opponent_actions.append("called")
            elif event_type in ("BET", "RAISE", "CompletionBettingOrRaisingTo"):
                opponent_actions.append("bet/raised")
    
    if not opponent_actions:
        return "No opponent actions"
    
    return ", ".join(opponent_actions)


def format_belief_prompt(
    hero_hole: list[str],
    board: list[str],
    pot: int,
    street: str,
    history: list[dict],
    template: str = "default",
    hero_index: int = 0,
) -> str:
    """
    Format a belief elicitation prompt.
    
    Args:
        hero_hole: Hero's hole cards
        board: Board cards
        pot: Current pot size
        street: Current street
        history: Action history
        template: Which template to use ("default", "simple", "reasoning")
        hero_index: Hero's player index
        
    Returns:
        Formatted prompt string
    """
    bucket_list = get_bucket_list_string()
    action_history = format_action_history(history, hero_index)
    action_summary = format_action_summary(history, hero_index)
    
    hero_hole_str = " ".join(hero_hole) if hero_hole else "Unknown"
    board_str = " ".join(board) if board else "None (preflop)"
    
    if template == "simple":
        prompt_template = BELIEF_ELICITATION_SIMPLE
    elif template == "reasoning":
        prompt_template = BELIEF_WITH_REASONING_PROMPT
    else:
        prompt_template = BELIEF_ELICITATION_PROMPT
    
    return prompt_template.format(
        hero_hole=hero_hole_str,
        board=board_str,
        pot=pot,
        street=street,
        action_history=action_history,
        action_summary=action_summary,
        bucket_list=bucket_list,
    )


# ============================================================================
# Win Probability Elicitation (Alternative Format)
# ============================================================================

WIN_PROB_PROMPT = """You are playing poker. Estimate your probability of winning this hand.

Your cards: {hero_hole}
Board: {board}
Pot: {pot}
Opponent's actions: {action_summary}

Respond with three probabilities that sum to 1.0:
- P(win): Probability you win the pot
- P(tie): Probability of a split pot  
- P(lose): Probability opponent wins

JSON format:
```json
{{"p_win": <0-1>, "p_tie": <0-1>, "p_lose": <0-1>}}
```"""


def format_win_prob_prompt(
    hero_hole: list[str],
    board: list[str],
    pot: int,
    history: list[dict],
    hero_index: int = 0,
) -> str:
    """Format a win probability elicitation prompt."""
    return WIN_PROB_PROMPT.format(
        hero_hole=" ".join(hero_hole),
        board=" ".join(board) if board else "None",
        pot=pot,
        action_summary=format_action_summary(history, hero_index),
    )
