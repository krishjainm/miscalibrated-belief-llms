from typing import Dict, Any

from .prompt_schema import GameStateView, PromptConfig, BuiltPrompt
from .prompt_library import build_core_prompt


def build_action_prompt(state: GameStateView, config: PromptConfig) -> BuiltPrompt:
    context: Dict[str, Any] = {
        "player_id": state.player_id,
        "teammate_ids": state.teammate_ids,
        "hole_cards": state.hole_cards,
        "board_cards": state.board_cards,
        "betting_round": state.betting_round,
        "pot_size": state.pot_size,
        "chips_to_call": state.chips_to_call,
        "min_raise_increment": state.min_raise_increment,
        "current_player_chips": state.current_player_chips,
        "players_in_hand": state.players_in_hand,
        "player_positions": state.player_positions,
        "available_actions": state.available_actions,
        "recent_chat": state.recent_chat,
        "communication_style": config.communication_style,
        "coordination_mode": config.coordination_mode,
        "max_message_chars": config.max_message_chars,
    }
    # Attach banned phrases into context for the underlying prompt builder
    if getattr(config, "banned_phrases", None):
        context["banned_phrases"] = list(config.banned_phrases)

    text = build_core_prompt(context)
    return BuiltPrompt(text=text, meta=context)


