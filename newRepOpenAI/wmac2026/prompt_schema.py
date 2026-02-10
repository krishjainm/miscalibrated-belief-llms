from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class GameStateView:
    player_id: int
    teammate_ids: List[int]
    hole_cards: List[str]
    board_cards: List[str]
    betting_round: str
    pot_size: int
    chips_to_call: int
    min_raise_increment: int
    current_player_chips: int
    players_in_hand: List[int]
    player_positions: Dict[int, str]
    available_actions: List[str]
    recent_chat: List[Dict[str, Any]]


@dataclass
class PromptConfig:
    communication_style: str
    coordination_mode: str
    max_message_chars: int = 200
    include_learning_references: bool = True
    include_strategic_questions: bool = True
    banned_phrases: Optional[List[str]] = None


@dataclass
class BuiltPrompt:
    text: str
    meta: Dict[str, Any]


