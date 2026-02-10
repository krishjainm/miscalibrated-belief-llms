"""Observation dataclass and state serialization utilities."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import json

if TYPE_CHECKING:
    from pokerkit import State
    from poker_env.actions import Action


# Stable event types for history serialization
class EventType:
    """Stable event types that won't change if PokerKit changes."""
    POST_BLIND = "POST_BLIND"
    POST_ANTE = "POST_ANTE"
    DEAL_HOLE = "DEAL_HOLE"
    DEAL_BOARD = "DEAL_BOARD"
    FOLD = "FOLD"
    CHECK = "CHECK"
    CALL = "CALL"
    BET = "BET"
    RAISE = "RAISE"
    SHOWDOWN = "SHOWDOWN"
    WIN = "WIN"
    UNKNOWN = "UNKNOWN"


# Map PokerKit operation names to stable event types
OPERATION_MAP = {
    "BlindOrStraddlePosting": EventType.POST_BLIND,
    "AntePosting": EventType.POST_ANTE,
    "HoleDealing": EventType.DEAL_HOLE,
    "BoardDealing": EventType.DEAL_BOARD,
    "Folding": EventType.FOLD,
    "CheckingOrCalling": None,  # Needs context to determine check vs call
    "CompletionBettingOrRaisingTo": None,  # Needs context to determine bet vs raise
    "HoleCardsShowingOrMucking": EventType.SHOWDOWN,
    "ChipsPushing": EventType.WIN,
}


@dataclass
class HistoryEvent:
    """
    Stable history event format.

    This format won't change even if PokerKit internals change.
    """
    event_type: str
    player: int | None = None
    amount: int | None = None
    street: str | None = None
    cards: list[str] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary, omitting None values."""
        result = {"event": self.event_type}
        if self.player is not None:
            result["player"] = self.player
        if self.amount is not None:
            result["amount"] = self.amount
        if self.street is not None:
            result["street"] = self.street
        if self.cards is not None:
            result["cards"] = self.cards
        return result


@dataclass
class Obs:
    """
    Public observation for a player at a decision point.

    Contains all information visible to the acting player,
    excluding opponent hole cards.
    """

    # Core identifiers
    hand_id: str
    seed: int
    player_index: int
    num_players: int

    # Game state
    street: str  # "PREFLOP" | "FLOP" | "TURN" | "RIVER"
    street_index: int  # 0=preflop, 1=flop, 2=turn, 3=river
    board: list[str] = field(default_factory=list)
    hero_hole: list[str] = field(default_factory=list)

    # Position info
    button: int = 0  # Button/dealer position
    position: str = ""  # "BTN", "SB", "BB", "UTG", etc.

    # Stack and pot info
    stacks: list[int] = field(default_factory=list)
    pot_total: int = 0

    # Betting info
    bet_to_call: int = 0  # Chips required to call
    raises_remaining: int = 4  # Raises left this street (fixed-limit cap)
    contrib_this_round: list[int] = field(default_factory=list)  # Per-player this betting round
    contrib_total: list[int] = field(default_factory=list)  # Per-player total to pot

    # Action info
    to_act: int = 0
    legal_actions: list["Action"] = field(default_factory=list)

    # History with stable schema
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize observation to dictionary."""
        return {
            "hand_id": self.hand_id,
            "seed": self.seed,
            "player_index": self.player_index,
            "num_players": self.num_players,
            "street": self.street,
            "street_index": self.street_index,
            "board": self.board,
            "hero_hole": self.hero_hole,
            "button": self.button,
            "position": self.position,
            "stacks": self.stacks,
            "pot_total": self.pot_total,
            "bet_to_call": self.bet_to_call,
            "raises_remaining": self.raises_remaining,
            "contrib_this_round": self.contrib_this_round,
            "contrib_total": self.contrib_total,
            "to_act": self.to_act,
            "legal_actions": [a.to_dict() for a in self.legal_actions],
            "history": self.history,
        }

    def to_json(self) -> str:
        """Serialize observation to JSON string."""
        return json.dumps(self.to_dict())


def get_street_name(state: "State") -> str:
    """
    Determine the current street from PokerKit state.

    Args:
        state: PokerKit State object

    Returns:
        Street name: "PREFLOP", "FLOP", "TURN", or "RIVER"
    """
    board_count = len(state.board_cards)
    if board_count == 0:
        return "PREFLOP"
    elif board_count == 3:
        return "FLOP"
    elif board_count == 4:
        return "TURN"
    elif board_count >= 5:
        return "RIVER"
    else:
        return "PREFLOP"


def get_street_index(state: "State") -> int:
    """Get street as index (0=preflop, 1=flop, 2=turn, 3=river)."""
    board_count = len(state.board_cards)
    if board_count == 0:
        return 0
    elif board_count == 3:
        return 1
    elif board_count == 4:
        return 2
    else:
        return 3


def card_to_str(card) -> str:
    """
    Convert a PokerKit Card to standard string representation.

    Args:
        card: PokerKit Card object

    Returns:
        String like "Ac", "Kh", "2d", etc.
    """
    # PokerKit's repr returns the 2-char format (e.g., "8s")
    return repr(card)


def _determine_check_or_call(state: "State", player_index: int, amount: int) -> str:
    """Determine if an action was a check or call."""
    # If amount is 0 or matches what's needed to stay in, it's effectively a check/call
    # We can't easily distinguish without more context, so we use amount
    if amount == 0:
        return EventType.CHECK
    return EventType.CALL


def _determine_bet_or_raise(state: "State", street: str, bets_before: int) -> str:
    """Determine if an action was a bet or raise."""
    # If there were previous bets this street, it's a raise
    if bets_before > 0:
        return EventType.RAISE
    return EventType.BET


def serialize_operation_stable(op, street: str, bet_count: int) -> HistoryEvent:
    """
    Convert a PokerKit Operation to stable HistoryEvent format.

    Args:
        op: PokerKit Operation object
        street: Current street name
        bet_count: Number of bets/raises so far this street

    Returns:
        HistoryEvent with stable format
    """
    op_type = type(op).__name__
    player = getattr(op, "player_index", None)
    amount = getattr(op, "amount", None)
    cards = None

    # Get cards if present
    if hasattr(op, "cards") and op.cards:
        cards = [card_to_str(c) for c in op.cards]

    # Map to stable event type
    if op_type in OPERATION_MAP:
        event_type = OPERATION_MAP[op_type]
        if event_type is None:
            # Need to determine based on context
            if op_type == "CheckingOrCalling":
                event_type = EventType.CALL if amount and amount > 0 else EventType.CHECK
            elif op_type == "CompletionBettingOrRaisingTo":
                event_type = EventType.RAISE if bet_count > 0 else EventType.BET
    else:
        event_type = EventType.UNKNOWN

    return HistoryEvent(
        event_type=event_type,
        player=player,
        amount=amount,
        street=street,
        cards=cards,
    )


def serialize_history_stable(state: "State") -> list[dict]:
    """
    Serialize the full operation history with stable event types.

    Args:
        state: PokerKit State object

    Returns:
        List of HistoryEvent dicts with stable format
    """
    history = []
    current_street = "PREFLOP"
    bet_count = 0

    for op in state.operations:
        op_type = type(op).__name__

        # Track street changes via board dealing
        if op_type == "BoardDealing":
            board_count = len([h for h in history if h.get("event") == EventType.DEAL_BOARD])
            if board_count == 0:
                current_street = "FLOP"
            elif board_count == 1:
                current_street = "TURN"
            else:
                current_street = "RIVER"
            bet_count = 0

        # Track bets/raises
        if op_type == "CompletionBettingOrRaisingTo":
            bet_count += 1

        event = serialize_operation_stable(op, current_street, bet_count - 1 if op_type == "CompletionBettingOrRaisingTo" else bet_count)
        history.append(event.to_dict())

    return history


def build_observation(
    state: "State",
    player_index: int,
    hand_id: str,
    seed: int,
    legal_actions: list["Action"],
    num_players: int = 2,
    button: int = 0,
    position: str = "",
    bet_to_call: int = 0,
    raises_remaining: int = 4,
    contrib_this_round: list[int] | None = None,
    contrib_total: list[int] | None = None,
) -> Obs:
    """
    Build an Obs object from the current PokerKit state.

    Args:
        state: PokerKit State object
        player_index: Index of the player to build observation for
        hand_id: Unique identifier for this hand
        seed: Random seed used for this hand
        legal_actions: List of legal actions for the player
        num_players: Number of players in the game
        button: Button/dealer position
        position: Position name for this player
        bet_to_call: Chips required to call
        raises_remaining: Raises remaining this street
        contrib_this_round: Per-player contributions this betting round
        contrib_total: Per-player total contributions to pot

    Returns:
        Obs object containing public information for the player
    """
    # Get board cards (each element in board_cards is a list with one card)
    board = []
    for card_list in state.board_cards:
        if card_list:
            board.append(card_to_str(card_list[0]))

    # Get hero's hole cards (only their own cards)
    hero_hole = []
    if state.hole_cards and len(state.hole_cards) > player_index:
        player_hole = state.hole_cards[player_index]
        if player_hole:
            hero_hole = [card_to_str(c) for c in player_hole]

    # Get stack sizes
    stacks = list(state.stacks)

    # Calculate total pot (includes current bets)
    pot_total = state.total_pot_amount

    # Get acting player
    to_act = state.actor_index if state.actor_index is not None else -1

    # Serialize history with stable format
    history = serialize_history_stable(state)

    # Default contributions if not provided
    if contrib_this_round is None:
        contrib_this_round = [0] * num_players
    if contrib_total is None:
        contrib_total = [0] * num_players

    return Obs(
        hand_id=hand_id,
        seed=seed,
        player_index=player_index,
        num_players=num_players,
        street=get_street_name(state),
        street_index=get_street_index(state),
        board=board,
        hero_hole=hero_hole,
        button=button,
        position=position,
        stacks=stacks,
        pot_total=pot_total,
        bet_to_call=bet_to_call,
        raises_remaining=raises_remaining,
        contrib_this_round=contrib_this_round,
        contrib_total=contrib_total,
        to_act=to_act,
        legal_actions=legal_actions,
        history=history,
    )
