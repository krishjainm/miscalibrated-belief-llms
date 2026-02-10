"""Action types and PokerKit method mapping for Fixed-Limit Hold'em."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pokerkit import State


class ActionType(Enum):
    """Legal action types in Fixed-Limit Texas Hold'em."""

    FOLD = "FOLD"
    CHECK_OR_CALL = "CHECK_OR_CALL"
    BET_OR_RAISE = "BET_OR_RAISE"


@dataclass(frozen=True)
class Action:
    """An action to be taken by a player."""

    type: ActionType

    def __repr__(self) -> str:
        return f"Action({self.type.value})"

    def to_dict(self) -> dict:
        """Serialize action to dictionary."""
        return {"type": self.type.value}

    @classmethod
    def from_dict(cls, data: dict) -> "Action":
        """Deserialize action from dictionary."""
        return cls(type=ActionType(data["type"]))


# Convenience constants
FOLD = Action(ActionType.FOLD)
CHECK_OR_CALL = Action(ActionType.CHECK_OR_CALL)
BET_OR_RAISE = Action(ActionType.BET_OR_RAISE)


def apply_action(state: "State", action: Action) -> None:
    """
    Apply an action to a PokerKit state.

    Maps our Action enum to PokerKit state methods:
    - FOLD -> state.fold()
    - CHECK_OR_CALL -> state.check_or_call()
    - BET_OR_RAISE -> state.complete_bet_or_raise_to()

    Args:
        state: PokerKit State object
        action: Action to apply

    Raises:
        ValueError: If the action is not legal in the current state
    """
    if action.type == ActionType.FOLD:
        if not state.can_fold():
            raise ValueError("Cannot fold in current state")
        state.fold()
    elif action.type == ActionType.CHECK_OR_CALL:
        if not state.can_check_or_call():
            raise ValueError("Cannot check/call in current state")
        state.check_or_call()
    elif action.type == ActionType.BET_OR_RAISE:
        if not state.can_complete_bet_or_raise_to():
            raise ValueError("Cannot bet/raise in current state")
        state.complete_bet_or_raise_to()
    else:
        raise ValueError(f"Unknown action type: {action.type}")


def get_legal_actions(state: "State") -> list[Action]:
    """
    Get list of legal actions for the current state.

    Uses PokerKit query methods to determine valid actions.

    Args:
        state: PokerKit State object

    Returns:
        List of legal Action objects
    """
    actions = []
    if state.can_fold():
        actions.append(FOLD)
    if state.can_check_or_call():
        actions.append(CHECK_OR_CALL)
    if state.can_complete_bet_or_raise_to():
        actions.append(BET_OR_RAISE)
    return actions
