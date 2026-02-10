"""Call-station agent that always checks or calls."""

from poker_env.agents.base import BaseAgent
from poker_env.actions import Action, ActionType, CHECK_OR_CALL
from poker_env.obs import Obs


class CallAgent(BaseAgent):
    """
    Agent that always checks or calls (never folds, never raises).

    This is a simple passive baseline agent, also known as a "calling station"
    in poker terminology.
    """

    def __init__(self, name: str = "CallAgent"):
        """
        Initialize the call agent.

        Args:
            name: Human-readable name for the agent
        """
        super().__init__(name=name)

    def act(self, obs: Obs) -> Action:
        """
        Select check/call if available, otherwise fold.

        Args:
            obs: Current observation including legal actions

        Returns:
            CHECK_OR_CALL if legal, otherwise FOLD

        Raises:
            ValueError: If no legal actions are available
        """
        if not obs.legal_actions:
            raise ValueError("No legal actions available")

        # Prefer check/call
        for action in obs.legal_actions:
            if action.type == ActionType.CHECK_OR_CALL:
                return action

        # Fall back to fold if check/call not available (shouldn't happen)
        for action in obs.legal_actions:
            if action.type == ActionType.FOLD:
                return action

        # If somehow neither is available, return first legal action
        return obs.legal_actions[0]
