"""Base agent interface for poker agents."""

from abc import ABC, abstractmethod
from typing import Optional

from poker_env.actions import Action
from poker_env.obs import Obs


class BaseAgent(ABC):
    """
    Abstract base class for poker agents.

    All agents must implement the act() method to select actions.
    The belief() method is optional but required for belief modeling research.
    """

    def __init__(self, name: str = "BaseAgent"):
        """
        Initialize the agent.

        Args:
            name: Human-readable name for the agent
        """
        self.name = name

    @abstractmethod
    def act(self, obs: Obs) -> Action:
        """
        Select an action given an observation.

        Args:
            obs: Current observation including legal actions

        Returns:
            Selected action from obs.legal_actions
        """
        raise NotImplementedError

    def belief(self, obs: Obs) -> Optional[dict]:
        """
        Return structured belief about the game state.

        Optional method for belief modeling research.
        Should return probabilities that sum to 1.

        Args:
            obs: Current observation

        Returns:
            Dict with belief probabilities, e.g.:
            {
                "p_win": 0.45,
                "p_tie": 0.05,
                "p_lose": 0.50
            }
            Returns None if agent doesn't support belief elicitation.
        """
        return None

    def reset(self) -> None:
        """
        Reset agent state for a new hand.

        Override this method if the agent maintains state across actions.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
