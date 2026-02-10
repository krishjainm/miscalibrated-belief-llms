"""Random agent that selects uniformly from legal actions."""

import random
from typing import Optional

from poker_env.agents.base import BaseAgent
from poker_env.actions import Action
from poker_env.obs import Obs


class RandomAgent(BaseAgent):
    """
    Agent that selects uniformly at random from legal actions.

    Useful as a baseline and for testing environment correctness.
    """

    def __init__(self, seed: Optional[int] = None, name: str = "RandomAgent"):
        """
        Initialize the random agent.

        Args:
            seed: Optional random seed for reproducibility
            name: Human-readable name for the agent
        """
        super().__init__(name=name)
        self.seed = seed
        self.rng = random.Random(seed)

    def act(self, obs: Obs) -> Action:
        """
        Select a random legal action.

        Args:
            obs: Current observation including legal actions

        Returns:
            Randomly selected action from obs.legal_actions

        Raises:
            ValueError: If no legal actions are available
        """
        if not obs.legal_actions:
            raise ValueError("No legal actions available")

        return self.rng.choice(obs.legal_actions)

    def reset(self) -> None:
        """Reset the random state (re-seed if seed was provided)."""
        if self.seed is not None:
            self.rng = random.Random(self.seed)
