"""
Equity oracle using Monte Carlo simulation.

This computes win/tie/lose probabilities given TRUE opponent hole cards.
This is NOT a Bayesian posterior (which would require only observable history).

Use for:
- Ground truth labels for evaluation
- Comparing agent beliefs against actual equity

Do NOT use as:
- A proxy for what the agent "should" believe (that requires posterior inference)
"""

import random
from typing import Optional
from itertools import combinations

from pokerkit import StandardHighHand

from poker_env.deck import FULL_DECK, parse_cards


class EquityOracle:
    """
    Oracle that computes equity (win/tie/lose) given true hole cards.

    This is an "outcome oracle" - it uses hidden information (opponent cards)
    to compute ground truth. The agent NEVER sees this before acting.

    Used for:
    - Logging ground truth for later analysis
    - Comparing stated beliefs to actual equity
    - Evaluating belief calibration

    NOT a Bayesian posterior - for that you'd need to compute
    P(opponent_hand | observable_history) which requires an opponent model.
    """

    def __init__(self, num_samples: int = 10000, seed: Optional[int] = None):
        """
        Initialize the oracle.

        Args:
            num_samples: Number of Monte Carlo samples for estimation
            seed: Optional random seed for reproducibility
        """
        self.num_samples = num_samples
        self.seed = seed
        self.rng = random.Random(seed)

    def compute(
        self,
        hero_hole: list[str],
        opponent_holes: list[list[str]],
        board: list[str],
    ) -> dict:
        """
        Compute equity (win/tie/lose) given true opponent hole cards.

        This is ground truth for evaluation, NOT what the agent should believe.

        Args:
            hero_hole: Hero's hole cards, e.g., ["Ac", "As"]
            opponent_holes: List of opponent hole cards, e.g., [["Kh", "Kd"], ["Qc", "Qd"]]
            board: Current board cards, e.g., ["Jc", "3d", "5c"]

        Returns:
            Dict with:
            - "equity_win": probability hero wins outright
            - "equity_tie": probability hero ties for best
            - "equity_lose": probability hero loses
        """
        # Handle single opponent (legacy format)
        if opponent_holes and isinstance(opponent_holes[0], str):
            opponent_holes = [opponent_holes]

        # If board is complete (5 cards), compute exact result
        if len(board) >= 5:
            return self._compute_exact(hero_hole, opponent_holes, board[:5])

        # Otherwise, Monte Carlo simulation
        return self._compute_monte_carlo(hero_hole, opponent_holes, board)

    def compute_headsup(
        self,
        hero_hole: list[str],
        villain_hole: list[str],
        board: list[str],
    ) -> dict:
        """
        Compute equity for heads-up (convenience method).

        Args:
            hero_hole: Hero's hole cards
            villain_hole: Villain's hole cards
            board: Current board cards

        Returns:
            Dict with equity_win, equity_tie, equity_lose
        """
        return self.compute(hero_hole, [villain_hole], board)

    def _compute_exact(
        self,
        hero_hole: list[str],
        opponent_holes: list[list[str]],
        board: list[str],
    ) -> dict:
        """Compute exact result when board is complete."""
        hero_hand = self._evaluate_hand(hero_hole, board)

        opponent_hands = [self._evaluate_hand(opp, board) for opp in opponent_holes]
        best_opponent = max(opponent_hands)

        if hero_hand > best_opponent:
            return {"equity_win": 1.0, "equity_tie": 0.0, "equity_lose": 0.0}
        elif hero_hand < best_opponent:
            return {"equity_win": 0.0, "equity_tie": 0.0, "equity_lose": 1.0}
        else:
            return {"equity_win": 0.0, "equity_tie": 1.0, "equity_lose": 0.0}

    def _compute_monte_carlo(
        self,
        hero_hole: list[str],
        opponent_holes: list[list[str]],
        board: list[str],
    ) -> dict:
        """Compute probabilities via Monte Carlo simulation."""
        # Cards that are no longer available
        dead_cards = set(hero_hole + board)
        for opp in opponent_holes:
            dead_cards.update(opp)

        # Remaining deck
        deck = [c for c in FULL_DECK if c not in dead_cards]

        # Number of cards needed to complete the board
        cards_needed = 5 - len(board)

        wins = 0
        ties = 0
        losses = 0

        # Use enumeration if possible (small number of combinations)
        possible_runouts = list(combinations(deck, cards_needed))

        if len(possible_runouts) <= self.num_samples:
            # Enumerate all possibilities
            for runout in possible_runouts:
                full_board = board + list(runout)
                result = self._compare_hands_multiway(hero_hole, opponent_holes, full_board)
                if result > 0:
                    wins += 1
                elif result < 0:
                    losses += 1
                else:
                    ties += 1

            total = len(possible_runouts)
        else:
            # Monte Carlo sampling
            for _ in range(self.num_samples):
                runout = self.rng.sample(deck, cards_needed)
                full_board = board + runout
                result = self._compare_hands_multiway(hero_hole, opponent_holes, full_board)
                if result > 0:
                    wins += 1
                elif result < 0:
                    losses += 1
                else:
                    ties += 1

            total = self.num_samples

        return {
            "equity_win": wins / total,
            "equity_tie": ties / total,
            "equity_lose": losses / total,
        }

    def _evaluate_hand(self, hole: list[str], board: list[str]) -> StandardHighHand:
        """
        Evaluate a hand using PokerKit.

        Args:
            hole: Two hole cards
            board: Five board cards

        Returns:
            StandardHighHand object for comparison
        """
        # Convert string cards to PokerKit format
        hole_str = "".join(hole)
        board_str = "".join(board)

        return StandardHighHand.from_game(hole_str, board_str)

    def _compare_hands_multiway(
        self,
        hero_hole: list[str],
        opponent_holes: list[list[str]],
        board: list[str],
    ) -> int:
        """
        Compare hero's hand against multiple opponents.

        Returns:
            > 0 if hero wins outright
            < 0 if any opponent beats hero
            0 if hero ties with best opponent(s)
        """
        hero_hand = self._evaluate_hand(hero_hole, board)

        opponent_hands = [self._evaluate_hand(opp, board) for opp in opponent_holes]
        best_opponent = max(opponent_hands)

        if hero_hand > best_opponent:
            return 1
        elif hero_hand < best_opponent:
            return -1
        else:
            return 0

    def reset_seed(self, seed: Optional[int] = None) -> None:
        """Reset the random number generator with a new seed."""
        self.seed = seed
        self.rng = random.Random(seed)


# Alias for backwards compatibility
WinProbOracle = EquityOracle
