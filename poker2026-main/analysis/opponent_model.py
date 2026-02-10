"""
Opponent model interface and implementations.

Defines the protocol for behavioral models that estimate P(action | hand, state).
Used by StrategyAwarePosterior to compute action-likelihood weighted posteriors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from enum import Enum

from analysis.buckets import RANKS


class ActionType(Enum):
    """Poker action types."""
    FOLD = "FOLD"
    CHECK_CALL = "CHECK_OR_CALL"
    BET_RAISE = "BET_OR_RAISE"


@dataclass
class PublicState:
    """
    Public game state visible to all players.
    
    Used by opponent models to determine action probabilities.
    """
    street: str  # "PREFLOP", "FLOP", "TURN", "RIVER"
    board: list[str]  # Board cards, e.g., ["Jc", "3d", "5c"]
    pot: int  # Total pot size
    bet_to_call: int  # Amount needed to call
    num_raises_this_street: int  # For raise cap tracking
    history: list[dict]  # Action history [{player, action, amount, street}, ...]
    
    @classmethod
    def from_obs(cls, obs: dict) -> "PublicState":
        """Create from observation dict."""
        return cls(
            street=obs.get("street", "PREFLOP"),
            board=obs.get("board", []),
            pot=obs.get("pot_total", 0),
            bet_to_call=obs.get("bet_to_call", 0),
            num_raises_this_street=4 - obs.get("raises_remaining", 4),
            history=obs.get("history", []),
        )


@runtime_checkable
class OpponentModel(Protocol):
    """
    Protocol for opponent behavioral models.
    
    Implementations must provide action_prob() which returns
    P(action | hand, state) for Bayesian posterior computation.
    """
    
    def action_prob(
        self,
        hand: tuple[str, str],
        state: PublicState,
        action: ActionType,
    ) -> float:
        """
        Compute probability of action given hand and state.
        
        Args:
            hand: Opponent's hole cards (card1, card2)
            state: Current public game state
            action: Action to compute probability for
            
        Returns:
            P(action | hand, state) in [0, 1]
        """
        ...


# ============================================================================
# Hand Strength Computation
# ============================================================================

def compute_hand_strength(
    hand: tuple[str, str],
    board: list[str],
) -> float:
    """
    Compute simplified hand strength in [0, 1].
    
    This is a heuristic approximation, not true equity.
    Uses preflop hand rankings + board texture adjustments.
    
    Args:
        hand: Hole cards
        board: Board cards
        
    Returns:
        Estimated hand strength in [0, 1]
    """
    r1, r2 = hand[0][0], hand[1][0]
    s1, s2 = hand[0][1], hand[1][1]
    suited = s1 == s2
    
    # Get rank indices (0 = A, 12 = 2)
    idx1, idx2 = RANKS.index(r1), RANKS.index(r2)
    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
    
    # Base strength from preflop hand rankings
    # Pairs
    if r1 == r2:
        # AA = 1.0, 22 = 0.5
        pair_strength = 1.0 - (idx1 / 12) * 0.5
        base = pair_strength
    else:
        # High card combinations
        gap = idx2 - idx1
        high_card_value = 1.0 - (idx1 / 12) * 0.3
        low_card_value = 1.0 - (idx2 / 12) * 0.2
        
        # Connectedness bonus
        connect_bonus = 0.1 if gap <= 2 else 0.0
        
        # Suited bonus
        suited_bonus = 0.1 if suited else 0.0
        
        base = (high_card_value * 0.4 + low_card_value * 0.3 + 
                connect_bonus + suited_bonus)
        base = min(0.9, base)  # Cap below pairs
    
    # Post-flop adjustments (simplified)
    if board:
        board_ranks = [c[0] for c in board]
        
        # Pair on board?
        if r1 in board_ranks or r2 in board_ranks:
            base = min(1.0, base + 0.15)
        
        # Flush draw?
        if suited:
            board_suits = [c[1] for c in board]
            flush_count = sum(1 for s in board_suits if s == s1)
            if flush_count >= 2:
                base = min(1.0, base + 0.1)
    
    return max(0.0, min(1.0, base))


# ============================================================================
# Parametric Opponent Implementation
# ============================================================================

class ParametricOpponent:
    """
    Simple parametric opponent model.
    
    Models opponent behavior with three tunable parameters:
    - aggression: How often they bet/raise vs check/call
    - fold_threshold: Minimum hand strength to continue
    - bluff_freq: How often they raise with weak hands
    
    This provides interpretable baseline opponent models for computing
    strategy-aware posteriors. Different parameter settings model
    different player types (tight-passive, loose-aggressive, etc.).
    """
    
    # Presets for common player types
    PRESETS = {
        "tight_passive": {"aggression": 0.2, "fold_threshold": 0.4, "bluff_freq": 0.02},
        "tight_aggressive": {"aggression": 0.6, "fold_threshold": 0.4, "bluff_freq": 0.08},
        "loose_passive": {"aggression": 0.2, "fold_threshold": 0.2, "bluff_freq": 0.05},
        "loose_aggressive": {"aggression": 0.6, "fold_threshold": 0.2, "bluff_freq": 0.15},
        "default": {"aggression": 0.4, "fold_threshold": 0.3, "bluff_freq": 0.1},
    }
    
    def __init__(
        self,
        aggression: float = 0.4,
        fold_threshold: float = 0.3,
        bluff_freq: float = 0.1,
        name: str = "ParametricOpponent",
    ):
        """
        Initialize parametric opponent model.
        
        Args:
            aggression: P(raise | should_continue), range [0, 1]
            fold_threshold: Fold if hand_strength < threshold, range [0, 1]
            bluff_freq: P(raise | weak hand), range [0, 1]
            name: Model name for logging
        """
        self.aggression = aggression
        self.fold_threshold = fold_threshold
        self.bluff_freq = bluff_freq
        self.name = name
    
    @classmethod
    def from_preset(cls, preset: str) -> "ParametricOpponent":
        """Create from a named preset."""
        if preset not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(cls.PRESETS.keys())}")
        params = cls.PRESETS[preset]
        return cls(**params, name=f"ParametricOpponent({preset})")
    
    def action_prob(
        self,
        hand: tuple[str, str],
        state: PublicState,
        action: ActionType,
    ) -> float:
        """
        Compute P(action | hand, state).
        
        Model:
        1. Compute hand strength
        2. If strength < fold_threshold: high P(fold), low P(continue)
        3. If continuing: split between call and raise based on aggression
        4. Add bluff component for raises with weak hands
        """
        strength = compute_hand_strength(hand, state.board)
        
        # Adjust threshold based on pot odds (simplified)
        effective_threshold = self.fold_threshold
        if state.bet_to_call > 0 and state.pot > 0:
            pot_odds = state.bet_to_call / (state.pot + state.bet_to_call)
            # Need less strength when pot odds are good
            effective_threshold = self.fold_threshold * (0.5 + pot_odds)
        
        # Compute base probabilities
        if strength < effective_threshold:
            # Weak hand: mostly fold, sometimes bluff
            p_fold = 1.0 - self.bluff_freq - 0.05  # Small call frequency
            p_call = 0.05
            p_raise = self.bluff_freq
        else:
            # Strong enough to continue
            p_fold = 0.02  # Rare folds with playable hands
            
            # Split remaining probability between call and raise
            continue_prob = 1.0 - p_fold
            
            # Stronger hands raise more
            strength_factor = min(1.0, strength / 0.8)
            effective_aggression = self.aggression * strength_factor
            
            p_raise = continue_prob * effective_aggression
            p_call = continue_prob * (1.0 - effective_aggression)
        
        # Normalize to ensure sum = 1
        total = p_fold + p_call + p_raise
        p_fold /= total
        p_call /= total
        p_raise /= total
        
        # Return requested action probability
        if action == ActionType.FOLD:
            return p_fold
        elif action == ActionType.CHECK_CALL:
            return p_call
        else:  # BET_RAISE
            return p_raise
    
    def get_action_distribution(
        self,
        hand: tuple[str, str],
        state: PublicState,
    ) -> dict[ActionType, float]:
        """Get full action distribution for a hand/state."""
        return {
            action: self.action_prob(hand, state, action)
            for action in ActionType
        }
    
    def to_dict(self) -> dict:
        """Serialize model parameters."""
        return {
            "type": "ParametricOpponent",
            "name": self.name,
            "aggression": self.aggression,
            "fold_threshold": self.fold_threshold,
            "bluff_freq": self.bluff_freq,
        }


# ============================================================================
# CFR Opponent Stub (Future Implementation)
# ============================================================================

class CFROpponent:
    """
    Nash equilibrium opponent using pre-trained CFR strategy.
    
    FUTURE IMPLEMENTATION
    =====================
    
    Limit Hold'em was solved by Counterfactual Regret Minimization (CFR):
    - Bowling et al., 2015: "Heads-up limit hold'em poker is solved"
    - Science, Vol 347, Issue 6218
    
    Pre-trained strategies are available from University of Alberta:
    - http://poker.cs.ualberta.ca/
    - Cepheus strategy files
    
    Integration would provide "Bayes-optimal vs game-theoretic opponent"
    baseline for the strongest possible posterior oracle.
    
    Challenges:
    - Strategy files are large (~10GB for full resolution)
    - Need to map our state representation to CFR information sets
    - May need abstracted/bucketed strategy for tractability
    
    Resources:
    - Cepheus: http://poker.cs.ualberta.ca/
    - OpenSpiel CFR implementations: https://github.com/google-deepmind/open_spiel
    - PokerRL: https://github.com/TinkeringCode/PokerRL
    """
    
    def __init__(self, strategy_path: str):
        """
        Initialize CFR opponent from pre-trained strategy.
        
        Args:
            strategy_path: Path to CFR strategy file
            
        Raises:
            NotImplementedError: CFR integration not yet implemented
        """
        raise NotImplementedError(
            "CFR opponent integration is planned for future implementation. "
            "See class docstring for resources and integration notes. "
            "For now, use ParametricOpponent as a baseline."
        )
    
    def action_prob(
        self,
        hand: tuple[str, str],
        state: PublicState,
        action: ActionType,
    ) -> float:
        """Would return Nash equilibrium action probability."""
        raise NotImplementedError("CFR integration not yet implemented")
