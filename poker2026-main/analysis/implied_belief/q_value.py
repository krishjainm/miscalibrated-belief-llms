"""
Monte Carlo Q-value estimation for poker.

Estimates Q(action; opponent_hand) via rollout simulations.
For limit hold'em, the fixed bet sizes make this tractable.
"""

import random
from typing import Optional
from itertools import combinations
from dataclasses import dataclass

from analysis.buckets import (
    BUCKET_NAMES,
    get_valid_hands_for_bucket,
    hand_to_bucket,
)
from analysis.opponent_model import (
    OpponentModel,
    ParametricOpponent,
    PublicState,
    ActionType,
    compute_hand_strength,
)


# All cards for deck generation
RANKS = "AKQJT98765432"
SUITS = "shdc"
FULL_DECK = [f"{r}{s}" for r in RANKS for s in SUITS]


@dataclass
class RolloutState:
    """State for a single rollout simulation."""
    hero_hole: tuple[str, str]
    villain_hole: tuple[str, str]
    board: list[str]
    pot: int
    hero_invested: int
    villain_invested: int
    street: str
    hero_to_act: bool
    small_bet: int
    big_bet: int
    remaining_deck: list[str]


class MonteCarloQValue:
    """
    Estimate Q(action; opponent_hand) via Monte Carlo rollouts.
    
    For limit hold'em:
    - Fixed bet sizes make branching manageable
    - Use opponent model for opponent responses
    - Sample board runouts
    
    The Q-value Q(a; h) represents the expected value of taking action a
    given that the opponent holds hand h.
    """
    
    def __init__(
        self,
        opponent_model: Optional[OpponentModel] = None,
        num_rollouts: int = 500,
        seed: Optional[int] = None,
        small_bet: int = 2,
        big_bet: int = 4,
    ):
        """
        Initialize Q-value estimator.
        
        Args:
            opponent_model: Model for opponent actions (default: ParametricOpponent)
            num_rollouts: Number of Monte Carlo samples per Q-value
            seed: Random seed for reproducibility
            small_bet: Small bet size (preflop/flop)
            big_bet: Big bet size (turn/river)
        """
        self.opponent_model = opponent_model or ParametricOpponent.from_preset("default")
        self.num_rollouts = num_rollouts
        self.rng = random.Random(seed)
        self.small_bet = small_bet
        self.big_bet = big_bet
    
    def compute_q_values(
        self,
        hero_hole: list[str],
        villain_hole: tuple[str, str],
        board: list[str],
        pot: int,
        bet_to_call: int,
        street: str,
        legal_actions: list[ActionType] | None = None,
    ) -> dict[ActionType, float]:
        """
        Compute Q-values for each action given specific villain hand.
        
        Args:
            hero_hole: Hero's hole cards
            villain_hole: Villain's hole cards (known for this Q computation)
            board: Current board cards
            pot: Current pot size
            bet_to_call: Amount to call
            street: Current street
            legal_actions: Legal actions (default: all)
            
        Returns:
            Dict mapping action to expected value
        """
        if legal_actions is None:
            legal_actions = list(ActionType)
        
        # Get remaining deck
        used_cards = set(hero_hole) | set(villain_hole) | set(board)
        remaining_deck = [c for c in FULL_DECK if c not in used_cards]
        
        # Compute Q for each action
        q_values = {}
        for action in legal_actions:
            ev = self._estimate_action_ev(
                hero_hole=tuple(hero_hole),
                villain_hole=villain_hole,
                board=list(board),
                pot=pot,
                bet_to_call=bet_to_call,
                street=street,
                action=action,
                remaining_deck=remaining_deck,
            )
            q_values[action] = ev
        
        return q_values
    
    def compute_q_values_by_bucket(
        self,
        hero_hole: list[str],
        board: list[str],
        pot: int,
        bet_to_call: int,
        street: str,
        legal_actions: list[ActionType] | None = None,
        samples_per_bucket: int = 10,
    ) -> dict[str, dict[ActionType, float]]:
        """
        Compute Q-values for each bucket.
        
        Samples hands from each bucket and averages Q-values.
        
        Args:
            hero_hole: Hero's hole cards
            board: Board cards
            pot: Pot size
            bet_to_call: Call amount
            street: Street
            legal_actions: Legal actions
            samples_per_bucket: Hands to sample per bucket
            
        Returns:
            Dict[bucket][action] -> Q-value
        """
        if legal_actions is None:
            legal_actions = list(ActionType)
        
        blockers = list(hero_hole) + board
        q_by_bucket = {}
        
        for bucket in BUCKET_NAMES:
            valid_hands = get_valid_hands_for_bucket(bucket, blockers)
            
            if not valid_hands:
                # No valid hands in this bucket
                q_by_bucket[bucket] = {a: 0.0 for a in legal_actions}
                continue
            
            # Sample hands from bucket
            n_samples = min(samples_per_bucket, len(valid_hands))
            sampled_hands = self.rng.sample(valid_hands, n_samples)
            
            # Average Q-values over sampled hands
            bucket_qs = {a: [] for a in legal_actions}
            for villain_hand in sampled_hands:
                hand_qs = self.compute_q_values(
                    hero_hole=hero_hole,
                    villain_hole=villain_hand,
                    board=board,
                    pot=pot,
                    bet_to_call=bet_to_call,
                    street=street,
                    legal_actions=legal_actions,
                )
                for action, q in hand_qs.items():
                    bucket_qs[action].append(q)
            
            q_by_bucket[bucket] = {
                a: sum(qs) / len(qs) if qs else 0.0
                for a, qs in bucket_qs.items()
            }
        
        return q_by_bucket
    
    def _estimate_action_ev(
        self,
        hero_hole: tuple[str, str],
        villain_hole: tuple[str, str],
        board: list[str],
        pot: int,
        bet_to_call: int,
        street: str,
        action: ActionType,
        remaining_deck: list[str],
    ) -> float:
        """Estimate EV of taking an action via rollouts."""
        evs = []
        
        for _ in range(self.num_rollouts):
            ev = self._single_rollout(
                hero_hole=hero_hole,
                villain_hole=villain_hole,
                board=board.copy(),
                pot=pot,
                bet_to_call=bet_to_call,
                street=street,
                hero_action=action,
                remaining_deck=remaining_deck.copy(),
            )
            evs.append(ev)
        
        return sum(evs) / len(evs) if evs else 0.0
    
    def _single_rollout(
        self,
        hero_hole: tuple[str, str],
        villain_hole: tuple[str, str],
        board: list[str],
        pot: int,
        bet_to_call: int,
        street: str,
        hero_action: ActionType,
        remaining_deck: list[str],
    ) -> float:
        """
        Run a single rollout from the current state.
        
        Returns hero's profit/loss from this point.
        """
        # Shuffle deck for this rollout
        self.rng.shuffle(remaining_deck)
        deck_idx = 0
        
        # Track investments this hand
        hero_invested = 0
        villain_invested = 0
        current_pot = pot
        
        # Get bet size for current street
        bet_size = self.big_bet if street in ("TURN", "RIVER") else self.small_bet
        
        # Apply hero's action
        if hero_action == ActionType.FOLD:
            # Hero folds, loses nothing more
            return -hero_invested
        elif hero_action == ActionType.CHECK_CALL:
            hero_invested += bet_to_call
            current_pot += bet_to_call
        elif hero_action == ActionType.BET_RAISE:
            hero_invested += bet_to_call + bet_size
            current_pot += bet_to_call + bet_size
        
        # Villain responds (simplified: one response then showdown)
        villain_state = PublicState(
            street=street,
            board=board,
            pot=current_pot,
            bet_to_call=bet_size if hero_action == ActionType.BET_RAISE else 0,
            num_raises_this_street=1 if hero_action == ActionType.BET_RAISE else 0,
            history=[],
        )
        
        # Sample villain action
        villain_action = self._sample_villain_action(villain_hole, villain_state)
        
        if villain_action == ActionType.FOLD:
            # Villain folds, hero wins pot
            return current_pot - hero_invested
        elif villain_action == ActionType.CHECK_CALL:
            villain_invested += villain_state.bet_to_call
            current_pot += villain_state.bet_to_call
        elif villain_action == ActionType.BET_RAISE:
            raise_amount = villain_state.bet_to_call + bet_size
            villain_invested += raise_amount
            current_pot += raise_amount
            # Simplify: hero calls the raise
            call_amount = raise_amount - (hero_invested - bet_to_call)
            hero_invested += max(0, call_amount)
            current_pot += max(0, call_amount)
        
        # Complete the board
        cards_needed = 5 - len(board)
        if cards_needed > 0 and deck_idx + cards_needed <= len(remaining_deck):
            for _ in range(cards_needed):
                board.append(remaining_deck[deck_idx])
                deck_idx += 1
        
        # Showdown
        hero_strength = compute_hand_strength(hero_hole, board)
        villain_strength = compute_hand_strength(villain_hole, board)
        
        if hero_strength > villain_strength:
            # Hero wins
            return current_pot - hero_invested
        elif hero_strength < villain_strength:
            # Hero loses
            return -hero_invested
        else:
            # Tie - split pot
            return (current_pot / 2) - hero_invested
    
    def _sample_villain_action(
        self,
        villain_hole: tuple[str, str],
        state: PublicState,
    ) -> ActionType:
        """Sample villain action from opponent model."""
        probs = {
            action: self.opponent_model.action_prob(villain_hole, state, action)
            for action in ActionType
        }
        
        # Normalize
        total = sum(probs.values())
        if total <= 0:
            return ActionType.CHECK_CALL
        
        probs = {a: p / total for a, p in probs.items()}
        
        # Sample
        r = self.rng.random()
        cumulative = 0.0
        for action, prob in probs.items():
            cumulative += prob
            if r <= cumulative:
                return action
        
        return ActionType.CHECK_CALL
