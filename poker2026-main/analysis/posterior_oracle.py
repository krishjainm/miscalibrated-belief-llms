"""
Posterior oracles for computing P(opponent_hand | public_info).

Two oracles are provided:
1. CardOnlyPosterior: Uniform over hands consistent with blockers (ignores actions)
2. StrategyAwarePosterior: Action-likelihood weighted using an opponent model

The posteriors are model-dependent - there is no "true" posterior without
specifying assumptions about opponent behavior. This is scientifically honest
and enables robustness analysis across different assumptions.
"""

from typing import Protocol
from dataclasses import dataclass

from analysis.buckets import (
    BUCKET_NAMES,
    get_valid_hands_for_bucket,
    get_bucket_prior,
    hand_to_bucket,
)
from analysis.opponent_model import (
    OpponentModel,
    PublicState,
    ActionType,
)


# ============================================================================
# CardOnlyPosterior (Oracle A)
# ============================================================================

class CardOnlyPosterior:
    """
    Baseline posterior: uniform over hands consistent with blockers.
    
    This oracle ignores betting history entirely. It provides a baseline
    that only accounts for card removal (blockers from hero's hand and board).
    
    Use for:
    - Baseline comparison (how much does action history help?)
    - Quick computation when opponent model is unknown
    """
    
    def compute(
        self,
        hero_hole: list[str],
        board: list[str],
        buckets: bool = True,
    ) -> dict[str, float]:
        """
        Compute uniform posterior over hands consistent with blockers.
        
        Args:
            hero_hole: Hero's hole cards
            board: Board cards
            buckets: If True, return bucket probabilities. If False, return hand probabilities.
            
        Returns:
            Probability distribution over buckets or hands
        """
        blockers = hero_hole + board
        
        if buckets:
            return get_bucket_prior(blockers)
        else:
            # Return uniform over all valid hands
            all_hands = []
            for bucket in BUCKET_NAMES:
                all_hands.extend(get_valid_hands_for_bucket(bucket, blockers))
            
            if not all_hands:
                return {}
            
            prob = 1.0 / len(all_hands)
            return {f"{h[0]}{h[1]}": prob for h in all_hands}
    
    def compute_for_hand(
        self,
        hero_hole: list[str],
        board: list[str],
        opponent_hand: tuple[str, str],
    ) -> float:
        """
        Compute probability of a specific opponent hand.
        
        Args:
            hero_hole: Hero's hole cards
            board: Board cards  
            opponent_hand: Specific hand to query
            
        Returns:
            P(opponent_hand | blockers) or 0 if hand conflicts with blockers
        """
        blockers = set(hero_hole + board)
        
        # Check if hand conflicts with blockers
        if opponent_hand[0] in blockers or opponent_hand[1] in blockers:
            return 0.0
        
        # Count total valid hands
        total_hands = 0
        for bucket in BUCKET_NAMES:
            total_hands += len(get_valid_hands_for_bucket(bucket, list(blockers)))
        
        if total_hands == 0:
            return 0.0
        
        return 1.0 / total_hands


# ============================================================================
# StrategyAwarePosterior (Oracle B)
# ============================================================================

@dataclass
class OpponentAction:
    """Record of an opponent action for likelihood computation."""
    street: str
    action: ActionType
    state_at_action: PublicState | None = None


class StrategyAwarePosterior:
    """
    Main posterior oracle: P(hand | public_history) using opponent model.
    
    Computes:
        P(h | H_public) ∝ P(h) × ∏_t P(a_t^opp | h, s_t)
    
    Where:
    - P(h) is the prior (uniform over valid hands)
    - P(a_t^opp | h, s_t) is the action likelihood from the opponent model
    
    The posterior is "Bayes-optimal relative to the chosen opponent model."
    Different opponent models yield different posteriors, enabling robustness
    analysis of LLM beliefs across different assumptions.
    """
    
    def __init__(self, opponent_model: OpponentModel):
        """
        Initialize with an opponent model.
        
        Args:
            opponent_model: Model that provides P(action | hand, state)
        """
        self.opponent_model = opponent_model
    
    def compute(
        self,
        hero_hole: list[str],
        board: list[str],
        opponent_actions: list[dict],
        buckets: bool = True,
    ) -> dict[str, float]:
        """
        Compute action-likelihood weighted posterior.
        
        Args:
            hero_hole: Hero's hole cards
            board: Board cards
            opponent_actions: List of opponent actions with state info
                Each dict should have: {street, action, state (optional)}
            buckets: If True, return bucket probabilities
            
        Returns:
            Posterior distribution over buckets or hands
        """
        blockers = hero_hole + board
        
        if buckets:
            return self._compute_bucket_posterior(blockers, opponent_actions)
        else:
            return self._compute_hand_posterior(blockers, opponent_actions)
    
    def _compute_bucket_posterior(
        self,
        blockers: list[str],
        opponent_actions: list[dict],
    ) -> dict[str, float]:
        """Compute posterior over buckets."""
        # Get prior over buckets
        prior = get_bucket_prior(blockers)
        
        if not opponent_actions:
            return prior
        
        # Compute likelihood for each bucket
        bucket_likelihoods = {}
        for bucket in BUCKET_NAMES:
            valid_hands = get_valid_hands_for_bucket(bucket, blockers)
            if not valid_hands:
                bucket_likelihoods[bucket] = 0.0
                continue
            
            # Average likelihood over hands in bucket
            total_likelihood = 0.0
            for hand in valid_hands:
                hand_likelihood = self._compute_hand_likelihood(hand, opponent_actions)
                total_likelihood += hand_likelihood
            
            bucket_likelihoods[bucket] = total_likelihood / len(valid_hands)
        
        # Compute posterior: P(bucket) ∝ prior(bucket) × likelihood(bucket)
        unnormalized = {
            bucket: prior[bucket] * bucket_likelihoods[bucket]
            for bucket in BUCKET_NAMES
        }
        
        total = sum(unnormalized.values())
        if total == 0:
            return prior  # Fall back to prior if all likelihoods are 0
        
        return {bucket: p / total for bucket, p in unnormalized.items()}
    
    def _compute_hand_posterior(
        self,
        blockers: list[str],
        opponent_actions: list[dict],
    ) -> dict[str, float]:
        """Compute posterior over individual hands."""
        # Get all valid hands
        all_hands = []
        for bucket in BUCKET_NAMES:
            all_hands.extend(get_valid_hands_for_bucket(bucket, blockers))
        
        if not all_hands:
            return {}
        
        # Uniform prior
        prior_prob = 1.0 / len(all_hands)
        
        # Compute likelihood for each hand
        hand_posteriors = {}
        for hand in all_hands:
            likelihood = self._compute_hand_likelihood(hand, opponent_actions)
            hand_posteriors[f"{hand[0]}{hand[1]}"] = prior_prob * likelihood
        
        # Normalize
        total = sum(hand_posteriors.values())
        if total == 0:
            return {h: prior_prob for h in hand_posteriors}
        
        return {h: p / total for h, p in hand_posteriors.items()}
    
    def _compute_hand_likelihood(
        self,
        hand: tuple[str, str],
        opponent_actions: list[dict],
    ) -> float:
        """
        Compute P(actions | hand) = ∏_t P(a_t | hand, s_t).
        
        Args:
            hand: Specific opponent hand
            opponent_actions: List of action records
            
        Returns:
            Product of action likelihoods (may be very small)
        """
        likelihood = 1.0
        
        for action_record in opponent_actions:
            # Parse action type
            action_str = action_record.get("action", "")
            if action_str in ("FOLD", "Folding"):
                action_type = ActionType.FOLD
            elif action_str in ("CHECK_OR_CALL", "CHECK", "CALL", "CheckingOrCalling"):
                action_type = ActionType.CHECK_CALL
            elif action_str in ("BET_OR_RAISE", "BET", "RAISE", "CompletionBettingOrRaisingTo"):
                action_type = ActionType.BET_RAISE
            else:
                # Unknown action type, skip
                continue
            
            # Get or create state
            if "state" in action_record and action_record["state"]:
                state = action_record["state"]
                if not isinstance(state, PublicState):
                    state = PublicState.from_obs(state)
            else:
                # Create minimal state from action record
                state = PublicState(
                    street=action_record.get("street", "PREFLOP"),
                    board=[],
                    pot=0,
                    bet_to_call=0,
                    num_raises_this_street=0,
                    history=[],
                )
            
            # Get action probability from opponent model
            action_prob = self.opponent_model.action_prob(hand, state, action_type)
            
            # Multiply into likelihood (with floor to avoid 0)
            likelihood *= max(action_prob, 1e-10)
        
        return likelihood
    
    def compute_for_hand(
        self,
        hero_hole: list[str],
        board: list[str],
        opponent_actions: list[dict],
        opponent_hand: tuple[str, str],
    ) -> float:
        """
        Compute posterior probability of a specific opponent hand.
        
        Args:
            hero_hole: Hero's hole cards
            board: Board cards
            opponent_actions: Action history
            opponent_hand: Hand to query
            
        Returns:
            P(opponent_hand | public_history)
        """
        blockers = set(hero_hole + board)
        
        # Check blockers
        if opponent_hand[0] in blockers or opponent_hand[1] in blockers:
            return 0.0
        
        # Compute full posterior and look up this hand
        posterior = self.compute(
            hero_hole=hero_hole,
            board=board,
            opponent_actions=opponent_actions,
            buckets=False,
        )
        
        hand_key = f"{opponent_hand[0]}{opponent_hand[1]}"
        return posterior.get(hand_key, 0.0)


# ============================================================================
# Utility Functions
# ============================================================================

def extract_opponent_actions(history: list[dict], hero_index: int = 0) -> list[dict]:
    """
    Extract opponent actions from a game history.
    
    Args:
        history: Full action history from game
        hero_index: Index of hero player (actions from other players are opponent)
        
    Returns:
        List of opponent action records
    """
    opponent_actions = []
    
    for event in history:
        # Skip non-action events
        event_type = event.get("event", event.get("op", ""))
        if event_type not in ("FOLD", "CHECK", "CALL", "BET", "RAISE",
                              "Folding", "CheckingOrCalling", "CompletionBettingOrRaisingTo"):
            continue
        
        player = event.get("player")
        if player is None or player == hero_index:
            continue
        
        # This is an opponent action
        opponent_actions.append({
            "action": event_type,
            "street": event.get("street", "PREFLOP"),
            "amount": event.get("amount"),
            "player": player,
        })
    
    return opponent_actions
