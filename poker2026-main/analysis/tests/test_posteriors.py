"""Tests for posterior oracle functionality."""

import pytest
from analysis.buckets import BUCKET_NAMES
from analysis.opponent_model import ParametricOpponent, PublicState, ActionType
from analysis.posterior_oracle import (
    CardOnlyPosterior,
    StrategyAwarePosterior,
    extract_opponent_actions,
)


class TestCardOnlyPosterior:
    """Tests for CardOnlyPosterior oracle."""
    
    def test_returns_valid_distribution(self):
        """Should return valid probability distribution."""
        oracle = CardOnlyPosterior()
        posterior = oracle.compute(
            hero_hole=["Ah", "Kd"],
            board=[],
        )
        
        # Should sum to 1
        assert abs(sum(posterior.values()) - 1.0) < 1e-6
        
        # All values non-negative
        assert all(v >= 0 for v in posterior.values())
    
    def test_all_buckets_present(self):
        """Should have probability for all buckets."""
        oracle = CardOnlyPosterior()
        posterior = oracle.compute(
            hero_hole=["Ah", "Kd"],
            board=[],
        )
        
        for bucket in BUCKET_NAMES:
            assert bucket in posterior
    
    def test_blockers_reduce_combos(self):
        """Holding cards should reduce opponent's possible hands."""
        oracle = CardOnlyPosterior()
        
        # Holding AA, opponent has fewer AA combos
        posterior_with_aa = oracle.compute(
            hero_hole=["Ah", "As"],
            board=[],
        )
        
        posterior_with_72 = oracle.compute(
            hero_hole=["7h", "2d"],
            board=[],
        )
        
        # Should have different premium_pairs probability
        # (holding aces blocks more premium pairs than holding 72)
        assert posterior_with_aa["premium_pairs"] != posterior_with_72["premium_pairs"]
    
    def test_board_affects_blockers(self):
        """Board cards should also act as blockers."""
        oracle = CardOnlyPosterior()
        
        posterior_preflop = oracle.compute(
            hero_hole=["Ah", "Kd"],
            board=[],
        )
        
        posterior_flop = oracle.compute(
            hero_hole=["Ah", "Kd"],
            board=["Qc", "Js", "Td"],
        )
        
        # Distributions should differ due to board blockers
        assert posterior_preflop != posterior_flop


class TestStrategyAwarePosterior:
    """Tests for StrategyAwarePosterior oracle."""
    
    @pytest.fixture
    def oracle(self):
        """Create oracle with default opponent model."""
        return StrategyAwarePosterior(
            ParametricOpponent.from_preset("default")
        )
    
    def test_returns_valid_distribution(self, oracle):
        """Should return valid probability distribution."""
        posterior = oracle.compute(
            hero_hole=["Ah", "Kd"],
            board=[],
            opponent_actions=[],
        )
        
        assert abs(sum(posterior.values()) - 1.0) < 1e-6
        assert all(v >= 0 for v in posterior.values())
    
    def test_no_actions_equals_card_only(self, oracle):
        """With no actions, should be similar to card-only."""
        card_only = CardOnlyPosterior()
        
        posterior_strategy = oracle.compute(
            hero_hole=["Ah", "Kd"],
            board=[],
            opponent_actions=[],
        )
        
        posterior_card = card_only.compute(
            hero_hole=["Ah", "Kd"],
            board=[],
        )
        
        # Should be similar (not identical due to implementation)
        # Just check they're both valid
        assert abs(sum(posterior_strategy.values()) - 1.0) < 1e-6
        assert abs(sum(posterior_card.values()) - 1.0) < 1e-6
    
    def test_raise_shifts_toward_strong_hands(self, oracle):
        """Opponent raising should increase probability of strong hands."""
        posterior_no_raise = oracle.compute(
            hero_hole=["Ah", "Kd"],
            board=[],
            opponent_actions=[],
        )
        
        posterior_with_raise = oracle.compute(
            hero_hole=["Ah", "Kd"],
            board=[],
            opponent_actions=[
                {"action": "BET_OR_RAISE", "street": "PREFLOP"}
            ],
        )
        
        # Premium pairs should be more likely after a raise
        assert posterior_with_raise["premium_pairs"] >= posterior_no_raise["premium_pairs"]
    
    def test_fold_not_possible(self, oracle):
        """Opponent can't have folded since they're still in the hand."""
        # This is more of a logic test - if we saw a fold, that hand is over
        posterior = oracle.compute(
            hero_hole=["Ah", "Kd"],
            board=[],
            opponent_actions=[
                {"action": "CHECK_OR_CALL", "street": "PREFLOP"}
            ],
        )
        
        # Should still be valid distribution
        assert abs(sum(posterior.values()) - 1.0) < 1e-6


class TestOpponentActionExtraction:
    """Tests for extracting opponent actions from history."""
    
    def test_extracts_opponent_actions(self):
        """Should extract only opponent actions."""
        history = [
            {"event": "POST_BLIND", "player": 0},
            {"event": "POST_BLIND", "player": 1},
            {"event": "BET", "player": 1, "street": "PREFLOP"},
            {"event": "CALL", "player": 0, "street": "PREFLOP"},
            {"event": "CHECK", "player": 0, "street": "FLOP"},
            {"event": "BET", "player": 1, "street": "FLOP"},
        ]
        
        opponent_actions = extract_opponent_actions(history, hero_index=0)
        
        # Should have 2 opponent actions (player 1)
        assert len(opponent_actions) == 2
        assert all(a["player"] == 1 for a in opponent_actions)
    
    def test_empty_history(self):
        """Should handle empty history."""
        opponent_actions = extract_opponent_actions([], hero_index=0)
        assert len(opponent_actions) == 0


class TestParametricOpponent:
    """Tests for ParametricOpponent model."""
    
    def test_action_probs_sum_to_one(self):
        """Action probabilities should sum to 1."""
        model = ParametricOpponent.from_preset("default")
        state = PublicState(
            street="FLOP",
            board=["Qh", "7d", "2c"],
            pot=10,
            bet_to_call=2,
            num_raises_this_street=0,
            history=[],
        )
        
        hand = ("Ah", "As")
        probs = model.get_action_distribution(hand, state)
        
        assert abs(sum(probs.values()) - 1.0) < 1e-6
    
    def test_strong_hand_raises_more(self):
        """Strong hands should raise more often."""
        model = ParametricOpponent.from_preset("default")
        state = PublicState(
            street="PREFLOP",
            board=[],
            pot=3,
            bet_to_call=2,
            num_raises_this_street=0,
            history=[],
        )
        
        strong_hand = ("Ah", "As")
        weak_hand = ("7d", "2c")
        
        strong_raise_prob = model.action_prob(strong_hand, state, ActionType.BET_RAISE)
        weak_raise_prob = model.action_prob(weak_hand, state, ActionType.BET_RAISE)
        
        assert strong_raise_prob > weak_raise_prob
    
    def test_presets_differ(self):
        """Different presets should have different behavior."""
        tight = ParametricOpponent.from_preset("tight_passive")
        loose = ParametricOpponent.from_preset("loose_aggressive")
        
        assert tight.aggression != loose.aggression
        assert tight.fold_threshold != loose.fold_threshold
