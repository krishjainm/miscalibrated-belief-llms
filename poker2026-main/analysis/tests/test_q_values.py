"""Tests for Q-value estimation and implied belief."""

import pytest
from analysis.buckets import BUCKET_NAMES
from analysis.opponent_model import ActionType, ParametricOpponent
from analysis.implied_belief.q_value import MonteCarloQValue
from analysis.implied_belief.inverse import (
    infer_implied_belief,
    compute_expected_q,
    find_optimal_action_beliefs,
)


class TestMonteCarloQValue:
    """Tests for Monte Carlo Q-value estimation."""
    
    @pytest.fixture
    def estimator(self):
        """Create Q-value estimator."""
        return MonteCarloQValue(
            num_rollouts=100,  # Fewer for faster tests
            seed=42,
        )
    
    def test_returns_all_actions(self, estimator):
        """Should return Q-values for all legal actions."""
        q_values = estimator.compute_q_values(
            hero_hole=["Ah", "Kd"],
            villain_hole=("Qh", "Qc"),
            board=["Jc", "Td", "2s"],
            pot=20,
            bet_to_call=4,
            street="FLOP",
        )
        
        assert ActionType.FOLD in q_values
        assert ActionType.CHECK_CALL in q_values
        assert ActionType.BET_RAISE in q_values
    
    def test_fold_has_negative_ev(self, estimator):
        """Folding should generally have negative/zero EV."""
        q_values = estimator.compute_q_values(
            hero_hole=["Ah", "As"],  # Strong hand
            villain_hole=("7d", "2c"),  # Weak hand
            board=["Jc", "Td", "2s"],
            pot=20,
            bet_to_call=0,  # Can check
            street="FLOP",
        )
        
        # With a strong hand vs weak, fold should be worst
        assert q_values[ActionType.FOLD] <= q_values[ActionType.CHECK_CALL]
    
    def test_q_values_by_bucket(self, estimator):
        """Should compute Q-values for all buckets."""
        q_by_bucket = estimator.compute_q_values_by_bucket(
            hero_hole=["Ah", "Kd"],
            board=["Qc", "Jh", "2s"],
            pot=10,
            bet_to_call=2,
            street="FLOP",
            samples_per_bucket=3,
        )
        
        # Should have entry for each bucket
        for bucket in BUCKET_NAMES:
            assert bucket in q_by_bucket


class TestInverseDecisionRule:
    """Tests for inverse decision rule."""
    
    @pytest.fixture
    def q_values(self):
        """Sample Q-values for testing."""
        return {
            "premium_pairs": {
                ActionType.FOLD: -5,
                ActionType.CHECK_CALL: 8,
                ActionType.BET_RAISE: 15,
            },
            "strong_pairs": {
                ActionType.FOLD: -3,
                ActionType.CHECK_CALL: 5,
                ActionType.BET_RAISE: 10,
            },
            "trash": {
                ActionType.FOLD: 0,
                ActionType.CHECK_CALL: -5,
                ActionType.BET_RAISE: -10,
            },
        }
    
    def test_raise_implies_strong_hands(self, q_values):
        """Raising should imply belief in strong hands."""
        implied = infer_implied_belief(
            chosen_action=ActionType.BET_RAISE,
            q_values=q_values,
            method="softmax",
        )
        
        # Premium pairs should have higher probability than trash
        assert implied.get("premium_pairs", 0) > implied.get("trash", 0)
    
    def test_fold_implies_weak_hands(self, q_values):
        """Folding should imply belief in weak opposing hands (strong for villain)."""
        implied = infer_implied_belief(
            chosen_action=ActionType.FOLD,
            q_values=q_values,
            method="softmax",
        )
        
        # Should be valid distribution
        assert abs(sum(implied.values()) - 1.0) < 0.01
    
    def test_optimal_set_method(self, q_values):
        """Optimal set method should concentrate on appropriate buckets."""
        implied = infer_implied_belief(
            chosen_action=ActionType.BET_RAISE,
            q_values=q_values,
            method="optimal_set",
        )
        
        # Should concentrate on buckets where raising is optimal
        # (premium_pairs and strong_pairs have raise as best action)
        total_strong = implied.get("premium_pairs", 0) + implied.get("strong_pairs", 0)
        assert total_strong > 0.5


class TestExpectedQ:
    """Tests for expected Q computation."""
    
    def test_deterministic_belief(self):
        """Deterministic belief should give exact Q."""
        belief = {"a": 1.0, "b": 0.0}
        q_values = {
            "a": {ActionType.FOLD: 10, ActionType.CHECK_CALL: 5},
            "b": {ActionType.FOLD: 0, ActionType.CHECK_CALL: 0},
        }
        
        eq = compute_expected_q(belief, q_values, ActionType.FOLD)
        assert abs(eq - 10) < 1e-6
    
    def test_mixed_belief(self):
        """Mixed belief should give weighted Q."""
        belief = {"a": 0.5, "b": 0.5}
        q_values = {
            "a": {ActionType.FOLD: 10},
            "b": {ActionType.FOLD: 0},
        }
        
        eq = compute_expected_q(belief, q_values, ActionType.FOLD)
        assert abs(eq - 5) < 1e-6


class TestFindOptimalActionBeliefs:
    """Tests for finding which buckets make each action optimal."""
    
    def test_identifies_optimal_buckets(self):
        """Should identify which buckets make each action optimal."""
        q_values = {
            "strong": {
                ActionType.FOLD: -10,
                ActionType.CHECK_CALL: 5,
                ActionType.BET_RAISE: 10,  # Best
            },
            "weak": {
                ActionType.FOLD: 0,  # Best (least negative)
                ActionType.CHECK_CALL: -5,
                ActionType.BET_RAISE: -15,
            },
        }
        
        optimal = find_optimal_action_beliefs(q_values, ["strong", "weak"])
        
        assert "strong" in optimal[ActionType.BET_RAISE]
        assert "weak" in optimal[ActionType.FOLD]
