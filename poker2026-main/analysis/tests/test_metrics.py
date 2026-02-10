"""Tests for metrics functionality."""

import pytest
import numpy as np
from analysis.buckets import BUCKET_NAMES
from analysis.metrics.calibration import (
    compute_kl_divergence,
    compute_js_divergence,
    compute_pce,
    compute_brier_score,
    compute_ece,
)
from analysis.metrics.coherence import (
    check_coherence,
    compute_coherence_summary,
)
from analysis.metrics.update_coherence import (
    compute_update_agreement,
    compute_belief_delta,
)
from analysis.metrics.belief_action import (
    compute_belief_action_divergence,
    compute_decision_regret,
)
from analysis.opponent_model import ActionType


class TestDivergences:
    """Tests for divergence computations."""
    
    def test_kl_same_distribution(self):
        """KL divergence of identical distributions should be 0."""
        p = {"a": 0.5, "b": 0.5}
        kl = compute_kl_divergence(p, p)
        assert kl < 1e-6
    
    def test_kl_different_distributions(self):
        """KL divergence of different distributions should be positive."""
        p = {"a": 0.9, "b": 0.1}
        q = {"a": 0.1, "b": 0.9}
        kl = compute_kl_divergence(p, q)
        assert kl > 0
    
    def test_js_symmetric(self):
        """JS divergence should be symmetric."""
        p = {"a": 0.7, "b": 0.3}
        q = {"a": 0.3, "b": 0.7}
        
        js_pq = compute_js_divergence(p, q)
        js_qp = compute_js_divergence(q, p)
        
        assert abs(js_pq - js_qp) < 1e-6
    
    def test_js_bounded(self):
        """JS divergence should be bounded by ln(2)."""
        p = {"a": 1.0, "b": 0.0}
        q = {"a": 0.0, "b": 1.0}
        
        js = compute_js_divergence(p, q)
        assert js <= np.log(2) + 1e-6


class TestPCE:
    """Tests for Posterior Calibration Error."""
    
    def test_perfect_calibration(self):
        """PCE should be 0 when beliefs match oracle."""
        beliefs = [{"a": 0.5, "b": 0.5} for _ in range(10)]
        oracles = [{"a": 0.5, "b": 0.5} for _ in range(10)]
        
        result = compute_pce(beliefs, oracles)
        assert result["pce"] < 1e-6
    
    def test_miscalibration(self):
        """PCE should be positive for mismatched beliefs."""
        beliefs = [{"a": 0.9, "b": 0.1} for _ in range(10)]
        oracles = [{"a": 0.1, "b": 0.9} for _ in range(10)]
        
        result = compute_pce(beliefs, oracles)
        assert result["pce"] > 0
    
    def test_returns_per_sample(self):
        """Should return per-sample divergences."""
        beliefs = [{"a": 0.5, "b": 0.5}, {"a": 0.6, "b": 0.4}]
        oracles = [{"a": 0.5, "b": 0.5}, {"a": 0.8, "b": 0.2}]
        
        result = compute_pce(beliefs, oracles)
        assert len(result["per_sample"]) == 2


class TestBrierAndECE:
    """Tests for Brier score and ECE."""
    
    def test_brier_perfect_calibration(self):
        """Perfect predictions should have Brier = 0."""
        preds = [1.0, 0.0, 1.0, 0.0]
        outcomes = [True, False, True, False]
        
        brier = compute_brier_score(preds, outcomes)
        assert brier < 1e-6
    
    def test_brier_worst_case(self):
        """Completely wrong predictions should have Brier = 1."""
        preds = [1.0, 1.0, 0.0, 0.0]
        outcomes = [False, False, True, True]
        
        brier = compute_brier_score(preds, outcomes)
        assert abs(brier - 1.0) < 1e-6
    
    def test_ece_returns_bin_data(self):
        """ECE should return reliability diagram data."""
        preds = [0.1, 0.2, 0.8, 0.9]
        outcomes = [False, False, True, True]
        
        result = compute_ece(preds, outcomes, n_bins=5)
        assert "bin_data" in result
        assert len(result["bin_data"]) == 5


class TestCoherence:
    """Tests for coherence checking."""
    
    def test_valid_distribution_is_coherent(self):
        """Valid distribution should pass coherence check."""
        belief = {b: 1.0/len(BUCKET_NAMES) for b in BUCKET_NAMES}
        
        result = check_coherence(belief)
        assert result["is_coherent"]
        assert result["sum_violation"] < 0.01
    
    def test_sum_violation_detected(self):
        """Should detect when probabilities don't sum to 1."""
        belief = {b: 0.1 for b in BUCKET_NAMES}  # Sum = 1.4
        
        result = check_coherence(belief)
        assert not result["is_coherent"]
        assert result["sum_violation"] > 0.01
    
    def test_negative_mass_detected(self):
        """Should detect negative probabilities."""
        belief = {BUCKET_NAMES[0]: -0.1}
        for b in BUCKET_NAMES[1:]:
            belief[b] = 1.1 / (len(BUCKET_NAMES) - 1)
        
        result = check_coherence(belief)
        assert result["negative_mass"] > 0
    
    def test_coherence_summary(self):
        """Should compute aggregate statistics."""
        beliefs = [
            {b: 1.0/len(BUCKET_NAMES) for b in BUCKET_NAMES},  # Valid
            {b: 0.1 for b in BUCKET_NAMES},  # Invalid
        ]
        
        summary = compute_coherence_summary(beliefs)
        assert summary["n_beliefs"] == 2
        assert summary["coherence_rate"] == 0.5


class TestUpdateCoherence:
    """Tests for update coherence metrics."""
    
    def test_no_update_gives_zero_delta(self):
        """Same beliefs should have zero delta."""
        belief = {"a": 0.5, "b": 0.5}
        delta = compute_belief_delta(belief, belief, ["a", "b"])
        
        assert all(abs(v) < 1e-6 for v in delta.values())
    
    def test_update_agreement_perfect(self):
        """Perfect agreement should give correlation 1."""
        # LLM and oracle both update in same direction
        llm_seq = [
            {"a": 0.5, "b": 0.5},
            {"a": 0.7, "b": 0.3},
        ]
        oracle_seq = [
            {"a": 0.5, "b": 0.5},
            {"a": 0.8, "b": 0.2},
        ]
        
        result = compute_update_agreement(llm_seq, oracle_seq, ["a", "b"])
        assert result["avg_direction_agreement"] > 0.9


class TestBeliefAction:
    """Tests for belief-action metrics."""
    
    def test_divergence_same_belief(self):
        """Same beliefs should have zero divergence."""
        belief = {"a": 0.5, "b": 0.5}
        div = compute_belief_action_divergence(belief, belief)
        assert div < 1e-6
    
    def test_regret_optimal_action(self):
        """Optimal action should have zero regret."""
        oracle = {"bucket1": 1.0, "bucket2": 0.0}
        q_values = {
            "bucket1": {
                ActionType.FOLD: -10,
                ActionType.CHECK_CALL: 5,
                ActionType.BET_RAISE: 10,
            },
            "bucket2": {
                ActionType.FOLD: 0,
                ActionType.CHECK_CALL: 0,
                ActionType.BET_RAISE: 0,
            },
        }
        
        # BET_RAISE is optimal under oracle belief
        regret = compute_decision_regret(ActionType.BET_RAISE, oracle, q_values)
        assert regret < 1e-6
    
    def test_regret_suboptimal_action(self):
        """Suboptimal action should have positive regret."""
        oracle = {"bucket1": 1.0, "bucket2": 0.0}
        q_values = {
            "bucket1": {
                ActionType.FOLD: -10,
                ActionType.CHECK_CALL: 5,
                ActionType.BET_RAISE: 10,
            },
            "bucket2": {
                ActionType.FOLD: 0,
                ActionType.CHECK_CALL: 0,
                ActionType.BET_RAISE: 0,
            },
        }
        
        # FOLD is suboptimal
        regret = compute_decision_regret(ActionType.FOLD, oracle, q_values)
        assert regret > 0
