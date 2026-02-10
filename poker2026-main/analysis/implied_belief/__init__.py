"""
Implied belief inference via Q-value estimation.

Provides:
- MonteCarloQValue: Estimate Q(action; opponent_hand) via rollouts
- Inverse decision rule: Infer belief that makes action optimal
"""

from analysis.implied_belief.q_value import MonteCarloQValue
from analysis.implied_belief.inverse import (
    infer_implied_belief,
    find_optimal_action_beliefs,
)

__all__ = [
    "MonteCarloQValue",
    "infer_implied_belief",
    "find_optimal_action_beliefs",
]
