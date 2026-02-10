"""
Inverse decision rule for inferring action-implied beliefs.

Given an action and Q-values, infer what belief distribution would
make that action (near-)optimal. This reveals the "implicit belief"
behind an agent's action choice.
"""

import numpy as np
from scipy import optimize
from typing import Optional

from analysis.buckets import BUCKET_NAMES
from analysis.opponent_model import ActionType
from analysis.projection import project_to_simplex


def compute_expected_q(
    belief: dict[str, float],
    q_values: dict[str, dict[ActionType, float]],
    action: ActionType,
) -> float:
    """
    Compute expected Q-value under a belief distribution.
    
    E_h~b[Q(a; h)] = Σ_bucket belief[bucket] * Q[bucket][action]
    
    Args:
        belief: Belief distribution over buckets
        q_values: Q-values indexed by [bucket][action]
        action: Action to evaluate
        
    Returns:
        Expected Q-value
    """
    eq = 0.0
    for bucket, prob in belief.items():
        if bucket in q_values and action in q_values[bucket]:
            eq += prob * q_values[bucket][action]
    return eq


def compute_softmax_action_prob(
    belief: dict[str, float],
    q_values: dict[str, dict[ActionType, float]],
    action: ActionType,
    temperature: float = 1.0,
) -> float:
    """
    Compute P(action | belief) under softmax decision model.
    
    P(a | b) ∝ exp(β × E_h~b[Q(a; h)])
    
    Args:
        belief: Belief distribution
        q_values: Q-values by bucket and action
        action: Action to compute probability for
        temperature: Softmax temperature (β = 1/temperature)
        
    Returns:
        Action probability
    """
    actions = list(ActionType)
    
    # Compute expected Q for each action
    expected_qs = {
        a: compute_expected_q(belief, q_values, a)
        for a in actions
    }
    
    # Softmax
    beta = 1.0 / temperature
    max_q = max(expected_qs.values())
    exp_values = {a: np.exp(beta * (q - max_q)) for a, q in expected_qs.items()}
    total = sum(exp_values.values())
    
    if total <= 0:
        return 1.0 / len(actions)
    
    return exp_values[action] / total


def infer_implied_belief(
    chosen_action: ActionType,
    q_values: dict[str, dict[ActionType, float]],
    method: str = "softmax",
    temperature: float = 1.0,
    bucket_names: list[str] | None = None,
) -> dict[str, float]:
    """
    Infer belief that makes chosen_action (near-)optimal.
    
    Two methods:
    1. "softmax": Find belief that maximizes P(chosen_action | belief)
    2. "optimal_set": Find belief where chosen_action has highest Q
    
    Args:
        chosen_action: The action that was taken
        q_values: Q-values by [bucket][action]
        method: "softmax" or "optimal_set"
        temperature: Softmax temperature
        bucket_names: Bucket names
        
    Returns:
        Implied belief distribution
    """
    if bucket_names is None:
        bucket_names = BUCKET_NAMES
    
    # Filter to buckets that have Q-values
    valid_buckets = [b for b in bucket_names if b in q_values]
    
    if not valid_buckets:
        # No valid buckets, return uniform
        return {b: 1.0 / len(bucket_names) for b in bucket_names}
    
    if method == "optimal_set":
        return _infer_optimal_set_belief(chosen_action, q_values, valid_buckets)
    else:  # softmax
        return _infer_softmax_belief(chosen_action, q_values, valid_buckets, temperature)


def _infer_softmax_belief(
    chosen_action: ActionType,
    q_values: dict[str, dict[ActionType, float]],
    buckets: list[str],
    temperature: float,
) -> dict[str, float]:
    """
    Find belief that maximizes P(chosen_action | belief) under softmax.
    
    Uses optimization to find the belief distribution on the simplex
    that makes the chosen action most likely.
    """
    n = len(buckets)
    
    def neg_log_prob(x):
        """Negative log probability to minimize."""
        # Convert to belief dict
        belief = {b: max(x[i], 1e-10) for i, b in enumerate(buckets)}
        # Normalize
        total = sum(belief.values())
        belief = {b: p / total for b, p in belief.items()}
        
        prob = compute_softmax_action_prob(belief, q_values, chosen_action, temperature)
        return -np.log(max(prob, 1e-10))
    
    # Constraints: sum to 1, non-negative
    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x) - 1.0},
    ]
    bounds = [(0.0, 1.0) for _ in range(n)]
    
    # Start with uniform
    x0 = np.ones(n) / n
    
    try:
        result = optimize.minimize(
            neg_log_prob,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100},
        )
        x_opt = result.x
    except Exception:
        x_opt = x0
    
    # Convert to dict and ensure valid distribution
    belief = {b: float(max(x_opt[i], 0)) for i, b in enumerate(buckets)}
    
    # Add missing buckets with 0
    for b in BUCKET_NAMES:
        if b not in belief:
            belief[b] = 0.0
    
    return project_to_simplex(belief)


def _infer_optimal_set_belief(
    chosen_action: ActionType,
    q_values: dict[str, dict[ActionType, float]],
    buckets: list[str],
) -> dict[str, float]:
    """
    Find belief concentrated on buckets where chosen_action is optimal.
    
    Identifies buckets where Q(chosen_action; bucket) >= Q(other_action; bucket)
    and returns uniform belief over those buckets.
    """
    optimal_buckets = []
    
    for bucket in buckets:
        bucket_qs = q_values.get(bucket, {})
        if not bucket_qs:
            continue
        
        chosen_q = bucket_qs.get(chosen_action, float('-inf'))
        max_other_q = max(
            (q for a, q in bucket_qs.items() if a != chosen_action),
            default=float('-inf')
        )
        
        if chosen_q >= max_other_q:
            optimal_buckets.append(bucket)
    
    # If no bucket makes action optimal, fall back to softmax
    if not optimal_buckets:
        return _infer_softmax_belief(chosen_action, q_values, buckets, 1.0)
    
    # Uniform over optimal buckets
    prob = 1.0 / len(optimal_buckets)
    belief = {b: prob if b in optimal_buckets else 0.0 for b in BUCKET_NAMES}
    
    return belief


def find_optimal_action_beliefs(
    q_values: dict[str, dict[ActionType, float]],
    bucket_names: list[str] | None = None,
) -> dict[ActionType, list[str]]:
    """
    Find which buckets make each action optimal.
    
    Args:
        q_values: Q-values by [bucket][action]
        bucket_names: Bucket names
        
    Returns:
        Dict mapping action to list of buckets where it's optimal
    """
    if bucket_names is None:
        bucket_names = BUCKET_NAMES
    
    result = {a: [] for a in ActionType}
    
    for bucket in bucket_names:
        bucket_qs = q_values.get(bucket, {})
        if not bucket_qs:
            continue
        
        # Find action with max Q
        best_action = max(bucket_qs.items(), key=lambda x: x[1])[0]
        result[best_action].append(bucket)
    
    return result


def compute_action_optimality_margin(
    action: ActionType,
    belief: dict[str, float],
    q_values: dict[str, dict[ActionType, float]],
) -> float:
    """
    Compute how much worse/better action is vs optimal action under belief.
    
    Margin = E[Q(action)] - max_a E[Q(a)]
    
    Negative margin means action is suboptimal.
    
    Args:
        action: Action to evaluate
        belief: Belief distribution
        q_values: Q-values
        
    Returns:
        Optimality margin (0 if optimal, negative if suboptimal)
    """
    expected_qs = {
        a: compute_expected_q(belief, q_values, a)
        for a in ActionType
    }
    
    max_eq = max(expected_qs.values())
    action_eq = expected_qs.get(action, 0.0)
    
    return action_eq - max_eq
