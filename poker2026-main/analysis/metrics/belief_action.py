"""
Belief-action divergence and decision regret metrics.

Measures the gap between:
- What an agent says it believes (stated belief)
- What belief its action implies (implied belief)
- What the oracle believes (ground truth)

Key metrics:
- Belief-action divergence: distance between stated and implied beliefs
- Decision regret: EV loss from action vs optimal action under oracle belief
"""

import numpy as np
from typing import Sequence, Optional

from analysis.buckets import BUCKET_NAMES
from analysis.opponent_model import ActionType
from analysis.metrics.calibration import compute_kl_divergence, compute_js_divergence, compute_l2_distance
from analysis.implied_belief.inverse import compute_expected_q, compute_action_optimality_margin


def compute_belief_action_divergence(
    stated_belief: dict[str, float],
    implied_belief: dict[str, float],
    method: str = "kl",
) -> float:
    """
    Compute divergence between stated and action-implied beliefs.
    
    A large divergence indicates the agent's actions don't match
    what it claims to believe - a "knowing vs doing" gap.
    
    Args:
        stated_belief: What the agent said it believes
        implied_belief: What belief its action implies
        method: Divergence method ("kl", "js", "l2")
        
    Returns:
        Divergence value
    """
    if method == "kl":
        return compute_kl_divergence(stated_belief, implied_belief)
    elif method == "js":
        return compute_js_divergence(stated_belief, implied_belief)
    elif method == "l2":
        return compute_l2_distance(stated_belief, implied_belief)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_decision_regret(
    chosen_action: ActionType,
    oracle_belief: dict[str, float],
    q_values: dict[str, dict[ActionType, float]],
) -> float:
    """
    Compute regret of chosen action under oracle belief.
    
    Regret = max_a E_h~oracle[Q(a; h)] - E_h~oracle[Q(a_chosen; h)]
    
    This measures how much EV was lost by not taking the optimal
    action according to the oracle posterior.
    
    Args:
        chosen_action: The action that was taken
        oracle_belief: Oracle's posterior distribution
        q_values: Q-values by [bucket][action]
        
    Returns:
        Regret (non-negative, 0 if action was optimal)
    """
    # Compute expected Q for each action under oracle belief
    expected_qs = {
        action: compute_expected_q(oracle_belief, q_values, action)
        for action in ActionType
    }
    
    max_eq = max(expected_qs.values())
    chosen_eq = expected_qs.get(chosen_action, 0.0)
    
    return max(0.0, max_eq - chosen_eq)


def compute_stated_belief_regret(
    stated_belief: dict[str, float],
    chosen_action: ActionType,
    q_values: dict[str, dict[ActionType, float]],
) -> float:
    """
    Compute regret under stated belief.
    
    If the agent's action doesn't maximize EQ under its stated belief,
    this reveals internal inconsistency.
    
    Args:
        stated_belief: Agent's stated belief
        chosen_action: Action agent took
        q_values: Q-values
        
    Returns:
        Regret under stated belief
    """
    return -compute_action_optimality_margin(chosen_action, stated_belief, q_values)


def compute_belief_action_summary(
    stated_beliefs: Sequence[dict[str, float]],
    implied_beliefs: Sequence[dict[str, float]],
    oracle_beliefs: Sequence[dict[str, float]],
    chosen_actions: Sequence[ActionType],
    q_values_list: Sequence[dict[str, dict[ActionType, float]]],
) -> dict:
    """
    Compute aggregate belief-action metrics over multiple decisions.
    
    Args:
        stated_beliefs: Agent's stated beliefs at each decision
        implied_beliefs: Implied beliefs from actions
        oracle_beliefs: Oracle posteriors
        chosen_actions: Actions taken
        q_values_list: Q-values at each decision
        
    Returns:
        Dict with aggregate metrics
    """
    n = len(stated_beliefs)
    if n == 0:
        return {
            "n_decisions": 0,
            "avg_stated_implied_divergence": None,
            "avg_stated_oracle_divergence": None,
            "avg_decision_regret": None,
        }
    
    # Compute per-decision metrics
    stated_implied_divs = []
    stated_oracle_divs = []
    regrets = []
    internal_regrets = []
    
    for i in range(n):
        # Stated vs implied divergence
        div_si = compute_belief_action_divergence(
            stated_beliefs[i], implied_beliefs[i], method="js"
        )
        stated_implied_divs.append(div_si)
        
        # Stated vs oracle divergence
        div_so = compute_belief_action_divergence(
            stated_beliefs[i], oracle_beliefs[i], method="js"
        )
        stated_oracle_divs.append(div_so)
        
        # Decision regret under oracle
        regret = compute_decision_regret(
            chosen_actions[i], oracle_beliefs[i], q_values_list[i]
        )
        regrets.append(regret)
        
        # Internal consistency: regret under stated belief
        int_regret = compute_stated_belief_regret(
            stated_beliefs[i], chosen_actions[i], q_values_list[i]
        )
        internal_regrets.append(int_regret)
    
    return {
        "n_decisions": n,
        # Stated vs implied
        "avg_stated_implied_divergence": float(np.mean(stated_implied_divs)),
        "max_stated_implied_divergence": float(max(stated_implied_divs)),
        # Stated vs oracle
        "avg_stated_oracle_divergence": float(np.mean(stated_oracle_divs)),
        "max_stated_oracle_divergence": float(max(stated_oracle_divs)),
        # Decision quality
        "avg_decision_regret": float(np.mean(regrets)),
        "max_decision_regret": float(max(regrets)),
        "zero_regret_rate": sum(1 for r in regrets if r < 0.01) / n,
        # Internal consistency
        "avg_internal_regret": float(np.mean(internal_regrets)),
        "consistent_rate": sum(1 for r in internal_regrets if r < 0.01) / n,
    }


def identify_plays_well_despite_bad_beliefs(
    stated_beliefs: Sequence[dict[str, float]],
    oracle_beliefs: Sequence[dict[str, float]],
    chosen_actions: Sequence[ActionType],
    q_values_list: Sequence[dict[str, dict[ActionType, float]]],
    belief_threshold: float = 0.3,  # JS divergence threshold for "bad beliefs"
    regret_threshold: float = 0.5,  # Regret threshold for "plays well"
) -> dict:
    """
    Identify decisions where agent plays well despite bad beliefs.
    
    This is the key "plays well anyway" phenomenon that challenges
    assumptions about how LLMs reason.
    
    Args:
        stated_beliefs: Agent's stated beliefs
        oracle_beliefs: Oracle posteriors
        chosen_actions: Actions taken
        q_values_list: Q-values
        belief_threshold: Divergence above which beliefs are "bad"
        regret_threshold: Regret below which performance is "good"
        
    Returns:
        Dict with analysis of the phenomenon
    """
    n = len(stated_beliefs)
    if n == 0:
        return {"n_decisions": 0}
    
    # Categorize each decision
    bad_beliefs_good_play = 0
    bad_beliefs_bad_play = 0
    good_beliefs_good_play = 0
    good_beliefs_bad_play = 0
    
    for i in range(n):
        # Compute belief quality
        belief_div = compute_js_divergence(stated_beliefs[i], oracle_beliefs[i])
        is_bad_belief = belief_div > belief_threshold
        
        # Compute play quality
        regret = compute_decision_regret(
            chosen_actions[i], oracle_beliefs[i], q_values_list[i]
        )
        is_good_play = regret < regret_threshold
        
        # Categorize
        if is_bad_belief and is_good_play:
            bad_beliefs_good_play += 1
        elif is_bad_belief and not is_good_play:
            bad_beliefs_bad_play += 1
        elif not is_bad_belief and is_good_play:
            good_beliefs_good_play += 1
        else:
            good_beliefs_bad_play += 1
    
    return {
        "n_decisions": n,
        "bad_beliefs_good_play": bad_beliefs_good_play,
        "bad_beliefs_bad_play": bad_beliefs_bad_play,
        "good_beliefs_good_play": good_beliefs_good_play,
        "good_beliefs_bad_play": good_beliefs_bad_play,
        # Key rates
        "plays_well_despite_bad_beliefs_rate": bad_beliefs_good_play / n if n > 0 else 0,
        "bad_beliefs_rate": (bad_beliefs_good_play + bad_beliefs_bad_play) / n if n > 0 else 0,
        "good_play_rate": (bad_beliefs_good_play + good_beliefs_good_play) / n if n > 0 else 0,
        # Conditional rates
        "good_play_given_bad_beliefs": (
            bad_beliefs_good_play / (bad_beliefs_good_play + bad_beliefs_bad_play)
            if (bad_beliefs_good_play + bad_beliefs_bad_play) > 0 else None
        ),
        "good_play_given_good_beliefs": (
            good_beliefs_good_play / (good_beliefs_good_play + good_beliefs_bad_play)
            if (good_beliefs_good_play + good_beliefs_bad_play) > 0 else None
        ),
        "thresholds": {
            "belief_threshold": belief_threshold,
            "regret_threshold": regret_threshold,
        },
    }
