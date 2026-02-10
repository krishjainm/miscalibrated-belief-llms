"""
Coherence metrics for belief evaluation.

Checks probability axiom violations in LLM beliefs:
- Sum violation (probabilities don't sum to 1)
- Negative mass (negative probabilities)
- Missing buckets
- Repair distance (how far from valid distribution)
"""

import numpy as np
from typing import Sequence

from analysis.buckets import BUCKET_NAMES
from analysis.projection import repair_belief


def check_coherence(
    belief: dict[str, float],
    bucket_names: list[str] | None = None,
    tolerance: float = 0.01,
) -> dict:
    """
    Check probability axiom violations in a belief distribution.
    
    Args:
        belief: Belief distribution {bucket: probability}
        bucket_names: Expected bucket names
        tolerance: Acceptable deviation from sum=1
        
    Returns:
        Dict with:
        - is_coherent: bool
        - sum_violation: deviation from 1
        - negative_mass: total negative probability mass
        - missing_buckets: count of missing buckets
        - repair_distance: L2 distance to valid distribution
        - issues: list of issue descriptions
    """
    if bucket_names is None:
        bucket_names = BUCKET_NAMES
    
    issues = []
    
    # Check missing buckets
    missing = [b for b in bucket_names if b not in belief]
    if missing:
        issues.append(f"Missing {len(missing)} buckets")
    
    # Check negative values
    negatives = {k: v for k, v in belief.items() if isinstance(v, (int, float)) and v < 0}
    negative_mass = sum(abs(v) for v in negatives.values())
    if negatives:
        issues.append(f"Negative probabilities in {len(negatives)} buckets (total: {negative_mass:.4f})")
    
    # Check sum
    valid_values = [v for v in belief.values() if isinstance(v, (int, float)) and v == v]
    prob_sum = sum(valid_values)
    sum_violation = abs(prob_sum - 1.0)
    if sum_violation > tolerance:
        issues.append(f"Sum = {prob_sum:.4f} (deviation: {sum_violation:.4f})")
    
    # Check for invalid values (NaN, Inf)
    invalid = {k: v for k, v in belief.items() 
               if not isinstance(v, (int, float)) or 
               (isinstance(v, float) and (v != v or abs(v) == float('inf')))}
    if invalid:
        issues.append(f"Invalid values in {len(invalid)} buckets")
    
    # Compute repair distance
    repaired, repair_info = repair_belief(belief, bucket_names)
    repair_distance = repair_info["repair_distance_l2"]
    
    is_coherent = len(issues) == 0
    
    return {
        "is_coherent": is_coherent,
        "sum_violation": sum_violation,
        "negative_mass": negative_mass,
        "missing_buckets": len(missing),
        "invalid_values": len(invalid),
        "repair_distance": repair_distance,
        "issues": issues,
        "original_sum": prob_sum,
    }


def compute_coherence_summary(
    beliefs: Sequence[dict[str, float]],
    bucket_names: list[str] | None = None,
) -> dict:
    """
    Compute aggregate coherence statistics over multiple beliefs.
    
    Args:
        beliefs: List of belief distributions
        bucket_names: Expected bucket names
        
    Returns:
        Dict with aggregate statistics
    """
    if not beliefs:
        return {
            "n_beliefs": 0,
            "coherence_rate": 0.0,
            "avg_sum_violation": 0.0,
            "avg_negative_mass": 0.0,
            "avg_repair_distance": 0.0,
        }
    
    # Check each belief
    results = [check_coherence(b, bucket_names) for b in beliefs]
    
    n_coherent = sum(1 for r in results if r["is_coherent"])
    coherence_rate = n_coherent / len(beliefs)
    
    avg_sum_violation = float(np.mean([r["sum_violation"] for r in results]))
    avg_negative_mass = float(np.mean([r["negative_mass"] for r in results]))
    avg_repair_distance = float(np.mean([r["repair_distance"] for r in results]))
    
    # Compute percentiles for repair distance
    repair_distances = [r["repair_distance"] for r in results]
    
    return {
        "n_beliefs": len(beliefs),
        "n_coherent": n_coherent,
        "coherence_rate": coherence_rate,
        "avg_sum_violation": avg_sum_violation,
        "max_sum_violation": float(max(r["sum_violation"] for r in results)),
        "avg_negative_mass": avg_negative_mass,
        "max_negative_mass": float(max(r["negative_mass"] for r in results)),
        "avg_repair_distance": avg_repair_distance,
        "median_repair_distance": float(np.median(repair_distances)),
        "p90_repair_distance": float(np.percentile(repair_distances, 90)),
        "max_repair_distance": float(max(repair_distances)),
        "avg_missing_buckets": float(np.mean([r["missing_buckets"] for r in results])),
    }


def compute_axiom_violation_rate(
    beliefs: Sequence[dict[str, float]],
    bucket_names: list[str] | None = None,
) -> dict:
    """
    Compute rate of specific axiom violations.
    
    Args:
        beliefs: List of belief distributions
        bucket_names: Expected bucket names
        
    Returns:
        Dict with violation rates
    """
    if not beliefs:
        return {}
    
    results = [check_coherence(b, bucket_names) for b in beliefs]
    n = len(beliefs)
    
    return {
        "any_violation_rate": 1.0 - sum(1 for r in results if r["is_coherent"]) / n,
        "sum_violation_rate": sum(1 for r in results if r["sum_violation"] > 0.01) / n,
        "negative_prob_rate": sum(1 for r in results if r["negative_mass"] > 0) / n,
        "missing_bucket_rate": sum(1 for r in results if r["missing_buckets"] > 0) / n,
        "invalid_value_rate": sum(1 for r in results if r["invalid_values"] > 0) / n,
    }
