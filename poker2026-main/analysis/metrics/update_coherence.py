"""
Update coherence metrics for belief dynamics analysis.

Measures how well LLM beliefs update in response to new information,
compared to Bayesian oracle updates.

Key metrics:
- Update agreement: correlation between LLM and oracle belief changes
- Monotonicity violations: wrong-direction updates
- Update magnitude ratio: LLM update size vs oracle update size
"""

import numpy as np
from typing import Sequence
from scipy import stats

from analysis.buckets import BUCKET_NAMES
from analysis.metrics.calibration import compute_kl_divergence, compute_l2_distance


def compute_belief_delta(
    belief_before: dict[str, float],
    belief_after: dict[str, float],
    bucket_names: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute change in belief from before to after.
    
    Args:
        belief_before: Belief distribution before update
        belief_after: Belief distribution after update
        bucket_names: Bucket names for consistent ordering
        
    Returns:
        Dict mapping bucket to probability change
    """
    if bucket_names is None:
        bucket_names = BUCKET_NAMES
    
    return {
        b: belief_after.get(b, 0.0) - belief_before.get(b, 0.0)
        for b in bucket_names
    }


def compute_update_magnitude(
    belief_before: dict[str, float],
    belief_after: dict[str, float],
    method: str = "l2",
) -> float:
    """
    Compute magnitude of belief update.
    
    Args:
        belief_before: Belief before
        belief_after: Belief after
        method: "l2", "l1", or "kl"
        
    Returns:
        Update magnitude
    """
    if method == "l2":
        return compute_l2_distance(belief_before, belief_after)
    elif method == "l1":
        keys = set(belief_before.keys()) | set(belief_after.keys())
        return sum(abs(belief_before.get(k, 0) - belief_after.get(k, 0)) for k in keys)
    elif method == "kl":
        return compute_kl_divergence(belief_before, belief_after)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_update_agreement(
    llm_beliefs_over_time: Sequence[dict[str, float]],
    oracle_beliefs_over_time: Sequence[dict[str, float]],
    bucket_names: list[str] | None = None,
) -> dict:
    """
    Compare how LLM updates beliefs vs how oracle updates.
    
    Measures whether LLM belief changes are correlated with oracle
    belief changes - i.e., do they update in the same direction?
    
    Args:
        llm_beliefs_over_time: Sequence of LLM beliefs
        oracle_beliefs_over_time: Sequence of oracle beliefs
        bucket_names: Bucket names
        
    Returns:
        Dict with update agreement metrics
    """
    if bucket_names is None:
        bucket_names = BUCKET_NAMES
    
    if len(llm_beliefs_over_time) != len(oracle_beliefs_over_time):
        raise ValueError("Belief sequences must have same length")
    
    if len(llm_beliefs_over_time) < 2:
        return {
            "update_correlation": None,
            "avg_direction_agreement": None,
            "update_magnitude_ratio": None,
            "n_updates": 0,
        }
    
    # Compute deltas for each update
    llm_deltas = []
    oracle_deltas = []
    llm_magnitudes = []
    oracle_magnitudes = []
    direction_agreements = []
    
    for i in range(1, len(llm_beliefs_over_time)):
        llm_delta = compute_belief_delta(
            llm_beliefs_over_time[i-1], llm_beliefs_over_time[i], bucket_names
        )
        oracle_delta = compute_belief_delta(
            oracle_beliefs_over_time[i-1], oracle_beliefs_over_time[i], bucket_names
        )
        
        llm_deltas.append(llm_delta)
        oracle_deltas.append(oracle_delta)
        
        # Compute magnitudes
        llm_mag = compute_update_magnitude(
            llm_beliefs_over_time[i-1], llm_beliefs_over_time[i]
        )
        oracle_mag = compute_update_magnitude(
            oracle_beliefs_over_time[i-1], oracle_beliefs_over_time[i]
        )
        llm_magnitudes.append(llm_mag)
        oracle_magnitudes.append(oracle_mag)
        
        # Compute direction agreement per bucket
        agreements = []
        for b in bucket_names:
            llm_d = llm_delta.get(b, 0)
            oracle_d = oracle_delta.get(b, 0)
            # Agreement if both move same direction (or both stay same)
            if (llm_d >= 0 and oracle_d >= 0) or (llm_d <= 0 and oracle_d <= 0):
                agreements.append(1.0)
            else:
                agreements.append(0.0)
        direction_agreements.append(np.mean(agreements))
    
    # Compute correlation across all delta vectors
    llm_flat = []
    oracle_flat = []
    for llm_d, oracle_d in zip(llm_deltas, oracle_deltas):
        for b in bucket_names:
            llm_flat.append(llm_d.get(b, 0))
            oracle_flat.append(oracle_d.get(b, 0))
    
    # Pearson correlation
    if len(llm_flat) > 1 and np.std(llm_flat) > 0 and np.std(oracle_flat) > 0:
        correlation, p_value = stats.pearsonr(llm_flat, oracle_flat)
    else:
        correlation = None
        p_value = None
    
    # Update magnitude ratio
    avg_llm_mag = np.mean(llm_magnitudes) if llm_magnitudes else 0
    avg_oracle_mag = np.mean(oracle_magnitudes) if oracle_magnitudes else 0
    mag_ratio = avg_llm_mag / avg_oracle_mag if avg_oracle_mag > 1e-10 else None
    
    return {
        "update_correlation": float(correlation) if correlation is not None else None,
        "correlation_p_value": float(p_value) if p_value is not None else None,
        "avg_direction_agreement": float(np.mean(direction_agreements)),
        "update_magnitude_ratio": float(mag_ratio) if mag_ratio is not None else None,
        "avg_llm_update_magnitude": float(avg_llm_mag),
        "avg_oracle_update_magnitude": float(avg_oracle_mag),
        "n_updates": len(llm_deltas),
    }


def compute_monotonicity_violations(
    beliefs_over_time: Sequence[dict[str, float]],
    expected_directions: dict[str, str] | None = None,
    bucket_names: list[str] | None = None,
) -> dict:
    """
    Count monotonicity violations in belief updates.
    
    For certain events (e.g., "opponent has pair"), new information
    should only move beliefs in one direction. This checks for
    violations of such monotonicity.
    
    Args:
        beliefs_over_time: Sequence of belief distributions
        expected_directions: Dict mapping bucket to expected direction
            ("increase", "decrease", or "any")
        bucket_names: Bucket names
        
    Returns:
        Dict with violation counts and rates
    """
    if bucket_names is None:
        bucket_names = BUCKET_NAMES
    
    if expected_directions is None:
        expected_directions = {}  # No expected directions = no violations
    
    if len(beliefs_over_time) < 2:
        return {
            "total_violations": 0,
            "violation_rate": 0.0,
            "violations_by_bucket": {},
            "n_updates": 0,
        }
    
    violations_by_bucket = {b: 0 for b in bucket_names}
    total_checks = 0
    
    for i in range(1, len(beliefs_over_time)):
        before = beliefs_over_time[i-1]
        after = beliefs_over_time[i]
        
        for bucket in bucket_names:
            if bucket not in expected_directions:
                continue
            
            expected = expected_directions[bucket]
            delta = after.get(bucket, 0) - before.get(bucket, 0)
            
            total_checks += 1
            
            if expected == "increase" and delta < -1e-6:
                violations_by_bucket[bucket] += 1
            elif expected == "decrease" and delta > 1e-6:
                violations_by_bucket[bucket] += 1
    
    total_violations = sum(violations_by_bucket.values())
    violation_rate = total_violations / total_checks if total_checks > 0 else 0.0
    
    return {
        "total_violations": total_violations,
        "violation_rate": violation_rate,
        "violations_by_bucket": violations_by_bucket,
        "n_updates": len(beliefs_over_time) - 1,
        "n_checks": total_checks,
    }


def compute_update_quality_summary(
    llm_sequences: Sequence[Sequence[dict[str, float]]],
    oracle_sequences: Sequence[Sequence[dict[str, float]]],
) -> dict:
    """
    Compute aggregate update quality over multiple hands.
    
    Args:
        llm_sequences: List of belief sequences (one per hand)
        oracle_sequences: Corresponding oracle sequences
        
    Returns:
        Aggregate update quality metrics
    """
    all_correlations = []
    all_direction_agreements = []
    all_magnitude_ratios = []
    total_updates = 0
    
    for llm_seq, oracle_seq in zip(llm_sequences, oracle_sequences):
        if len(llm_seq) < 2:
            continue
        
        result = compute_update_agreement(llm_seq, oracle_seq)
        
        if result["update_correlation"] is not None:
            all_correlations.append(result["update_correlation"])
        if result["avg_direction_agreement"] is not None:
            all_direction_agreements.append(result["avg_direction_agreement"])
        if result["update_magnitude_ratio"] is not None:
            all_magnitude_ratios.append(result["update_magnitude_ratio"])
        total_updates += result["n_updates"]
    
    return {
        "avg_update_correlation": float(np.mean(all_correlations)) if all_correlations else None,
        "avg_direction_agreement": float(np.mean(all_direction_agreements)) if all_direction_agreements else None,
        "avg_magnitude_ratio": float(np.mean(all_magnitude_ratios)) if all_magnitude_ratios else None,
        "total_updates": total_updates,
        "n_hands_with_updates": len(all_correlations),
    }
