"""
Calibration metrics for belief evaluation.

Provides:
- Posterior Calibration Error (PCE): Average divergence from oracle
- KL and JS divergence between distributions
- ECE and Brier score for event probabilities
"""

import numpy as np
from typing import Sequence

from analysis.buckets import BUCKET_NAMES


# ============================================================================
# Distribution Divergences
# ============================================================================

def compute_kl_divergence(
    p: dict[str, float],
    q: dict[str, float],
    eps: float = 1e-10,
) -> float:
    """
    Compute KL divergence: KL(p || q).
    
    Measures how much information is lost when q is used to approximate p.
    Note: KL(p || q) != KL(q || p).
    
    Args:
        p: "True" distribution (e.g., oracle posterior)
        q: "Approximate" distribution (e.g., LLM belief)
        eps: Smoothing constant to avoid log(0)
        
    Returns:
        KL divergence (non-negative, unbounded above)
    """
    keys = list(set(p.keys()) | set(q.keys()))
    
    kl = 0.0
    for k in keys:
        p_k = max(p.get(k, 0.0), eps)
        q_k = max(q.get(k, 0.0), eps)
        kl += p_k * np.log(p_k / q_k)
    
    return float(max(0.0, kl))


def compute_js_divergence(
    p: dict[str, float],
    q: dict[str, float],
    eps: float = 1e-10,
) -> float:
    """
    Compute Jensen-Shannon divergence.
    
    Symmetric divergence: JS(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)
    where m = 0.5 * (p + q).
    
    Args:
        p: First distribution
        q: Second distribution
        eps: Smoothing constant
        
    Returns:
        JS divergence in [0, ln(2)] ≈ [0, 0.693]
    """
    keys = list(set(p.keys()) | set(q.keys()))
    
    # Compute midpoint distribution
    m = {}
    for k in keys:
        m[k] = 0.5 * (p.get(k, 0.0) + q.get(k, 0.0))
    
    js = 0.5 * compute_kl_divergence(p, m, eps) + 0.5 * compute_kl_divergence(q, m, eps)
    
    return float(js)


def compute_l2_distance(
    p: dict[str, float],
    q: dict[str, float],
) -> float:
    """
    Compute L2 (Euclidean) distance between distributions.
    
    Args:
        p: First distribution
        q: Second distribution
        
    Returns:
        L2 distance
    """
    keys = list(set(p.keys()) | set(q.keys()))
    
    squared_diff = sum((p.get(k, 0.0) - q.get(k, 0.0)) ** 2 for k in keys)
    
    return float(np.sqrt(squared_diff))


def compute_total_variation(
    p: dict[str, float],
    q: dict[str, float],
) -> float:
    """
    Compute total variation distance.
    
    TV(p, q) = 0.5 * sum(|p_i - q_i|)
    
    Args:
        p: First distribution
        q: Second distribution
        
    Returns:
        Total variation in [0, 1]
    """
    keys = list(set(p.keys()) | set(q.keys()))
    
    tv = 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)
    
    return float(tv)


# ============================================================================
# Posterior Calibration Error (PCE)
# ============================================================================

def compute_pce(
    llm_beliefs: Sequence[dict[str, float]],
    oracle_beliefs: Sequence[dict[str, float]],
    method: str = "kl",
) -> dict:
    """
    Compute Posterior Calibration Error.
    
    PCE measures how well LLM beliefs match oracle posteriors across
    multiple decision points.
    
    PCE = (1/N) Σ D(oracle_i || llm_i)
    
    Args:
        llm_beliefs: List of LLM belief distributions
        oracle_beliefs: List of oracle posterior distributions
        method: Divergence method - "kl", "js", "l2", "tv"
        
    Returns:
        Dict with:
        - pce: Average divergence
        - per_sample: List of per-sample divergences
        - std: Standard deviation
        - method: Method used
    """
    if len(llm_beliefs) != len(oracle_beliefs):
        raise ValueError("Number of LLM beliefs must match oracle beliefs")
    
    if len(llm_beliefs) == 0:
        return {"pce": 0.0, "per_sample": [], "std": 0.0, "method": method}
    
    # Select divergence function
    if method == "kl":
        div_fn = compute_kl_divergence
    elif method == "js":
        div_fn = compute_js_divergence
    elif method == "l2":
        div_fn = compute_l2_distance
    elif method == "tv":
        div_fn = compute_total_variation
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute per-sample divergences
    divergences = []
    for llm, oracle in zip(llm_beliefs, oracle_beliefs):
        d = div_fn(oracle, llm)
        divergences.append(d)
    
    pce = float(np.mean(divergences))
    std = float(np.std(divergences))
    
    return {
        "pce": pce,
        "per_sample": divergences,
        "std": std,
        "method": method,
        "n_samples": len(divergences),
    }


# ============================================================================
# Event-Level Calibration (ECE, Brier)
# ============================================================================

def compute_brier_score(
    predicted_probs: Sequence[float],
    outcomes: Sequence[bool],
) -> float:
    """
    Compute Brier score for probabilistic predictions.
    
    Brier = (1/N) Σ (p_i - y_i)²
    
    Lower is better. Range [0, 1].
    
    Args:
        predicted_probs: Predicted probabilities P(event)
        outcomes: Actual outcomes (True if event occurred)
        
    Returns:
        Brier score
    """
    if len(predicted_probs) != len(outcomes):
        raise ValueError("Predictions and outcomes must have same length")
    
    if len(predicted_probs) == 0:
        return 0.0
    
    squared_errors = [
        (p - (1.0 if y else 0.0)) ** 2
        for p, y in zip(predicted_probs, outcomes)
    ]
    
    return float(np.mean(squared_errors))


def compute_ece(
    predicted_probs: Sequence[float],
    outcomes: Sequence[bool],
    n_bins: int = 10,
) -> dict:
    """
    Compute Expected Calibration Error.
    
    ECE measures the difference between predicted confidence and actual accuracy
    across confidence bins.
    
    Args:
        predicted_probs: Predicted probabilities
        outcomes: Actual outcomes
        n_bins: Number of calibration bins
        
    Returns:
        Dict with:
        - ece: Expected calibration error
        - bin_data: Per-bin statistics for reliability diagram
    """
    if len(predicted_probs) != len(outcomes):
        raise ValueError("Predictions and outcomes must have same length")
    
    if len(predicted_probs) == 0:
        return {"ece": 0.0, "bin_data": []}
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    
    predictions = np.array(predicted_probs)
    actuals = np.array([1.0 if o else 0.0 for o in outcomes])
    
    ece = 0.0
    total_samples = len(predictions)
    
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        
        # Find samples in this bin
        if i == n_bins - 1:
            # Include upper boundary for last bin
            mask = (predictions >= lo) & (predictions <= hi)
        else:
            mask = (predictions >= lo) & (predictions < hi)
        
        n_in_bin = np.sum(mask)
        
        if n_in_bin > 0:
            avg_confidence = float(np.mean(predictions[mask]))
            avg_accuracy = float(np.mean(actuals[mask]))
            
            # Weighted contribution to ECE
            ece += (n_in_bin / total_samples) * abs(avg_accuracy - avg_confidence)
            
            bin_data.append({
                "bin_start": float(lo),
                "bin_end": float(hi),
                "n_samples": int(n_in_bin),
                "avg_confidence": avg_confidence,
                "avg_accuracy": avg_accuracy,
                "calibration_error": abs(avg_accuracy - avg_confidence),
            })
        else:
            bin_data.append({
                "bin_start": float(lo),
                "bin_end": float(hi),
                "n_samples": 0,
                "avg_confidence": None,
                "avg_accuracy": None,
                "calibration_error": None,
            })
    
    return {
        "ece": float(ece),
        "bin_data": bin_data,
        "n_samples": total_samples,
        "n_bins": n_bins,
    }


def compute_event_calibration(
    llm_event_probs: Sequence[float],
    actual_outcomes: Sequence[bool],
) -> dict:
    """
    Compute calibration metrics for event probability predictions.
    
    Args:
        llm_event_probs: LLM's predicted P(event) at each decision
        actual_outcomes: Whether the event actually occurred
        
    Returns:
        Dict with ECE, Brier score, and reliability diagram data
    """
    brier = compute_brier_score(llm_event_probs, actual_outcomes)
    ece_result = compute_ece(llm_event_probs, actual_outcomes)
    
    return {
        "brier_score": brier,
        "ece": ece_result["ece"],
        "reliability_diagram": ece_result["bin_data"],
        "n_samples": len(llm_event_probs),
    }
