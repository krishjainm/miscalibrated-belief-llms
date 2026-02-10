"""
Simplex projection for probability distribution repair.

Projects arbitrary vectors onto the probability simplex to create
valid probability distributions. Used to "repair" incoherent LLM beliefs.
"""

import numpy as np
from typing import Sequence

from analysis.buckets import BUCKET_NAMES


def project_to_simplex_vector(v: np.ndarray) -> np.ndarray:
    """
    Project a vector onto the probability simplex.
    
    Uses the algorithm from:
    "Efficient Projections onto the ℓ1-Ball for Learning in High Dimensions"
    Duchi et al., ICML 2008
    
    Args:
        v: Input vector (any real values)
        
    Returns:
        Projected vector on simplex (non-negative, sums to 1)
    """
    n = len(v)
    if n == 0:
        return v
    
    # Sort in descending order
    u = np.sort(v)[::-1]
    
    # Find the threshold
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    
    # Project
    w = np.maximum(v - theta, 0)
    
    return w


def project_to_simplex(
    probs: dict[str, float],
    bucket_names: list[str] | None = None,
) -> dict[str, float]:
    """
    Project a probability dictionary onto the simplex.
    
    Handles:
    - Negative values
    - Sum != 1
    - Missing buckets (set to 0 before projection)
    
    Args:
        probs: Input probability dict (may be invalid)
        bucket_names: Expected bucket names (defaults to BUCKET_NAMES)
        
    Returns:
        Valid probability distribution (non-negative, sums to 1)
    """
    if bucket_names is None:
        bucket_names = BUCKET_NAMES
    
    # Convert to vector with consistent ordering
    v = np.array([probs.get(b, 0.0) for b in bucket_names], dtype=float)
    
    # Replace NaN/Inf with 0
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check if already valid
    if np.all(v >= 0) and np.abs(np.sum(v) - 1.0) < 1e-10:
        return {b: float(v[i]) for i, b in enumerate(bucket_names)}
    
    # Project to simplex
    w = project_to_simplex_vector(v)
    
    # Convert back to dict
    return {b: float(w[i]) for i, b in enumerate(bucket_names)}


def compute_projection_distance(
    original: dict[str, float],
    projected: dict[str, float],
    method: str = "l2",
) -> float:
    """
    Compute distance between original and projected distributions.
    
    This measures "how far" the LLM's belief was from being coherent.
    
    Args:
        original: Original (possibly incoherent) belief
        projected: Projected (coherent) belief
        method: Distance method - "l2", "l1", or "kl"
        
    Returns:
        Distance value
    """
    buckets = list(set(original.keys()) | set(projected.keys()))
    
    orig_vec = np.array([original.get(b, 0.0) for b in buckets])
    proj_vec = np.array([projected.get(b, 0.0) for b in buckets])
    
    if method == "l2":
        return float(np.sqrt(np.sum((orig_vec - proj_vec) ** 2)))
    elif method == "l1":
        return float(np.sum(np.abs(orig_vec - proj_vec)))
    elif method == "kl":
        # KL divergence (with smoothing to avoid log(0))
        eps = 1e-10
        orig_smooth = np.maximum(orig_vec, eps)
        orig_smooth = orig_smooth / np.sum(orig_smooth)
        proj_smooth = np.maximum(proj_vec, eps)
        return float(np.sum(orig_smooth * np.log(orig_smooth / proj_smooth)))
    else:
        raise ValueError(f"Unknown method: {method}")


def repair_belief(
    probs: dict[str, float],
    bucket_names: list[str] | None = None,
) -> tuple[dict[str, float], dict]:
    """
    Repair an incoherent belief distribution.
    
    Args:
        probs: Input probability dict (may be invalid)
        bucket_names: Expected bucket names
        
    Returns:
        Tuple of (repaired_probs, repair_info)
        repair_info contains:
        - was_valid: bool
        - repair_distance_l2: float
        - original_sum: float
        - negative_mass: float
    """
    if bucket_names is None:
        bucket_names = BUCKET_NAMES
    
    # Compute original stats
    values = [probs.get(b, 0.0) for b in bucket_names]
    original_sum = sum(v for v in values if isinstance(v, (int, float)) and v == v)
    negative_mass = sum(abs(v) for v in values if isinstance(v, (int, float)) and v < 0)
    
    # Check if already valid
    was_valid = (
        all(b in probs for b in bucket_names) and
        all(probs.get(b, 0) >= 0 for b in bucket_names) and
        abs(original_sum - 1.0) < 0.01
    )
    
    # Project
    projected = project_to_simplex(probs, bucket_names)
    
    # Compute repair distance
    repair_distance = compute_projection_distance(probs, projected, "l2")
    
    repair_info = {
        "was_valid": was_valid,
        "repair_distance_l2": repair_distance,
        "original_sum": original_sum,
        "negative_mass": negative_mass,
    }
    
    return projected, repair_info
