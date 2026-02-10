"""
Metrics package for belief modeling analysis.

Provides metrics for:
- Posterior Calibration Error (PCE)
- Coherence checks
- Update coherence
- Belief-action divergence
"""

from analysis.metrics.calibration import (
    compute_pce,
    compute_kl_divergence,
    compute_js_divergence,
    compute_event_calibration,
    compute_brier_score,
    compute_ece,
)
from analysis.metrics.coherence import (
    check_coherence,
    compute_coherence_summary,
)
from analysis.metrics.update_coherence import (
    compute_update_agreement,
    compute_monotonicity_violations,
)
from analysis.metrics.belief_action import (
    compute_belief_action_divergence,
    compute_decision_regret,
)

__all__ = [
    # Calibration
    "compute_pce",
    "compute_kl_divergence",
    "compute_js_divergence",
    "compute_event_calibration",
    "compute_brier_score",
    "compute_ece",
    # Coherence
    "check_coherence",
    "compute_coherence_summary",
    # Update coherence
    "compute_update_agreement",
    "compute_monotonicity_violations",
    # Belief-action
    "compute_belief_action_divergence",
    "compute_decision_regret",
]
