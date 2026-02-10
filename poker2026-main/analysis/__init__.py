"""
Analysis package for LLM belief modeling in poker.

Provides:
- Hand bucket definitions and mapping
- Posterior oracles (card-only and strategy-aware)
- Belief elicitation schemas and projection
- Metrics for calibration, coherence, and belief-action divergence
- Implied belief inference via Monte Carlo Q-values
"""

from analysis.buckets import (
    BUCKETS,
    BUCKET_NAMES,
    hand_to_bucket,
    get_valid_hands_for_bucket,
    get_all_hands_in_bucket,
    get_bucket_prior,
)
from analysis.opponent_model import (
    OpponentModel,
    ParametricOpponent,
    CFROpponent,
    PublicState,
    ActionType,
)
from analysis.posterior_oracle import (
    CardOnlyPosterior,
    StrategyAwarePosterior,
    extract_opponent_actions,
)
from analysis.belief_schema import (
    BeliefOutput,
    check_belief_coherence,
    parse_belief_from_text,
)
from analysis.projection import (
    project_to_simplex,
    repair_belief,
)

__all__ = [
    # Buckets
    "BUCKETS",
    "BUCKET_NAMES",
    "hand_to_bucket",
    "get_valid_hands_for_bucket",
    "get_all_hands_in_bucket",
    "get_bucket_prior",
    # Opponent models
    "OpponentModel",
    "ParametricOpponent",
    "CFROpponent",
    "PublicState",
    "ActionType",
    # Posterior oracles
    "CardOnlyPosterior",
    "StrategyAwarePosterior",
    "extract_opponent_actions",
    # Belief schema
    "BeliefOutput",
    "check_belief_coherence",
    "parse_belief_from_text",
    # Projection
    "project_to_simplex",
    "repair_belief",
]
