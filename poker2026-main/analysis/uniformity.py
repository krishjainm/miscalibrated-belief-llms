"""
Prompt uniformity testing for belief elicitation.

Measures whether LLM beliefs are stable across equivalent elicitation formats.
If beliefs vary wildly with prompt phrasing, we're measuring the prompt, not the model.

Key metrics:
- Intra-state variance: How much do beliefs vary for the same game state?
- JS stddev: Standard deviation of JS divergence across prompt variants
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

from analysis.buckets import BUCKET_NAMES
from analysis.metrics.calibration import compute_js_divergence


@dataclass
class PromptVariant:
    """A prompt variant for uniformity testing."""
    template_id: str
    template_name: str
    prompt_text: str
    prompt_hash: str = field(init=False)
    
    def __post_init__(self):
        self.prompt_hash = hashlib.md5(self.prompt_text.encode()).hexdigest()[:12]


@dataclass 
class UniformityResult:
    """Results from uniformity testing on a single game state."""
    state_id: str
    n_variants: int
    beliefs: list[dict[str, float]]
    template_ids: list[str]
    
    # Computed metrics
    mean_belief: dict[str, float] = field(default_factory=dict)
    belief_stddev: dict[str, float] = field(default_factory=dict)
    pairwise_js: list[float] = field(default_factory=list)
    js_mean: float = 0.0
    js_stddev: float = 0.0
    max_js: float = 0.0
    
    def compute_metrics(self):
        """Compute uniformity metrics from collected beliefs."""
        if not self.beliefs:
            return
        
        # Mean belief per bucket
        for bucket in BUCKET_NAMES:
            values = [b.get(bucket, 0.0) for b in self.beliefs]
            self.mean_belief[bucket] = float(np.mean(values))
            self.belief_stddev[bucket] = float(np.std(values))
        
        # Pairwise JS divergences
        self.pairwise_js = []
        for i in range(len(self.beliefs)):
            for j in range(i + 1, len(self.beliefs)):
                js = compute_js_divergence(self.beliefs[i], self.beliefs[j])
                self.pairwise_js.append(js)
        
        if self.pairwise_js:
            self.js_mean = float(np.mean(self.pairwise_js))
            self.js_stddev = float(np.std(self.pairwise_js))
            self.max_js = float(max(self.pairwise_js))


def generate_prompt_variants(
    hero_hole: list[str],
    board: list[str],
    pot: int,
    street: str,
    history: list[dict],
) -> list[PromptVariant]:
    """
    Generate multiple prompt variants for the same game state.
    
    Args:
        hero_hole: Hero's hole cards
        board: Board cards
        pot: Pot size
        street: Current street
        history: Action history
        
    Returns:
        List of prompt variants
    """
    from analysis.prompts import (
        format_belief_prompt,
        get_bucket_list_string,
        format_action_history,
    )
    
    variants = []
    
    # Variant 1: Default template
    variants.append(PromptVariant(
        template_id="default",
        template_name="Standard structured prompt",
        prompt_text=format_belief_prompt(
            hero_hole, board, pot, street, history, template="default"
        ),
    ))
    
    # Variant 2: Simple template
    variants.append(PromptVariant(
        template_id="simple",
        template_name="Minimal prompt",
        prompt_text=format_belief_prompt(
            hero_hole, board, pot, street, history, template="simple"
        ),
    ))
    
    # Variant 3: Reasoning template
    variants.append(PromptVariant(
        template_id="reasoning",
        template_name="Chain-of-thought prompt",
        prompt_text=format_belief_prompt(
            hero_hole, board, pot, street, history, template="reasoning"
        ),
    ))
    
    # Variant 4: Direct question (custom)
    bucket_list = get_bucket_list_string()
    action_str = format_action_history(history)
    direct_prompt = f"""What hands might your opponent have?

Cards: {' '.join(hero_hole)}
Board: {' '.join(board) if board else 'None'}
Pot: {pot}
Actions: {action_str}

Give probability for each category (sum to 1):
{bucket_list}

Output JSON only."""
    
    variants.append(PromptVariant(
        template_id="direct",
        template_name="Direct question format",
        prompt_text=direct_prompt,
    ))
    
    # Variant 5: Bayesian framing
    bayesian_prompt = f"""You are computing a Bayesian posterior over opponent hands.

Prior: Uniform over all hands not blocked by visible cards.
Evidence: Opponent's betting actions.

Your hand: {' '.join(hero_hole)}
Board: {' '.join(board) if board else 'Preflop'}
Pot: {pot}
Opponent actions: {action_str}

Update your prior based on the likelihood of each hand type taking those actions.
Output posterior probabilities (must sum to 1.0):

{bucket_list}

JSON format: {{"bucket_name": probability, ...}}"""
    
    variants.append(PromptVariant(
        template_id="bayesian",
        template_name="Bayesian framing",
        prompt_text=bayesian_prompt,
    ))
    
    return variants


def test_uniformity(
    game_state: dict,
    llm_query_fn: Callable[[str], dict[str, float]],
    state_id: str = "test",
) -> UniformityResult:
    """
    Test belief uniformity across prompt variants.
    
    Args:
        game_state: Dict with hero_hole, board, pot, street, history
        llm_query_fn: Function that takes prompt text and returns belief dict
        state_id: Identifier for this game state
        
    Returns:
        UniformityResult with computed metrics
    """
    variants = generate_prompt_variants(
        hero_hole=game_state["hero_hole"],
        board=game_state.get("board", []),
        pot=game_state.get("pot", 0),
        street=game_state.get("street", "PREFLOP"),
        history=game_state.get("history", []),
    )
    
    beliefs = []
    template_ids = []
    
    for variant in variants:
        belief = llm_query_fn(variant.prompt_text)
        beliefs.append(belief)
        template_ids.append(variant.template_id)
    
    result = UniformityResult(
        state_id=state_id,
        n_variants=len(variants),
        beliefs=beliefs,
        template_ids=template_ids,
    )
    result.compute_metrics()
    
    return result


def compute_uniformity_summary(results: list[UniformityResult]) -> dict:
    """
    Compute aggregate uniformity statistics over multiple states.
    
    Args:
        results: List of UniformityResult from multiple game states
        
    Returns:
        Summary statistics
    """
    if not results:
        return {"n_states": 0}
    
    js_means = [r.js_mean for r in results]
    js_stddevs = [r.js_stddev for r in results]
    max_js_values = [r.max_js for r in results]
    
    # Per-bucket stddev aggregation
    bucket_stddevs = {bucket: [] for bucket in BUCKET_NAMES}
    for r in results:
        for bucket, std in r.belief_stddev.items():
            bucket_stddevs[bucket].append(std)
    
    avg_bucket_stddev = {
        bucket: float(np.mean(stds)) if stds else 0.0
        for bucket, stds in bucket_stddevs.items()
    }
    
    return {
        "n_states": len(results),
        "avg_js_mean": float(np.mean(js_means)),
        "avg_js_stddev": float(np.mean(js_stddevs)),
        "avg_max_js": float(np.mean(max_js_values)),
        "worst_max_js": float(max(max_js_values)),
        "avg_bucket_stddev": avg_bucket_stddev,
        "overall_bucket_stddev": float(np.mean(list(avg_bucket_stddev.values()))),
        # Interpretation thresholds
        "uniformity_good": float(np.mean(js_means)) < 0.1,  # Low variance
        "uniformity_acceptable": float(np.mean(js_means)) < 0.2,
    }


@dataclass
class UniformityTestConfig:
    """Configuration for uniformity testing."""
    n_states: int = 10
    variants_per_state: int = 5
    seed: int = 42
    
    def to_dict(self) -> dict:
        return {
            "n_states": self.n_states,
            "variants_per_state": self.variants_per_state,
            "seed": self.seed,
        }
