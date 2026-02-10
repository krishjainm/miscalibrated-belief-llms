"""
Belief schema for structured LLM belief elicitation.

Defines the BeliefOutput dataclass for capturing beliefs elicited from LLMs,
along with validation and coherence checking utilities.
"""

from dataclasses import dataclass, field
from typing import Any
import json

from analysis.buckets import BUCKET_NAMES


@dataclass
class BeliefOutput:
    """
    Structured belief elicited from an LLM agent.
    
    Captures the agent's probability distribution over opponent hand buckets,
    along with optional event probabilities and metadata for analysis.
    """
    
    # Core belief: distribution over buckets (should sum to 1)
    bucket_probs: dict[str, float] = field(default_factory=dict)
    
    # Optional event probabilities for coherence testing
    # e.g., {"opponent_has_pair": 0.3, "opponent_has_ace": 0.4}
    event_probs: dict[str, float] | None = None
    
    # Raw LLM response for debugging
    raw_response: str = ""
    
    # Parsing metadata
    parse_success: bool = False
    parse_error: str | None = None
    
    # Coherence status (set by check_coherence)
    is_coherent: bool = False
    coherence_issues: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "bucket_probs": self.bucket_probs,
            "event_probs": self.event_probs,
            "raw_response": self.raw_response,
            "parse_success": self.parse_success,
            "parse_error": self.parse_error,
            "is_coherent": self.is_coherent,
            "coherence_issues": self.coherence_issues,
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: dict) -> "BeliefOutput":
        """Deserialize from dictionary."""
        return cls(
            bucket_probs=data.get("bucket_probs", {}),
            event_probs=data.get("event_probs"),
            raw_response=data.get("raw_response", ""),
            parse_success=data.get("parse_success", False),
            parse_error=data.get("parse_error"),
            is_coherent=data.get("is_coherent", False),
            coherence_issues=data.get("coherence_issues", []),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "BeliefOutput":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


def check_belief_coherence(belief: BeliefOutput) -> dict:
    """
    Check probability axiom violations in a belief.
    
    Args:
        belief: BeliefOutput to check
        
    Returns:
        Dict with coherence metrics and issues
    """
    issues = []
    metrics = {}
    
    probs = belief.bucket_probs
    
    # Check 1: All buckets present
    missing_buckets = [b for b in BUCKET_NAMES if b not in probs]
    if missing_buckets:
        issues.append(f"Missing buckets: {missing_buckets}")
    metrics["missing_buckets"] = len(missing_buckets)
    
    # Check 2: Non-negative probabilities
    negative_probs = {k: v for k, v in probs.items() if v < 0}
    if negative_probs:
        issues.append(f"Negative probabilities: {negative_probs}")
    metrics["negative_mass"] = sum(abs(v) for v in negative_probs.values())
    
    # Check 3: Sum to 1
    prob_sum = sum(probs.values())
    sum_deviation = abs(prob_sum - 1.0)
    if sum_deviation > 0.01:  # Tolerance of 1%
        issues.append(f"Probabilities sum to {prob_sum:.4f}, not 1.0")
    metrics["sum_violation"] = sum_deviation
    
    # Check 4: No NaN or Inf
    invalid_values = {k: v for k, v in probs.items() 
                      if not isinstance(v, (int, float)) or 
                      (isinstance(v, float) and (v != v or abs(v) == float('inf')))}
    if invalid_values:
        issues.append(f"Invalid values: {invalid_values}")
    metrics["invalid_values"] = len(invalid_values)
    
    # Update belief object
    belief.is_coherent = len(issues) == 0
    belief.coherence_issues = issues
    
    return {
        "is_coherent": len(issues) == 0,
        "issues": issues,
        "metrics": metrics,
    }


def parse_belief_from_text(
    text: str,
    bucket_names: list[str] | None = None,
) -> BeliefOutput:
    """
    Parse a belief distribution from LLM text output.
    
    Attempts to extract bucket probabilities from various formats:
    - JSON format: {"bucket": prob, ...}
    - List format: bucket: prob
    - Inline format: bucket (prob), bucket (prob), ...
    
    Args:
        text: Raw LLM response text
        bucket_names: Expected bucket names (defaults to BUCKET_NAMES)
        
    Returns:
        BeliefOutput with parsed probabilities or parse error
    """
    if bucket_names is None:
        bucket_names = BUCKET_NAMES
    
    belief = BeliefOutput(raw_response=text)
    
    # Try JSON parsing first
    try:
        # Look for JSON object in the text
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            data = json.loads(json_str)
            
            # Check if it's our expected format
            if isinstance(data, dict):
                # Extract bucket probabilities
                for bucket in bucket_names:
                    if bucket in data:
                        try:
                            belief.bucket_probs[bucket] = float(data[bucket])
                        except (ValueError, TypeError):
                            pass
                
                if belief.bucket_probs:
                    belief.parse_success = True
                    return belief
    except json.JSONDecodeError:
        pass
    
    # Try line-by-line parsing
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Try "bucket: prob" or "bucket = prob" format
        for bucket in bucket_names:
            if bucket in line.lower():
                # Extract number after the bucket name
                parts = line.split(bucket, 1)
                if len(parts) > 1:
                    remainder = parts[1]
                    # Look for a number
                    import re
                    numbers = re.findall(r'[\d.]+', remainder)
                    if numbers:
                        try:
                            prob = float(numbers[0])
                            # Check if it might be a percentage
                            if prob > 1.0 and prob <= 100:
                                prob /= 100.0
                            belief.bucket_probs[bucket] = prob
                        except ValueError:
                            pass
    
    if belief.bucket_probs:
        belief.parse_success = True
    else:
        belief.parse_error = "Could not parse any bucket probabilities from response"
    
    return belief
