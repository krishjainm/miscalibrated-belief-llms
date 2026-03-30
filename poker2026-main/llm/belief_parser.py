"""Parse bucket belief JSON from LLM outputs (direct or CoT)."""

from __future__ import annotations

import json
import re
from typing import Any

from analysis.buckets import BUCKET_NAMES


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Find the first JSON object in text and parse it."""
    if not text or not text.strip():
        return None
    text = text.strip()
    # Strip markdown fences
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if m:
            text = m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    blob = text[start : end + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None


def parse_bucket_belief(
    raw: str,
    *,
    cot_mode: bool = False,
) -> tuple[dict[str, float] | None, str | None]:
    """
    Parse 14-bucket distribution from model output.

    For CoT, looks for PROBABILITIES section first; otherwise parses full text.
    Returns (belief_dict or None, error_message or None).
    """
    text = raw
    if cot_mode and "PROBABILITIES" in raw.upper():
        # Take content after PROBABILITIES (case-insensitive)
        idx = re.search(r"PROBABILITIES\s*:?", raw, re.IGNORECASE)
        if idx:
            text = raw[idx.end() :]

    data = extract_json_object(text)
    if not data:
        return None, "no_json"

    out: dict[str, float] = {}
    for b in BUCKET_NAMES:
        if b in data:
            try:
                out[b] = float(data[b])
            except (TypeError, ValueError):
                return None, f"bad_value:{b}"

    if len(out) != len(BUCKET_NAMES):
        missing = [b for b in BUCKET_NAMES if b not in out]
        return None, f"missing_buckets:{missing}"

    s = sum(out.values())
    if s <= 0:
        return None, "zero_sum"
    # Normalize lightly
    out = {k: v / s for k, v in out.items()}
    return out, None


def repair_nonnegative_l1(belief: dict[str, float]) -> dict[str, float]:
    """Clip negatives to 0 and L1-normalize (paper appendix style)."""
    clipped = {k: max(0.0, float(v)) for k, v in belief.items()}
    s = sum(clipped.values())
    if s <= 0:
        return {b: 1.0 / len(BUCKET_NAMES) for b in BUCKET_NAMES}
    return {k: v / s for k, v in clipped.items()}
