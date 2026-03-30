"""
Analyze CoT vs direct belief/action prompting variants.

This script groups decision records by:
  - belief_mode (from llm_extra["belief"]["belief_mode"])
  - action_mode (from llm_extra["action"]["action_mode"])

Then computes:
  - PCE (JS/KL/etc.) vs CardOnly and StrategyAware oracles (if present)
  - belief parse success rate
  - interpretability summaries (when available):
      * local logit lens entropy (if llm_extra["..."]["local_interp"] exists)
      * next-token entropy proxy from API top_logprobs (approx)
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from typing import Any, Optional

import numpy as np

from analysis.buckets import BUCKET_NAMES
from analysis.metrics.calibration import compute_pce, compute_js_divergence


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _softmax(logps: list[float]) -> list[float]:
    m = max(logps)
    exps = [math.exp(lp - m) for lp in logps]
    s = sum(exps)
    if s <= 0:
        return [1.0 / len(exps)] * len(exps)
    return [e / s for e in exps]


def _entropy_from_logprobs_top(top_map: dict[str, float]) -> Optional[float]:
    """
    Approximate entropy over the candidate set provided by top_logprobs.

    top_map is token -> logprob.
    """
    if not top_map:
        return None
    logps = list(top_map.values())
    ps = _softmax(logps)
    ent = -sum(p * math.log(p + 1e-12) for p in ps)
    return float(ent)


def _approx_first_token_entropy(meta: Any) -> Optional[float]:
    """
    meta is a dict that may contain:
      - "logprobs": list[{..., "top_logprobs": {...}}] produced by providers.py
    We compute entropy from the first position with top_logprobs.
    """
    if not isinstance(meta, dict):
        return None
    logprobs = meta.get("logprobs")
    if not isinstance(logprobs, list) or not logprobs:
        return None
    for pos in logprobs:
        if not isinstance(pos, dict):
            continue
        top = pos.get("top_logprobs")
        if isinstance(top, dict) and top:
            return _entropy_from_logprobs_top(top)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CoT vs direct prompting variants")
    parser.add_argument("input", help="JSONL file (ideally enriched via analysis/build_dataset.py)")
    parser.add_argument(
        "--oracle",
        choices=["strategy_aware", "card_only"],
        default="strategy_aware",
        help="Which oracle posterior to compare beliefs against.",
    )
    parser.add_argument("--divergence", choices=["js", "kl", "l2", "tv"], default="js")
    args = parser.parse_args()

    decisions = [
        r
        for r in _read_jsonl(args.input)
        if isinstance(r, dict) and r.get("type") not in ("run_config", "hand_summary")
    ]

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in decisions:
        llm_extra = r.get("llm_extra") or {}
        belief_meta = llm_extra.get("belief") if isinstance(llm_extra, dict) else None
        action_meta = llm_extra.get("action") if isinstance(llm_extra, dict) else None
        belief_mode = belief_meta.get("belief_mode") if isinstance(belief_meta, dict) else None
        action_mode = action_meta.get("action_mode") if isinstance(action_meta, dict) else None
        if not belief_mode:
            belief_mode = "unknown"
        if not action_mode:
            action_mode = "unknown"
        key = f"belief={belief_mode}, action={action_mode}"
        groups[key].append(r)

    oracle_key = "oracle_strategy_aware" if args.oracle == "strategy_aware" else "oracle_card_only"

    for key, rs in sorted(groups.items()):
        beliefs: list[dict[str, float]] = []
        oracles: list[dict[str, float]] = []
        parse_ok = []
        interp_entropies = []
        for r in rs:
            belief = r.get("agent_belief")
            if not isinstance(belief, dict) or not belief:
                continue
            oracle_b = r.get(oracle_key)
            if not isinstance(oracle_b, dict) or not oracle_b:
                continue

            # record parse success if available
            llm_extra = r.get("llm_extra") or {}
            belief_meta = llm_extra.get("belief") if isinstance(llm_extra, dict) else None
            if isinstance(belief_meta, dict) and "ok" in belief_meta:
                parse_ok.append(bool(belief_meta.get("ok")))

            # interpretability next-token entropy proxy (action call)
            action_meta = (llm_extra.get("action") if isinstance(llm_extra, dict) else None)
            ent = _approx_first_token_entropy(action_meta)
            if ent is not None:
                interp_entropies.append(ent)

            # Ensure all buckets exist for metric stability
            fixed = {b: float(belief.get(b, 0.0)) for b in BUCKET_NAMES}
            beliefs.append(fixed)
            oracles.append({b: float(oracle_b.get(b, 0.0)) for b in BUCKET_NAMES})

        print(f"\n=== {key} ===")
        print(f"n_decisions_total={len(rs)}")
        print(f"n_used_for_metrics={len(beliefs)}")
        if parse_ok:
            print(f"belief_parse_success_rate={sum(parse_ok)/len(parse_ok):.3f}")
        else:
            print("belief_parse_success_rate=NA")

        if beliefs:
            pce = compute_pce(beliefs, oracles, method=args.divergence)
            print(f"PCE_{args.divergence}={pce['pce']:.6f} (std={pce['std']:.6f})")
        if interp_entropies:
            print(f"mean_action_first_token_entropy_proxy={float(np.mean(interp_entropies)):.4f}")


if __name__ == "__main__":
    main()

