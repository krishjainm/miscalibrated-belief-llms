"""
Decision-point dataset builder for belief analysis.

Post-processes experiment logs to add oracle posteriors,
enabling offline analysis without re-running experiments.
"""

import json
import argparse
from pathlib import Path
from typing import Optional

from analysis.buckets import hand_to_bucket, BUCKET_NAMES
from analysis.opponent_model import ParametricOpponent, PublicState
from analysis.posterior_oracle import (
    CardOnlyPosterior,
    StrategyAwarePosterior,
    extract_opponent_actions,
)


def build_analysis_dataset(
    log_path: str,
    output_path: str,
    opponent_preset: str = "default",
) -> dict:
    """
    Post-process experiment logs to add oracle posteriors.
    
    For each decision point in the log:
    1. Load the observation and hidden state
    2. Compute CardOnlyPosterior
    3. Compute StrategyAwarePosterior (using opponent model)
    4. Add true_opponent_bucket from hidden state
    5. Write enriched record to output
    
    Args:
        log_path: Path to input JSONL log file
        output_path: Path for enriched output file
        opponent_preset: Opponent model preset for strategy-aware oracle
        
    Returns:
        Summary statistics
    """
    # Create oracles
    card_only = CardOnlyPosterior()
    opponent_model = ParametricOpponent.from_preset(opponent_preset)
    strategy_aware = StrategyAwarePosterior(opponent_model)
    
    # Statistics
    stats = {
        "decisions_processed": 0,
        "hands_processed": 0,
        "decisions_with_oracle": 0,
        "opponent_preset": opponent_preset,
    }
    
    # Process log file
    output_records = []
    
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            record = json.loads(line)
            record_type = record.get("type")
            
            # Pass through config and summary records unchanged
            if record_type in ("run_config", "hand_summary"):
                if record_type == "hand_summary":
                    stats["hands_processed"] += 1
                output_records.append(record)
                continue
            
            # Process decision records
            stats["decisions_processed"] += 1
            
            # Extract needed info
            obs = record.get("obs", {})
            hidden = record.get("hidden", {})
            
            hero_hole = obs.get("hero_hole", [])
            board = obs.get("board", [])
            history = obs.get("history", [])
            player_index = obs.get("player_index", 0)
            
            # Get opponent hole cards from hidden state
            opponent_hole = None
            for key, value in hidden.items():
                if key.startswith("player") and key.endswith("_hole"):
                    player_num = int(key.replace("player", "").replace("_hole", ""))
                    if player_num != player_index:
                        opponent_hole = value
                        break
            
            # Compute oracle posteriors if we have needed info
            if hero_hole and len(hero_hole) == 2:
                # Card-only posterior
                try:
                    oracle_card_only = card_only.compute(
                        hero_hole=hero_hole,
                        board=board,
                        buckets=True,
                    )
                    record["oracle_card_only"] = oracle_card_only
                except Exception as e:
                    record["oracle_card_only"] = None
                    record["oracle_card_only_error"] = str(e)
                
                # Strategy-aware posterior
                try:
                    opponent_actions = extract_opponent_actions(history, player_index)
                    oracle_strategy_aware = strategy_aware.compute(
                        hero_hole=hero_hole,
                        board=board,
                        opponent_actions=opponent_actions,
                        buckets=True,
                    )
                    record["oracle_strategy_aware"] = oracle_strategy_aware
                    stats["decisions_with_oracle"] += 1
                except Exception as e:
                    record["oracle_strategy_aware"] = None
                    record["oracle_strategy_aware_error"] = str(e)
            
            # Add true opponent bucket
            if opponent_hole and len(opponent_hole) == 2:
                try:
                    record["true_opponent_bucket"] = hand_to_bucket(opponent_hole, board)
                    record["true_opponent_hole"] = opponent_hole
                except Exception:
                    record["true_opponent_bucket"] = None
            
            output_records.append(record)
    
    # Write output
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for record in output_records:
            f.write(json.dumps(record) + "\n")
    
    stats["output_path"] = output_path
    return stats


def load_analysis_dataset(path: str) -> list[dict]:
    """
    Load enriched decision records from analysis dataset.
    
    Args:
        path: Path to enriched JSONL file
        
    Returns:
        List of decision record dicts (excludes config/summary records)
    """
    decisions = []
    
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            record = json.loads(line)
            
            # Skip non-decision records
            if record.get("type") in ("run_config", "hand_summary"):
                continue
            
            decisions.append(record)
    
    return decisions


def load_hand_summaries_from_dataset(path: str) -> list[dict]:
    """Load only hand summary records."""
    summaries = []
    
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                if record.get("type") == "hand_summary":
                    summaries.append(record)
    
    return summaries


def extract_beliefs_and_oracles(
    dataset: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Extract parallel lists of beliefs and oracles from dataset.
    
    Args:
        dataset: List of decision records
        
    Returns:
        Tuple of (agent_beliefs, card_only_oracles, strategy_aware_oracles)
        Each is a list of bucket probability dicts
    """
    agent_beliefs = []
    card_only_oracles = []
    strategy_aware_oracles = []
    
    for record in dataset:
        # Agent belief (may be None if not LLM agent)
        agent_belief = record.get("agent_belief")
        if agent_belief and isinstance(agent_belief, dict):
            agent_beliefs.append(agent_belief)
        else:
            # Create uniform placeholder
            agent_beliefs.append({b: 1.0/len(BUCKET_NAMES) for b in BUCKET_NAMES})
        
        # Card-only oracle
        co = record.get("oracle_card_only")
        if co and isinstance(co, dict):
            card_only_oracles.append(co)
        else:
            card_only_oracles.append({b: 1.0/len(BUCKET_NAMES) for b in BUCKET_NAMES})
        
        # Strategy-aware oracle
        sa = record.get("oracle_strategy_aware")
        if sa and isinstance(sa, dict):
            strategy_aware_oracles.append(sa)
        else:
            strategy_aware_oracles.append({b: 1.0/len(BUCKET_NAMES) for b in BUCKET_NAMES})
    
    return agent_beliefs, card_only_oracles, strategy_aware_oracles


def main():
    """CLI for building analysis datasets."""
    parser = argparse.ArgumentParser(
        description="Build analysis dataset with oracle posteriors"
    )
    parser.add_argument(
        "input",
        help="Input experiment log (JSONL)",
    )
    parser.add_argument(
        "output",
        help="Output enriched dataset (JSONL)",
    )
    parser.add_argument(
        "--opponent",
        choices=["default", "tight_passive", "tight_aggressive", 
                 "loose_passive", "loose_aggressive"],
        default="default",
        help="Opponent model preset for strategy-aware oracle",
    )
    
    args = parser.parse_args()
    
    print(f"Building analysis dataset...")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Opponent model: {args.opponent}")
    
    stats = build_analysis_dataset(
        log_path=args.input,
        output_path=args.output,
        opponent_preset=args.opponent,
    )
    
    print(f"\nDone!")
    print(f"  Hands processed: {stats['hands_processed']}")
    print(f"  Decisions processed: {stats['decisions_processed']}")
    print(f"  Decisions with oracle: {stats['decisions_with_oracle']}")


if __name__ == "__main__":
    main()
