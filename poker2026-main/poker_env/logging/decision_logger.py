"""JSONL decision logging for experiment replay and analysis."""

import json
import hashlib
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, asdict

from poker_env.obs import Obs
from poker_env.actions import Action


@dataclass
class DecisionRecord:
    """
    Record of a single decision point for logging.

    Contains all information needed to replay and analyze
    a decision point in a poker hand.
    """

    hand_id: str
    seed: int
    decision_idx: int
    player_to_act: int
    street: str
    obs: dict
    hidden: dict
    legal_actions: list[str]
    agent_belief: Optional[dict]
    agent_action: str
    equity_given_true_hands: Optional[dict]  # Renamed from oracle_truth

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class HandSummary:
    """Summary record for a completed hand."""

    hand_id: str
    seed: int
    num_decisions: int
    final_stacks: list[int]
    deltas: dict  # {"player0_delta": float, "player1_delta": float, ...}
    showdown: dict


@dataclass
class RunConfig:
    """Configuration for a run, used for reproducibility tracking."""

    env_config_hash: str
    agent_configs: list[dict]  # Config for each agent
    prompt_version: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class DecisionLogger:
    """
    Logger that writes decision records to JSONL files.

    Records one JSON object per line for each decision point,
    plus a summary record at the end of each hand.
    Supports 2-6 players.

    Includes config hashes for paper-quality reproducibility.
    """

    def __init__(
        self,
        output_path: str,
        env_config_hash: str = "",
        agent_configs: list[dict] | None = None,
        prompt_version: str | None = None,
    ):
        """
        Initialize the logger.

        Args:
            output_path: Path to output JSONL file
            env_config_hash: Hash of environment config for reproducibility
            agent_configs: List of agent configurations
            prompt_version: Version string for prompt templates
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Run config for reproducibility
        self.run_config = RunConfig(
            env_config_hash=env_config_hash,
            agent_configs=agent_configs or [],
            prompt_version=prompt_version,
        )

        # Current hand state
        self._current_hand_id: Optional[str] = None
        self._current_seed: int = 0
        self._decision_idx: int = 0
        self._records: list[DecisionRecord] = []

        # Open file in append mode
        self._file = None
        self._header_written = False

    def __enter__(self):
        """Context manager entry."""
        self._file = open(self.output_path, "a")
        # Write run config header if file is empty
        if self.output_path.stat().st_size == 0:
            self._write_header()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._file:
            self._file.close()
            self._file = None

    def _write_header(self):
        """Write run configuration header."""
        header = {
            "type": "run_config",
            **self.run_config.to_dict(),
        }
        self._write_record(header)
        self._header_written = True

    def start_hand(self, hand_id: str, seed: int) -> None:
        """
        Start logging a new hand.

        Args:
            hand_id: Unique identifier for the hand
            seed: Random seed used for the hand
        """
        self._current_hand_id = hand_id
        self._current_seed = seed
        self._decision_idx = 0
        self._records = []

    def log_decision(
        self,
        obs: Obs,
        hidden: dict,
        agent_action: Action,
        agent_belief: Optional[dict] = None,
        equity_given_true_hands: Optional[dict] = None,
    ) -> None:
        """
        Log a decision point.

        Args:
            obs: Observation at the decision point
            hidden: Hidden state (all players' hole cards, etc.)
            agent_action: Action selected by the agent
            agent_belief: Optional belief dict from agent
            equity_given_true_hands: Ground truth equity from oracle (computed AFTER agent acts)
        """
        record = DecisionRecord(
            hand_id=obs.hand_id,
            seed=obs.seed,
            decision_idx=self._decision_idx,
            player_to_act=obs.to_act,
            street=obs.street,
            obs=obs.to_dict(),
            hidden=hidden,
            legal_actions=[a.type.value for a in obs.legal_actions],
            agent_belief=agent_belief,
            agent_action=agent_action.type.value,
            equity_given_true_hands=equity_given_true_hands,
        )

        self._records.append(record)
        self._write_record(record.to_dict())
        self._decision_idx += 1

    def end_hand(
        self,
        final_stacks: list[int],
        deltas: dict,
        showdown: dict,
    ) -> None:
        """
        Log hand completion summary.

        Args:
            final_stacks: Final stack sizes for all players
            deltas: Stack changes for all players {"player0_delta": float, ...}
            showdown: Showdown information (hole cards, board)
        """
        summary = HandSummary(
            hand_id=self._current_hand_id or "",
            seed=self._current_seed,
            num_decisions=self._decision_idx,
            final_stacks=final_stacks,
            deltas=deltas,
            showdown=showdown,
        )

        self._write_record({
            "type": "hand_summary",
            **asdict(summary),
        })

        # Reset state
        self._current_hand_id = None
        self._decision_idx = 0
        self._records = []

    def _write_record(self, record: dict) -> None:
        """Write a single record to the JSONL file."""
        if self._file:
            self._file.write(json.dumps(record) + "\n")
            self._file.flush()


def load_decisions(jsonl_path: str) -> list[dict]:
    """
    Load decision records from a JSONL file.

    Args:
        jsonl_path: Path to JSONL file

    Returns:
        List of decision record dictionaries
    """
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_hand_summaries(jsonl_path: str) -> list[dict]:
    """
    Load only hand summary records from a JSONL file.

    Args:
        jsonl_path: Path to JSONL file

    Returns:
        List of hand summary dictionaries
    """
    summaries = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                if record.get("type") == "hand_summary":
                    summaries.append(record)
    return summaries


def load_run_config(jsonl_path: str) -> Optional[dict]:
    """
    Load run configuration from a JSONL file.

    Args:
        jsonl_path: Path to JSONL file

    Returns:
        Run config dict or None if not found
    """
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                if record.get("type") == "run_config":
                    return record
    return None
