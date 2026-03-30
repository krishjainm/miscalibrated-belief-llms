#!/usr/bin/env python3
"""
Run heads-up poker experiments with an API-based LLM on one seat and a baseline bot on others.

Requires optional dependencies from requirements-llm.txt and the right API key in the environment.

Examples:
    # GPT-4o mini vs call-station, direct JSON belief (no CoT)
    set OPENAI_API_KEY=...
    python run_llm_experiment.py --preset gpt-4o-mini --belief-mode direct --hands 5 -v

    # Claude with chain-of-thought belief elicitation
    set ANTHROPIC_API_KEY=...
    python run_llm_experiment.py --preset claude-3-5-sonnet --belief-mode cot --hands 3 -v

    # Mistral (OpenAI-compatible endpoint)
    set MISTRAL_API_KEY=...
    python run_llm_experiment.py --preset mistral-small --belief-mode direct --hands 5 -v

    # Qwen on Together
    set TOGETHER_API_KEY=...
    python run_llm_experiment.py --preset qwen2.5-72b-together --belief-mode direct --hands 3 -v
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_experiment import create_agent, run_experiment
from poker_env.agents import BaseAgent

from llm.llm_agent import LLMAgent
from llm.model_registry import list_presets


def build_agents(
    num_players: int,
    llm_player: int,
    preset: str,
    opponent: str,
    belief_mode: str,
    action_mode: str,
    temperature: float,
    base_seed: int,
    request_logprobs: int | None,
    local_interp_model_name: str | None,
    interp_layers: list[int] | None,
    interp_top_k: int,
    interp_max_calls: int,
) -> list[BaseAgent]:
    agents: list[BaseAgent] = []
    for i in range(num_players):
        if i == llm_player:
            agents.append(
                LLMAgent(
                    preset,
                    belief_mode=belief_mode,
                    action_mode=action_mode,
                    temperature=temperature,
                    name=f"Player{i}_{preset}",
                    request_logprobs=request_logprobs,
                    local_interp_model_name=local_interp_model_name,
                    interp_layers=interp_layers,
                    interp_top_k=interp_top_k,
                    interp_max_calls=interp_max_calls,
                )
            )
        else:
            seed = base_seed + i if opponent == "random" else None
            agents.append(
                create_agent(opponent, seed=seed, name=f"Player{i}_{opponent}")
            )
    return agents


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run belief modeling experiments with API LLM agents"
    )
    parser.add_argument(
        "--preset",
        type=str,
        required=True,
        choices=list_presets(),
        help="Model preset (see llm/model_registry.py)",
    )
    parser.add_argument(
        "--belief-mode",
        type=str,
        default="direct",
        choices=["direct", "cot"],
        help="direct: JSON-only belief prompt; cot: reasoning + JSON sections",
    )
    parser.add_argument(
        "--action-mode",
        type=str,
        default="direct",
        choices=["direct", "cot"],
        help="direct: JSON-only action prompt; cot: REASONING + JSON sections",
    )
    parser.add_argument(
        "--llm-player",
        type=int,
        default=0,
        help="Seat index (0..N-1) controlled by the LLM",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="call",
        choices=["random", "call"],
        help="Baseline bot type for non-LLM seats",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Players at table (heads-up recommended)",
    )
    parser.add_argument(
        "--hands",
        type=int,
        default=10,
        help="Number of hands",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="logs/llm_experiment.jsonl",
        help="JSONL log path",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for API calls",
    )
    parser.add_argument(
        "--no-oracle",
        action="store_true",
        help="Skip equity oracle (faster)",
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=None,
        metavar="K",
        help="If set, request top-K logprobs per token (OpenAI-compatible APIs only)",
    )
    parser.add_argument(
        "--local-interp-model",
        type=str,
        default=None,
        help="Enable local interpretability (logit lens + attention diagnostics) using this HF checkpoint.",
    )
    parser.add_argument(
        "--interp-layers",
        type=str,
        default=None,
        metavar="LAYERS",
        help=(
            "Comma-separated layer indices for local interpretability (e.g. -1,-2). "
            "If values start with '-', use equals form so argparse does not treat them as flags: "
            "--interp-layers=-1,-2 (PowerShell/bash)."
        ),
    )
    parser.add_argument(
        "--interp-top-k",
        type=int,
        default=10,
        help="Top-K tokens to log per layer for local interpretability.",
    )
    parser.add_argument(
        "--interp-max-calls",
        type=int,
        default=0,
        help="Max number of local interpretability runs total (0 disables).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    args = parser.parse_args()

    if not (0 <= args.llm_player < args.num_players):
        parser.error("--llm-player must be in [0, num-players)")

    prompt_version = f"belief_{args.belief_mode}_cot_prompts_v1"
    prompt_version += f"_action_{args.action_mode}_v1"

    interp_layers = None
    if args.interp_layers:
        interp_layers = [int(x.strip()) for x in args.interp_layers.split(",") if x.strip()]

    agents = build_agents(
        num_players=args.num_players,
        llm_player=args.llm_player,
        preset=args.preset,
        opponent=args.opponent,
        belief_mode=args.belief_mode,
        action_mode=args.action_mode,
        temperature=args.temperature,
        base_seed=args.seed,
        request_logprobs=args.top_logprobs,
        local_interp_model_name=args.local_interp_model,
        interp_layers=interp_layers,
        interp_top_k=args.interp_top_k,
        interp_max_calls=args.interp_max_calls,
    )

    run_experiment(
        num_hands=args.hands,
        agent_types=["llm_placeholder"],
        num_players=args.num_players,
        output_path=args.out,
        base_seed=args.seed,
        compute_oracle=not args.no_oracle,
        verbose=args.verbose,
        agents=agents,
        prompt_version=prompt_version,
    )


if __name__ == "__main__":
    main()
