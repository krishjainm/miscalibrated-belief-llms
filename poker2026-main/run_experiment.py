#!/usr/bin/env python3
"""
Run poker experiments with specified agents.

Usage:
    # 2 players (heads-up)
    python run_experiment.py --agent random --hands 100 --seed 42 --out logs/test.jsonl -v

    # 4 players (will show warning - heads-up recommended for belief research)
    python run_experiment.py --num-players 4 --agents random,call,random,call --hands 100 -v

    # 6 players all random
    python run_experiment.py --num-players 6 --agent random --hands 100 -v
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional

from poker_env.env import PokerKitEnv
from poker_env.agents import BaseAgent, RandomAgent, CallAgent
from poker_env.oracle import EquityOracle
from poker_env.logging import DecisionLogger


def create_agent(agent_type: str, seed: Optional[int] = None, name: str = "") -> BaseAgent:
    """
    Create an agent by type name.

    Args:
        agent_type: One of "random", "call"
        seed: Optional seed for random agent
        name: Optional name for the agent

    Returns:
        BaseAgent instance
    """
    if agent_type == "random":
        return RandomAgent(seed=seed, name=name or "RandomAgent")
    elif agent_type == "call":
        return CallAgent(name=name or "CallAgent")
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def create_agents(
    agent_types: list[str],
    num_players: int,
    base_seed: int,
) -> list[BaseAgent]:
    """
    Create agents for all players.

    Args:
        agent_types: List of agent types (will be cycled if shorter than num_players)
        num_players: Number of players
        base_seed: Base seed for random agents

    Returns:
        List of BaseAgent instances
    """
    agents = []
    for i in range(num_players):
        agent_type = agent_types[i % len(agent_types)]
        seed = base_seed + i if agent_type == "random" else None
        agents.append(create_agent(agent_type, seed=seed, name=f"Player{i}_{agent_type}"))
    return agents


def get_agent_configs(agents: list[BaseAgent]) -> list[dict]:
    """Get configuration dicts for all agents."""
    configs = []
    for agent in agents:
        config = {
            "name": agent.name,
            "type": type(agent).__name__,
        }
        if hasattr(agent, "seed"):
            config["seed"] = agent.seed
        configs.append(config)
    return configs


def run_single_hand(
    env: PokerKitEnv,
    agents: list[BaseAgent],
    oracle: EquityOracle,
    logger: Optional[DecisionLogger],
    seed: int,
    compute_oracle: bool = True,
) -> dict:
    """
    Run a single hand of poker.

    Args:
        env: Poker environment
        agents: List of agents for each player
        oracle: Equity oracle (ground truth, computed AFTER agent acts)
        logger: Optional decision logger
        seed: Random seed for this hand
        compute_oracle: Whether to compute oracle truth at each decision

    Returns:
        Dict with hand results
    """
    # Reset agents
    for agent in agents:
        agent.reset()

    # Start new hand
    obs = env.reset(seed=seed)

    if logger:
        logger.start_hand(env.hand_id, seed)

    done = False
    while not done:
        player = env.current_player()
        agent = agents[player]

        # Agent acts based ONLY on observation (no oracle info)
        action = agent.act(obs)
        belief = agent.belief(obs)

        # Oracle computed AFTER agent acts - for logging only, never seen by agent
        equity_truth = None
        if compute_oracle:
            hidden = env.get_hidden_state()
            hero_hole = hidden.get(f"player{player}_hole", [])

            # Gather all opponent hole cards
            opponent_holes = []
            for i in range(env.num_players):
                if i != player:
                    opp_hole = hidden.get(f"player{i}_hole", [])
                    if opp_hole:
                        opponent_holes.append(opp_hole)

            # Each element in board_cards is a list with one card
            board = [repr(card_list[0]) for card_list in env.state.board_cards if card_list]

            if hero_hole and opponent_holes:
                equity_truth = oracle.compute(hero_hole, opponent_holes, board)

        # Log decision (oracle truth is for evaluation, agent never saw it)
        if logger:
            logger.log_decision(
                obs=obs,
                hidden=env.get_hidden_state(),
                agent_action=action,
                agent_belief=belief,
                equity_given_true_hands=equity_truth,
            )

        # Apply action
        obs, reward, done, info = env.step(action)

    # Log hand completion
    if logger:
        deltas = info.get("deltas", {})
        logger.end_hand(
            final_stacks=info.get("final_stacks", []),
            deltas=deltas,
            showdown=info.get("showdown", {}),
        )

    return {
        "hand_id": env.hand_id,
        "seed": seed,
        "final_stacks": info.get("final_stacks", []),
        "deltas": info.get("deltas", {}),
        "rewards": info.get("rewards", {}),
    }


def run_experiment(
    num_hands: int,
    agent_types: list[str],
    num_players: int,
    output_path: str,
    base_seed: int = 42,
    stacks: tuple[int, ...] | None = None,
    blinds: tuple[int, int] = (1, 2),
    small_bet: int = 2,
    big_bet: int = 4,
    compute_oracle: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Run a full experiment with multiple hands.

    Args:
        num_hands: Number of hands to play
        agent_types: List of agent types for each player
        num_players: Number of players (2-6, heads-up recommended)
        output_path: Path for JSONL output
        base_seed: Base random seed
        stacks: Starting stack sizes (None = 200 each)
        blinds: Blind amounts
        small_bet: Small bet size
        big_bet: Big bet size
        compute_oracle: Whether to compute oracle truth
        verbose: Print progress

    Returns:
        Dict with experiment summary
    """
    # Suppress multi-way warning during experiment if user explicitly chose it
    with warnings.catch_warnings():
        if num_players > 2:
            warnings.filterwarnings("ignore", message="Multi-way.*")

        # Create environment
        env = PokerKitEnv(
            num_players=num_players,
            stacks=stacks,
            blinds=blinds,
            small_bet=small_bet,
            big_bet=big_bet,
        )

    # Create agents
    agents = create_agents(agent_types, num_players, base_seed)

    # Create oracle
    oracle = EquityOracle(num_samples=5000, seed=base_seed + 100)

    # Tracking
    total_deltas = {f"player{i}": 0.0 for i in range(num_players)}
    hand_results = []

    # Create logger with config info
    with DecisionLogger(
        output_path,
        env_config_hash=env.get_config_hash(),
        agent_configs=get_agent_configs(agents),
    ) as logger:
        for i in range(num_hands):
            hand_seed = base_seed + i * 1000

            result = run_single_hand(
                env=env,
                agents=agents,
                oracle=oracle,
                logger=logger,
                seed=hand_seed,
                compute_oracle=compute_oracle,
            )

            hand_results.append(result)
            for key, delta in result.get("deltas", {}).items():
                player_key = key.replace("_delta", "")
                if player_key in total_deltas:
                    total_deltas[player_key] += delta

            if verbose and (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_hands} hands")

    # Summary
    summary = {
        "num_hands": num_hands,
        "num_players": num_players,
        "agents": [a.name for a in agents],
        "env_config_hash": env.get_config_hash(),
        "total_deltas": total_deltas,
        "avg_deltas": {k: v / num_hands for k, v in total_deltas.items()},
        "output_path": output_path,
    }

    if verbose:
        print(f"\n=== Experiment Summary ({num_players} players) ===")
        print(f"Hands played: {num_hands}")
        print(f"Config hash: {env.get_config_hash()}")
        for i, agent in enumerate(agents):
            delta = total_deltas[f"player{i}"]
            avg = delta / num_hands
            print(f"Player {i} ({agent.name}): {delta:+.1f} chips ({avg:+.2f}/hand)")
        print(f"Output: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run poker experiments with specified agents"
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players (2-6, default: 2). Heads-up recommended for belief research.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="random",
        choices=["random", "call"],
        help="Default agent type for all players (default: random)",
    )
    parser.add_argument(
        "--agents",
        type=str,
        default=None,
        help="Comma-separated agent types for each player (e.g., 'random,call,random,call')",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default=None,
        choices=["random", "call"],
        help="Agent type for opponents (heads-up only, overrides --agent for player 1)",
    )
    parser.add_argument(
        "--hands",
        type=int,
        default=100,
        help="Number of hands to play (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="logs/experiment.jsonl",
        help="Output JSONL file path (default: logs/experiment.jsonl)",
    )
    parser.add_argument(
        "--no-oracle",
        action="store_true",
        help="Skip oracle computation (faster)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress",
    )

    args = parser.parse_args()

    # Determine agent types
    if args.agents:
        # Explicit list of agents
        agent_types = [a.strip() for a in args.agents.split(",")]
    elif args.opponent and args.num_players == 2:
        # Heads-up with different agents
        agent_types = [args.agent, args.opponent]
    else:
        # All same agent type
        agent_types = [args.agent]

    run_experiment(
        num_hands=args.hands,
        agent_types=agent_types,
        num_players=args.num_players,
        output_path=args.out,
        base_seed=args.seed,
        compute_oracle=not args.no_oracle,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
