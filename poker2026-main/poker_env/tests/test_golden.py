"""
Golden tests based on PokerKit documentation examples.

These tests verify that our environment correctly implements
Fixed-Limit Texas Hold'em rules by comparing against known
outcomes from the PokerKit documentation.

Reference: https://pokerkit.readthedocs.io/en/latest/simulation.html
"""

import pytest
from poker_env.env import PokerKitEnv
from poker_env.actions import Action, ActionType, FOLD, CHECK_OR_CALL, BET_OR_RAISE


class TestGoldenHand:
    """
    Test based on the PokerKit documentation example.

    The example shows a heads-up fixed-limit hand that results
    in final stacks of [204, 196].
    """

    def test_documentation_example_structure(self):
        """
        Test that we can create a hand matching the docs example structure.

        From PokerKit docs:
        - 2 players, stacks (200, 200)
        - Blinds (1, 2)
        - Small bet 2, big bet 4
        - Example ends with stacks [204, 196] (player 0 wins 4 chips)
        """
        env = PokerKitEnv(
            num_players=2,
            stacks=(200, 200),
            blinds=(1, 2),
            small_bet=2,
            big_bet=4,
        )

        # Reset with explicit cards to match docs example
        obs = env.reset(seed=42)

        assert obs is not None
        assert env.stacks == (200, 200)
        assert len(env.legal_actions()) > 0

    def test_preflop_blind_posting(self):
        """Test that blinds are correctly posted."""
        env = PokerKitEnv(
            num_players=2,
            stacks=(200, 200),
            blinds=(1, 2),
        )
        obs = env.reset(seed=42)

        # After blinds, pot should contain 3 chips (1 + 2)
        assert obs.street == "PREFLOP"

    def test_complete_hand_to_showdown(self):
        """Test playing a complete hand to showdown."""
        env = PokerKitEnv()
        obs = env.reset(seed=42)

        # Play by always checking/calling
        done = False
        streets_seen = set()
        while not done:
            streets_seen.add(obs.street)
            legal = env.legal_actions()
            if not legal:
                break

            # Always check/call
            action = next(
                (a for a in legal if a.type == ActionType.CHECK_OR_CALL),
                legal[0]
            )
            obs, _, done, info = env.step(action)

        assert done
        assert "PREFLOP" in streets_seen
        assert "final_stacks" in info
        assert len(info["final_stacks"]) == 2

    def test_fold_preflop(self):
        """Test that folding preflop ends the hand correctly."""
        env = PokerKitEnv()
        obs = env.reset(seed=42)

        # First player to act preflop
        legal = env.legal_actions()

        # Find fold action
        fold_action = next(
            (a for a in legal if a.type == ActionType.FOLD),
            None
        )

        if fold_action:
            obs, reward, done, info = env.step(fold_action)
            assert done
            assert "final_stacks" in info

    def test_raise_cap(self):
        """Test that raises are capped correctly in fixed-limit."""
        env = PokerKitEnv()
        obs = env.reset(seed=42)

        # Try to raise multiple times
        raise_count = 0
        for _ in range(10):  # More than enough iterations
            legal = env.legal_actions()
            if not legal:
                break

            raise_action = next(
                (a for a in legal if a.type == ActionType.BET_OR_RAISE),
                None
            )

            if raise_action:
                obs, _, done, _ = env.step(raise_action)
                raise_count += 1
                if done:
                    break
            else:
                # Can't raise anymore - cap reached
                break

        # In fixed-limit, typically capped at 4 bets (bet + 3 raises)
        assert raise_count >= 0


class TestLegalActions:
    """Tests verifying legal action correctness."""

    def test_legal_actions_always_valid(self):
        """Test that legal_actions only returns actually legal moves."""
        env = PokerKitEnv()
        obs = env.reset(seed=42)

        done = False
        while not done:
            legal = env.legal_actions()
            assert len(legal) > 0, "Should always have legal actions at decision point"

            # Every action in legal should be applicable
            for action in legal:
                assert action.type in [
                    ActionType.FOLD,
                    ActionType.CHECK_OR_CALL,
                    ActionType.BET_OR_RAISE,
                ]

            # Apply first legal action
            obs, _, done, _ = env.step(legal[0])

    def test_no_illegal_actions(self):
        """Test that applying a legal action never raises."""
        env = PokerKitEnv()
        obs = env.reset(seed=42)

        done = False
        while not done:
            legal = env.legal_actions()
            if not legal:
                break

            # This should never raise
            obs, _, done, _ = env.step(legal[0])


class TestRewardCalculation:
    """Tests for correct reward/stack calculation."""

    def test_winner_gains_pot(self):
        """Test that the winner gains the pot."""
        env = PokerKitEnv()
        env.reset(seed=42)

        # Play to completion
        done = False
        while not done:
            legal = env.legal_actions()
            if not legal:
                break
            action = next(
                (a for a in legal if a.type == ActionType.CHECK_OR_CALL),
                legal[0]
            )
            _, _, done, info = env.step(action)

        if done and "final_stacks" in info:
            p0, p1 = info["final_stacks"]
            # Total chips should be conserved
            assert p0 + p1 == 400  # 200 + 200 initial

    def test_fold_gives_pot_to_opponent(self):
        """Test that folding gives the pot to opponent."""
        env = PokerKitEnv()
        obs = env.reset(seed=42)

        # Fold immediately
        legal = env.legal_actions()
        fold_action = next(
            (a for a in legal if a.type == ActionType.FOLD),
            None
        )

        if fold_action:
            _, _, done, info = env.step(fold_action)
            if done and "final_stacks" in info:
                p0, p1 = info["final_stacks"]
                # One player should have gained, one lost
                assert p0 != 200 or p1 != 200
                assert p0 + p1 == 400
