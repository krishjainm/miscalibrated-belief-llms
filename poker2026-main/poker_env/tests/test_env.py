"""Basic tests for the PokerKitEnv environment."""

import warnings
import pytest
from poker_env.env import PokerKitEnv, HandManifest
from poker_env.actions import Action, ActionType, FOLD, CHECK_OR_CALL, BET_OR_RAISE


class TestPokerKitEnvBasic:
    """Basic functionality tests for PokerKitEnv."""

    def test_env_creation(self):
        """Test that environment can be created."""
        env = PokerKitEnv()
        assert env is not None
        assert env.num_players == 2
        assert env.stacks == (200, 200)
        assert env.blinds == (1, 2)

    def test_env_custom_config(self):
        """Test environment with custom configuration."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = PokerKitEnv(
                num_players=4,
                stacks=(500, 500, 500, 500),
                blinds=(2, 4),
                small_bet=4,
                big_bet=8,
            )
        assert env.num_players == 4
        assert env.stacks == (500, 500, 500, 500)
        assert env.blinds == (2, 4)

    def test_env_default_stacks(self):
        """Test that default stacks work for any player count."""
        for n in range(2, 7):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                env = PokerKitEnv(num_players=n)
            assert env.num_players == n
            assert len(env.stacks) == n
            assert all(s == 200 for s in env.stacks)

    def test_multiway_warning(self):
        """Test that multi-way play shows a warning."""
        with pytest.warns(UserWarning, match="Multi-way"):
            env = PokerKitEnv(num_players=4)

    def test_reset_returns_obs(self):
        """Test that reset returns a valid observation."""
        env = PokerKitEnv()
        obs = env.reset(seed=42)

        assert obs is not None
        assert obs.hand_id != ""
        assert obs.seed == 42
        assert len(obs.hero_hole) == 2
        assert obs.street == "PREFLOP"
        assert len(obs.legal_actions) > 0

    def test_obs_new_fields(self):
        """Test new observation fields."""
        env = PokerKitEnv()
        obs = env.reset(seed=42)

        # Check new fields exist
        assert obs.street_index == 0  # PREFLOP
        assert obs.button == 0
        assert obs.position in ["BTN/SB", "BB"]
        assert obs.bet_to_call >= 0
        assert obs.raises_remaining >= 0
        assert len(obs.contrib_this_round) == 2
        assert len(obs.contrib_total) == 2

    def test_legal_actions_not_empty(self):
        """Test that legal actions are never empty at decision point."""
        env = PokerKitEnv()
        env.reset(seed=42)

        legal = env.legal_actions()
        assert len(legal) > 0
        assert all(isinstance(a, Action) for a in legal)

    def test_step_advances_game(self):
        """Test that step advances the game state."""
        env = PokerKitEnv()
        obs1 = env.reset(seed=42)

        # Get first legal action and apply it
        action = obs1.legal_actions[0]
        obs2, reward, done, info = env.step(action)

        # Game should have progressed (either new player or done)
        assert obs2 is not None

    def test_rewards_in_info(self):
        """Test that rewards dict is in info."""
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

        if done:
            assert "rewards" in info
            assert 0 in info["rewards"]
            assert 1 in info["rewards"]

    def test_get_obs_player_specific(self):
        """Test that observations are player-specific."""
        env = PokerKitEnv()
        env.reset(seed=42)

        obs0 = env.get_obs(0)
        obs1 = env.get_obs(1)

        # Each player should only see their own hole cards
        assert obs0.player_index == 0
        assert obs1.player_index == 1
        # Both should have exactly 2 hole cards
        assert len(obs0.hero_hole) == 2
        assert len(obs1.hero_hole) == 2

    def test_current_player_valid(self):
        """Test that current_player returns valid index."""
        env = PokerKitEnv()
        env.reset(seed=42)

        player = env.current_player()
        assert player in [0, 1]

    def test_hidden_state_has_all_holes(self):
        """Test that hidden state includes all players' hole cards."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = PokerKitEnv(num_players=4)
        env.reset(seed=42)

        hidden = env.get_hidden_state()
        for i in range(4):
            assert f"player{i}_hole" in hidden
            assert len(hidden[f"player{i}_hole"]) == 2

    def test_config_hash(self):
        """Test that config hash is deterministic."""
        env1 = PokerKitEnv()
        env2 = PokerKitEnv()

        assert env1.get_config_hash() == env2.get_config_hash()

        # Different config = different hash
        env3 = PokerKitEnv(stacks=(100, 100))
        assert env1.get_config_hash() != env3.get_config_hash()


class TestManifest:
    """Tests for hand manifest export/import."""

    def test_export_manifest(self):
        """Test that manifest can be exported."""
        env = PokerKitEnv()
        env.reset(seed=42)

        # Play to completion
        done = False
        while not done:
            legal = env.legal_actions()
            if not legal:
                break
            _, _, done, _ = env.step(legal[0])

        manifest = env.export_manifest()
        assert manifest.seed == 42
        assert len(manifest.hole_cards) == 2
        assert len(manifest.actions) > 0
        assert manifest.env_config_hash == env.get_config_hash()

    def test_manifest_json_roundtrip(self):
        """Test manifest JSON serialization."""
        env = PokerKitEnv()
        env.reset(seed=42)

        # Play a bit
        for _ in range(3):
            legal = env.legal_actions()
            if not legal:
                break
            obs, _, done, _ = env.step(legal[0])
            if done:
                break

        manifest = env.export_manifest()
        json_str = manifest.to_json()
        loaded = HandManifest.from_json(json_str)

        assert loaded.seed == manifest.seed
        assert loaded.env_config_hash == manifest.env_config_hash


class TestActionApplication:
    """Tests for action application."""

    def test_fold_ends_hand(self):
        """Test that fold typically ends the hand."""
        env = PokerKitEnv()
        env.reset(seed=42)

        # Keep stepping with fold until hand ends
        done = False
        steps = 0
        while not done and steps < 100:
            legal = env.legal_actions()
            # Try to fold if possible
            fold_action = next((a for a in legal if a.type == ActionType.FOLD), None)
            if fold_action:
                _, _, done, _ = env.step(fold_action)
            else:
                # If can't fold, do any legal action
                _, _, done, _ = env.step(legal[0])
            steps += 1

        assert done, "Hand should complete within 100 steps"

    def test_check_call_continues(self):
        """Test that check/call allows game to continue."""
        env = PokerKitEnv()
        env.reset(seed=42)

        # Do a few check/calls
        for _ in range(4):
            legal = env.legal_actions()
            if not legal:
                break
            call_action = next(
                (a for a in legal if a.type == ActionType.CHECK_OR_CALL),
                legal[0]
            )
            _, _, done, _ = env.step(call_action)
            if done:
                break

    def test_bet_raise_is_valid(self):
        """Test that bet/raise is properly handled."""
        env = PokerKitEnv()
        env.reset(seed=42)

        legal = env.legal_actions()
        raise_action = next(
            (a for a in legal if a.type == ActionType.BET_OR_RAISE),
            None
        )

        if raise_action:
            obs, _, _, _ = env.step(raise_action)
            assert obs is not None


class TestHandCompletion:
    """Tests for hand completion scenarios."""

    def test_hand_completes(self):
        """Test that a hand eventually completes."""
        env = PokerKitEnv()
        env.reset(seed=42)

        done = False
        steps = 0
        while not done and steps < 200:
            legal = env.legal_actions()
            if not legal:
                break
            # Always check/call to eventually reach showdown
            action = next(
                (a for a in legal if a.type == ActionType.CHECK_OR_CALL),
                legal[0]
            )
            _, _, done, info = env.step(action)
            steps += 1

        assert done, "Hand should complete"

    def test_rewards_sum_to_zero(self):
        """Test that rewards are zero-sum."""
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

        if done and "rewards" in info:
            total = sum(info["rewards"].values())
            # Rewards should be zero-sum (accounting for rake=0)
            assert abs(total) < 0.01, "Rewards should be zero-sum"


class TestMultiWay:
    """Tests for multi-way (3-6 player) games."""

    @pytest.mark.parametrize("num_players", [3, 4, 5, 6])
    def test_multiway_creation(self, num_players):
        """Test that multi-way environments can be created."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = PokerKitEnv(num_players=num_players)
        assert env.num_players == num_players
        assert len(env.stacks) == num_players

    @pytest.mark.parametrize("num_players", [3, 4, 5, 6])
    def test_multiway_hand_completes(self, num_players):
        """Test that multi-way hands complete correctly."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = PokerKitEnv(num_players=num_players)
        env.reset(seed=42)

        done = False
        steps = 0
        while not done and steps < 500:
            legal = env.legal_actions()
            if not legal:
                break
            action = next(
                (a for a in legal if a.type == ActionType.CHECK_OR_CALL),
                legal[0]
            )
            _, _, done, info = env.step(action)
            steps += 1

        assert done, f"Hand should complete for {num_players} players"
        assert "final_stacks" in info
        assert len(info["final_stacks"]) == num_players

    def test_multiway_zero_sum(self):
        """Test that multi-way rewards are zero-sum."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = PokerKitEnv(num_players=4)
        env.reset(seed=42)

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

        if done and "rewards" in info:
            total = sum(info["rewards"].values())
            assert abs(total) < 0.01, "Multi-way rewards should be zero-sum"
