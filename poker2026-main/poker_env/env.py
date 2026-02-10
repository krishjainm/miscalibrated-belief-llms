"""
PokerKitEnv - Core poker environment wrapper for Fixed-Limit Texas Hold'em.

This module provides a clean reset/step interface around PokerKit's
FixedLimitTexasHoldem game, with support for deterministic dealing
and observation encoding. Supports 2-6 players (heads-up recommended for v1).
"""

import hashlib
import json
import uuid
import warnings
from typing import Optional
from dataclasses import dataclass, asdict

from pokerkit import Automation, FixedLimitTexasHoldem

from poker_env.actions import Action, ActionType, apply_action, get_legal_actions
from poker_env.obs import Obs, build_observation
from poker_env.deck import DeterministicDeck


# Automations to handle non-decision phases automatically
AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.CARD_BURNING,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)

# Fixed-limit typically allows 4 bets per street (1 bet + 3 raises)
MAX_RAISES_PER_STREET = 4


@dataclass
class EnvConfig:
    """Environment configuration for hashing and reproducibility."""
    num_players: int
    stacks: tuple[int, ...]
    blinds: tuple[int, int]
    small_bet: int
    big_bet: int

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        """Generate deterministic hash of config."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


@dataclass
class HandManifest:
    """
    Complete manifest for reproducing a hand exactly.
    
    Use export_manifest() to capture and reset_from_manifest() to replay.
    """
    seed: int
    deck_order: list[str]
    hole_cards: list[str]  # Dealt hole cards for each player
    board_cards: list[str]  # Dealt board cards
    actions: list[dict]  # All actions taken
    env_config: dict
    env_config_hash: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "HandManifest":
        data = json.loads(json_str)
        return cls(**data)


class PokerKitEnv:
    """
    Fixed-Limit Texas Hold'em environment supporting 2-6 players.

    Wraps PokerKit's FixedLimitTexasHoldem with a clean reset/step interface
    suitable for research on LLM belief modeling.

    Note: For belief modeling research, heads-up (2 players) is recommended.
    Multi-way play adds significant complexity for belief computation.

    Attributes:
        num_players: Number of players (2-6, heads-up recommended)
        stacks: Starting stack sizes for all players
        blinds: Small blind and big blind amounts
        small_bet: Bet size for preflop and flop
        big_bet: Bet size for turn and river
    """

    def __init__(
        self,
        num_players: int = 2,
        stacks: tuple[int, ...] | None = None,
        blinds: tuple[int, int] = (1, 2),
        small_bet: int = 2,
        big_bet: int = 4,
    ):
        """
        Initialize the poker environment.

        Args:
            num_players: Number of players (2-6). Heads-up (2) recommended for v1.
            stacks: Starting stacks for each player. If None, defaults to 200 for each.
            blinds: (small_blind, big_blind) amounts
            small_bet: Fixed bet size for preflop and flop streets
            big_bet: Fixed bet size for turn and river streets
        """
        if num_players < 2 or num_players > 6:
            raise ValueError(f"num_players must be 2-6, got {num_players}")

        if num_players > 2:
            warnings.warn(
                f"Multi-way ({num_players} players) is supported but heads-up (2 players) "
                "is recommended for belief modeling research. Multi-way adds complexity "
                "for belief computation and evaluation.",
                UserWarning
            )

        self.num_players = num_players
        self.stacks = stacks if stacks else tuple([200] * num_players)
        self.blinds = blinds
        self.small_bet = small_bet
        self.big_bet = big_bet

        if len(self.stacks) != num_players:
            raise ValueError(f"stacks length {len(self.stacks)} must match num_players {num_players}")

        # Config for hashing
        self.config = EnvConfig(
            num_players=num_players,
            stacks=self.stacks,
            blinds=blinds,
            small_bet=small_bet,
            big_bet=big_bet,
        )

        # State tracking
        self.state = None
        self.deck: Optional[DeterministicDeck] = None
        self.hand_id: str = ""
        self.seed: int = 0
        self.initial_stacks = list(self.stacks)

        # Track dealing progress
        self._holes_dealt = 0
        self._board_dealt = 0

        # Track actions for manifest
        self._action_history: list[dict] = []
        self._dealt_holes: list[str] = []
        self._dealt_board: list[str] = []
        self._initial_deck_order: list[str] = []

        # Track raises per street for cap
        self._raises_this_street = 0
        self._last_street = "PREFLOP"

    def reset(
        self,
        seed: int,
        hole_cards: list[str | None] | None = None,
        board: Optional[str] = None,
    ) -> Obs:
        """
        Reset to a new hand.

        Args:
            seed: Random seed for deterministic dealing
            hole_cards: Optional list of explicit hole cards for each player.
                        e.g., ["AcAs", "KhKd", None, "QsQc"] - None means deal randomly
            board: Optional explicit board cards (e.g., "Jc3d5c4h9s")

        Returns:
            Observation for the first decision point
        """
        self.seed = seed
        self.hand_id = str(uuid.uuid4())[:8]
        self.initial_stacks = list(self.stacks)

        # Reset tracking
        self._action_history = []
        self._dealt_holes = []
        self._dealt_board = []
        self._raises_this_street = 0
        self._last_street = "PREFLOP"

        # Create deterministic deck
        self.deck = DeterministicDeck.from_seed(seed, self.num_players)
        self._initial_deck_order = self.deck.cards.copy()

        if hole_cards:
            self.deck.set_explicit_holes(hole_cards)
        if board:
            self.deck.set_explicit_board(board)

        # Reset dealing counters
        self._holes_dealt = 0
        self._board_dealt = 0

        # Create new game state
        self.state = FixedLimitTexasHoldem.create_state(
            AUTOMATIONS,
            True,  # ante_trimming_status
            0,  # antes (no ante)
            self.blinds,
            self.small_bet,
            self.big_bet,
            self.stacks,
            self.num_players,
        )

        # Advance to first decision point
        self._advance_until_decision()

        # Return observation for acting player
        return self.get_obs(self.current_player())

    def reset_from_manifest(self, manifest: HandManifest) -> Obs:
        """
        Reset to replay a hand exactly from a manifest.

        Args:
            manifest: HandManifest from export_manifest()

        Returns:
            Observation for the first decision point
        """
        # Verify config matches
        if manifest.env_config_hash != self.config.hash():
            warnings.warn(
                f"Manifest config hash {manifest.env_config_hash} doesn't match "
                f"current config hash {self.config.hash()}. Results may differ.",
                UserWarning
            )

        # Reset with explicit cards from manifest
        return self.reset(
            seed=manifest.seed,
            hole_cards=manifest.hole_cards if manifest.hole_cards else None,
            board="".join(manifest.board_cards) if manifest.board_cards else None,
        )

    def export_manifest(self) -> HandManifest:
        """
        Export a manifest that can reproduce this hand exactly.

        Call this after a hand completes to capture full state.

        Returns:
            HandManifest with all data needed for replay
        """
        return HandManifest(
            seed=self.seed,
            deck_order=self._initial_deck_order,
            hole_cards=self._dealt_holes,
            board_cards=self._dealt_board,
            actions=self._action_history,
            env_config=self.config.to_dict(),
            env_config_hash=self.config.hash(),
        )

    def step(self, action: Action) -> tuple[Obs, float, bool, dict]:
        """
        Apply an action and advance the game.

        Args:
            action: Action to apply

        Returns:
            Tuple of (observation, reward, done, info)
            - observation: Obs for the next decision point
            - reward: Stack delta for player 0 (0 if not terminal)
            - done: True if hand is complete
            - info: Additional information dict with 'rewards' for all players
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Track action for manifest
        self._action_history.append({
            "player": self.current_player(),
            "action": action.type.value,
            "street": self._get_street_name(),
        })

        # Track raises for cap
        current_street = self._get_street_name()
        if current_street != self._last_street:
            self._raises_this_street = 0
            self._last_street = current_street

        if action.type == ActionType.BET_OR_RAISE:
            self._raises_this_street += 1

        # Apply the action
        apply_action(self.state, action)

        # Advance to next decision point
        hand_active = self._advance_until_decision()

        # Check if hand is complete
        done = not hand_active

        # Calculate rewards (stack deltas for all players)
        reward = 0.0
        info = {"rewards": {}}

        if done:
            final_stacks = list(self.state.stacks)
            info["final_stacks"] = final_stacks

            # Calculate deltas for all players
            deltas = {}
            rewards_dict = {}
            for i in range(self.num_players):
                delta = float(final_stacks[i] - self.initial_stacks[i])
                deltas[f"player{i}_delta"] = delta
                rewards_dict[i] = delta

            info["deltas"] = deltas
            info["rewards"] = rewards_dict

            # Primary reward is player 0's delta
            reward = rewards_dict[0]

            # Include hole cards at showdown if available
            info["showdown"] = self._get_showdown_info()

        # Get observation for next player (or final state if done)
        player = self.current_player() if not done else 0
        obs = self.get_obs(player)

        return obs, reward, done, info

    def current_player(self) -> int:
        """
        Get the index of the player to act.

        Returns:
            Player index (0 to num_players-1), or -1 if no decision needed
        """
        if self.state is None:
            return -1
        return self.state.actor_index if self.state.actor_index is not None else -1

    def legal_actions(self) -> list[Action]:
        """
        Get list of legal actions for the current player.

        Returns:
            List of legal Action objects (BET_OR_RAISE excluded if cap reached)
        """
        if self.state is None:
            return []
        return get_legal_actions(self.state)

    def get_raises_remaining(self) -> int:
        """
        Get number of raises remaining this street.

        In fixed-limit, typically capped at 4 bets (1 bet + 3 raises).

        Returns:
            Number of raises still allowed this street
        """
        return max(0, MAX_RAISES_PER_STREET - self._raises_this_street)

    def get_bet_to_call(self) -> int:
        """
        Get the amount required to call.

        Returns:
            Chips required to call (0 if check is available)
        """
        if self.state is None:
            return 0

        player = self.current_player()
        if player < 0:
            return 0

        # Get current bets
        current_bets = list(self.state.bets) if self.state.bets else [0] * self.num_players
        max_bet = max(current_bets)
        player_bet = current_bets[player] if player < len(current_bets) else 0

        return max_bet - player_bet

    def get_button_position(self) -> int:
        """
        Get the button (dealer) position.

        In heads-up, button is small blind and acts first preflop.

        Returns:
            Player index of the button
        """
        # In PokerKit, player 0 is typically the button/dealer
        return 0

    def get_player_position(self, player_index: int) -> str:
        """
        Get position name for a player.

        Args:
            player_index: Player index

        Returns:
            Position name: "BTN", "SB", "BB", "UTG", "MP", "CO"
        """
        if self.num_players == 2:
            return "BTN/SB" if player_index == 0 else "BB"

        positions = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
        if player_index < len(positions):
            return positions[player_index]
        return f"P{player_index}"

    def get_contributions(self) -> tuple[list[int], list[int]]:
        """
        Get player contributions to the pot.

        Returns:
            Tuple of (contrib_this_round, contrib_total) for each player
        """
        if self.state is None:
            return ([0] * self.num_players, [0] * self.num_players)

        # Current round contributions (current bets)
        contrib_round = list(self.state.bets) if self.state.bets else [0] * self.num_players

        # Total contributions (initial stack - current stack + current bet)
        contrib_total = []
        for i in range(self.num_players):
            current_stack = self.state.stacks[i]
            current_bet = contrib_round[i] if i < len(contrib_round) else 0
            total = self.initial_stacks[i] - current_stack + current_bet
            contrib_total.append(total)

        return (contrib_round, contrib_total)

    def get_obs(self, player_index: int) -> Obs:
        """
        Get observation for a specific player.

        Args:
            player_index: Index of the player (0 to num_players-1)

        Returns:
            Obs object with public information visible to that player
        """
        if self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        contrib_round, contrib_total = self.get_contributions()

        return build_observation(
            state=self.state,
            player_index=player_index,
            hand_id=self.hand_id,
            seed=self.seed,
            legal_actions=self.legal_actions(),
            num_players=self.num_players,
            button=self.get_button_position(),
            position=self.get_player_position(player_index),
            bet_to_call=self.get_bet_to_call(),
            raises_remaining=self.get_raises_remaining(),
            contrib_this_round=contrib_round,
            contrib_total=contrib_total,
        )

    def get_hidden_state(self) -> dict:
        """
        Get hidden state for logging/oracle purposes.

        Returns:
            Dict with all players' hole cards and remaining deck
        """
        if self.state is None:
            return {}

        hidden = {}

        # Get all hole cards
        if self.state.hole_cards:
            for i, hole in enumerate(self.state.hole_cards):
                if hole:
                    # Use repr() for 2-char format (e.g., "8s" not "EIGHT OF SPADES")
                    hidden[f"player{i}_hole"] = [repr(c) for c in hole]

        # Get remaining deck cards
        if self.deck:
            hidden["remaining_deck"] = self.deck.remaining_cards()

        return hidden

    def get_config_hash(self) -> str:
        """Get deterministic hash of environment configuration."""
        return self.config.hash()

    def _advance_until_decision(self) -> bool:
        """
        Advance the game state through non-decision phases.

        Handles dealing and lets automations handle other phases
        (blinds, bet collection, showdown, etc.).

        Returns:
            True if hand is still active and a decision is needed,
            False if hand is complete.
        """
        while self.state.status:
            # Check if we need to deal hole cards
            if self._needs_hole_dealing():
                self._deal_hole_cards()
                continue

            # Check if we need to deal board cards
            if self._needs_board_dealing():
                self._deal_board_cards()
                continue

            # Check if a player decision is needed
            if self.state.actor_index is not None:
                # Verify there are legal actions available
                if self.legal_actions():
                    return True

            # No action needed and no dealing needed - check if state progressed
            # If not, the hand may be complete
            break

        return False

    def _needs_hole_dealing(self) -> bool:
        """Check if hole cards need to be dealt."""
        if self._holes_dealt >= self.num_players:
            return False

        # Check if PokerKit is waiting for hole dealing
        try:
            if hasattr(self.state, "can_deal_hole"):
                return self.state.can_deal_hole()
        except Exception:
            pass

        return False

    def _needs_board_dealing(self) -> bool:
        """Check if board cards need to be dealt."""
        try:
            if hasattr(self.state, "can_deal_board"):
                return self.state.can_deal_board()
        except Exception:
            pass

        return False

    def _deal_hole_cards(self) -> None:
        """Deal hole cards to the next player."""
        if self.deck is None:
            raise RuntimeError("Deck not initialized")

        # Get cards from deterministic deck
        cards = self.deck.get_hole_cards(self._holes_dealt)

        # Track for manifest
        self._dealt_holes.append(cards)

        # Deal to PokerKit state
        self.state.deal_hole(cards)
        self._holes_dealt += 1

    def _deal_board_cards(self) -> None:
        """Deal board cards (flop, turn, or river)."""
        if self.deck is None:
            raise RuntimeError("Deck not initialized")

        # Determine how many cards to deal
        current_board = len(self.state.board_cards)
        if current_board == 0:
            count = 3  # Flop
        else:
            count = 1  # Turn or River

        # Get cards from deterministic deck
        cards = self.deck.get_board_cards(count, current_board)

        # Track for manifest (parse into individual cards)
        from poker_env.deck import parse_cards
        self._dealt_board.extend(parse_cards(cards))

        # Deal to PokerKit state
        self.state.deal_board(cards)
        self._board_dealt += count

        # Reset raise counter for new street
        self._raises_this_street = 0

    def _get_showdown_info(self) -> dict:
        """Get information about the showdown."""
        info = {}

        if self.state.hole_cards:
            for i, hole in enumerate(self.state.hole_cards):
                if hole:
                    info[f"player{i}_hole"] = [repr(c) for c in hole]

        if self.state.board_cards:
            # Each element in board_cards is a list with one card
            info["board"] = [repr(cl[0]) for cl in self.state.board_cards if cl]

        return info

    def render_text(self) -> str:
        """
        Render the current state as text for debugging.

        Returns:
            Multi-line string representation of the game state
        """
        if self.state is None:
            return "Environment not initialized"

        lines = [
            f"=== Hand {self.hand_id} (seed: {self.seed}) ===",
            f"Players: {self.num_players}",
            f"Street: {self._get_street_name()}",
            f"Board: {[repr(cl[0]) for cl in self.state.board_cards if cl]}",
            f"Pot: {self.state.total_pot_amount}",
            f"Stacks: {list(self.state.stacks)}",
            f"Bets: {list(self.state.bets) if self.state.bets else []}",
            f"To act: Player {self.current_player()}",
            f"Bet to call: {self.get_bet_to_call()}",
            f"Raises remaining: {self.get_raises_remaining()}",
            f"Legal actions: {[a.type.value for a in self.legal_actions()]}",
        ]

        # Add hole cards (for debugging)
        if self.state.hole_cards:
            for i, hole in enumerate(self.state.hole_cards):
                if hole:
                    lines.append(f"Player {i} hole: {[repr(c) for c in hole]}")

        return "\n".join(lines)

    def _get_street_name(self) -> str:
        """Get current street name."""
        board_count = len(self.state.board_cards)
        if board_count == 0:
            return "PREFLOP"
        elif board_count == 3:
            return "FLOP"
        elif board_count == 4:
            return "TURN"
        else:
            return "RIVER"

    def _get_street_index(self) -> int:
        """Get current street as index (0=preflop, 1=flop, 2=turn, 3=river)."""
        board_count = len(self.state.board_cards)
        if board_count == 0:
            return 0
        elif board_count == 3:
            return 1
        elif board_count == 4:
            return 2
        else:
            return 3
