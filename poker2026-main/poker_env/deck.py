"""Deterministic deck handling for reproducible poker hands."""

import random
from typing import Optional
from dataclasses import dataclass, field


# Standard 52-card deck in canonical order
RANKS = "23456789TJQKA"
SUITS = "cdhs"  # clubs, diamonds, hearts, spades

FULL_DECK = [f"{r}{s}" for r in RANKS for s in SUITS]


@dataclass
class DeterministicDeck:
    """
    A deck that can be seeded for reproducible dealing.

    Supports two modes:
    1. Seeded random: Shuffle with a seed, deal cards in order
    2. Explicit cards: Specify exact cards for hole/board dealing

    Attributes:
        seed: Random seed for shuffling
        num_players: Number of players (for hole card dealing)
        cards: List of remaining cards in the deck
        dealt: List of cards that have been dealt
    """

    seed: int
    num_players: int = 2
    cards: list[str] = field(default_factory=list)
    dealt: list[str] = field(default_factory=list)
    _explicit_holes: list[Optional[str]] = field(default_factory=list)
    _explicit_board: Optional[str] = None

    def __post_init__(self):
        """Initialize and shuffle the deck with the given seed."""
        if not self.cards:
            self.cards = FULL_DECK.copy()
            rng = random.Random(self.seed)
            rng.shuffle(self.cards)

    def set_explicit_holes(self, hole_cards: list[str | None]) -> None:
        """
        Set explicit hole cards for players.

        Args:
            hole_cards: List of hole cards for each player.
                        e.g., ["AcAs", "KhKd", None, "QsQc"]
                        None means deal randomly for that player.
        """
        # Pad with None if fewer entries than players
        self._explicit_holes = list(hole_cards)
        while len(self._explicit_holes) < self.num_players:
            self._explicit_holes.append(None)

        # Remove explicit cards from deck if specified
        for hole in self._explicit_holes:
            if hole:
                for card in parse_cards(hole):
                    if card in self.cards:
                        self.cards.remove(card)

    def set_explicit_board(self, board: str) -> None:
        """
        Set explicit board cards.

        Args:
            board: Board cards (e.g., "Jc3d5c4h9s")
        """
        self._explicit_board = board

        # Remove explicit cards from deck
        if board:
            for card in parse_cards(board):
                if card in self.cards:
                    self.cards.remove(card)

    def get_hole_cards(self, player_index: int) -> str:
        """
        Get hole cards for a player.

        Returns explicit cards if set, otherwise deals from shuffled deck.

        Args:
            player_index: Player index (0 to num_players-1)

        Returns:
            Two-card string like "AcAs"
        """
        if self._explicit_holes and len(self._explicit_holes) > player_index:
            explicit = self._explicit_holes[player_index]
            if explicit:
                return explicit

        # Deal two cards from shuffled deck
        cards = [self.cards.pop(0) for _ in range(2)]
        self.dealt.extend(cards)
        return "".join(cards)

    def get_board_cards(self, count: int, current_board_len: int = 0) -> str:
        """
        Get board cards.

        Returns explicit cards if set, otherwise deals from shuffled deck.

        Args:
            count: Number of cards to deal (3 for flop, 1 for turn/river)
            current_board_len: Current number of board cards (to slice explicit)

        Returns:
            Card string like "Jc3d5c"
        """
        if self._explicit_board:
            explicit_cards = parse_cards(self._explicit_board)
            start = current_board_len
            end = start + count
            if end <= len(explicit_cards):
                return "".join(explicit_cards[start:end])

        # Deal from shuffled deck
        cards = [self.cards.pop(0) for _ in range(count)]
        self.dealt.extend(cards)
        return "".join(cards)

    def remaining_cards(self) -> list[str]:
        """Get list of cards still in the deck."""
        return self.cards.copy()

    @classmethod
    def from_seed(cls, seed: int, num_players: int = 2) -> "DeterministicDeck":
        """Create a new deck shuffled with the given seed."""
        return cls(seed=seed, num_players=num_players)


def parse_cards(card_string: str) -> list[str]:
    """
    Parse a card string into individual cards.

    Args:
        card_string: String like "AcAs" or "Jc3d5c"

    Returns:
        List of cards like ["Ac", "As"] or ["Jc", "3d", "5c"]
    """
    cards = []
    i = 0
    while i < len(card_string):
        if i + 1 < len(card_string):
            cards.append(card_string[i : i + 2])
            i += 2
        else:
            break
    return cards


def validate_card(card: str) -> bool:
    """
    Check if a card string is valid.

    Args:
        card: Two-character card string like "Ac"

    Returns:
        True if valid, False otherwise
    """
    if len(card) != 2:
        return False
    rank, suit = card[0], card[1]
    return rank in RANKS and suit in SUITS


def validate_cards(card_string: str) -> bool:
    """
    Validate a string of cards.

    Args:
        card_string: String like "AcAs" or "Jc3d5c"

    Returns:
        True if all cards are valid, False otherwise
    """
    cards = parse_cards(card_string)
    return all(validate_card(c) for c in cards)
