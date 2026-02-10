"""
Hand bucket definitions for belief modeling.

Defines semantic buckets that group poker hands by type and strength.
Used for tractable belief elicitation (vs full 1326 combos).

Supports multiple bucket schemes for ablation:
- buckets_14: Default semantic buckets (recommended)
- buckets_7: Coarse grouping (pairs, broadway, suited, offsuit, trash)
- buckets_30: Fine-grained (splits by high card rank)
"""

from itertools import combinations
from typing import Iterator
from dataclasses import dataclass

# All ranks and suits
RANKS = "AKQJT98765432"
SUITS = "shdc"  # spades, hearts, diamonds, clubs

# ============================================================================
# Bucket Definitions
# ============================================================================

# Each bucket maps to a list of hand patterns
# "s" suffix = suited, "o" suffix = offsuit, no suffix = pair
BUCKETS: dict[str, list[str]] = {
    # Premium pairs - monsters
    "premium_pairs": ["AA", "KK", "QQ"],
    
    # Strong pairs - still very strong
    "strong_pairs": ["JJ", "TT"],
    
    # Medium pairs - playable but vulnerable
    "medium_pairs": ["99", "88", "77", "66"],
    
    # Small pairs - set mining hands
    "small_pairs": ["55", "44", "33", "22"],
    
    # Premium broadway - top non-pair hands
    "premium_broadway": ["AKs", "AKo", "AQs"],
    
    # Strong broadway - strong unpaired hands
    "strong_broadway": ["AQo", "AJs", "KQs", "ATs"],
    
    # Medium broadway - playable broadway
    "medium_broadway": ["KQo", "KJs", "QJs", "AJo", "KTs", "QTs", "JTs"],
    
    # Suited aces - flush potential
    "suited_aces": ["A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s"],
    
    # Suited connectors - straight + flush potential
    "suited_connectors": ["T9s", "98s", "87s", "76s", "65s", "54s"],
    
    # Suited gappers - one-gap suited hands
    "suited_gappers": ["J9s", "T8s", "97s", "86s", "75s", "64s", "53s"],
    
    # Offsuit connectors - straight potential only
    "offsuit_connectors": ["T9o", "98o", "87o", "76o", "65o"],
    
    # Weak broadway - marginal broadway hands
    "weak_broadway": ["KTo", "QTo", "JTo", "K9s", "Q9s", "J9o", "K9o", "Q9o"],
    
    # Speculative suited - small suited cards
    "speculative_suited": [
        "43s", "42s", "32s",  # wheel potential
        "52s", "62s", "72s", "82s", "92s",  # suited with 2
        "63s", "73s", "83s", "93s",  # suited with 3
        "74s", "84s", "94s",  # suited with 4
        "85s", "95s",  # suited with 5
        "96s",  # suited with 6
    ],
    
    # Trash - everything else (computed dynamically)
    "trash": [],  # Will be filled in during initialization
}

# Ordered list of bucket names (for consistent iteration)
BUCKET_NAMES = [
    "premium_pairs",
    "strong_pairs", 
    "medium_pairs",
    "small_pairs",
    "premium_broadway",
    "strong_broadway",
    "medium_broadway",
    "suited_aces",
    "suited_connectors",
    "suited_gappers",
    "offsuit_connectors",
    "weak_broadway",
    "speculative_suited",
    "trash",
]


# ============================================================================
# Hand Parsing and Generation
# ============================================================================

def _parse_hand_pattern(pattern: str) -> tuple[str, str, str]:
    """
    Parse a hand pattern like "AKs" or "AA" into (rank1, rank2, suited_type).
    
    Returns:
        (rank1, rank2, type) where type is "pair", "suited", or "offsuit"
    """
    if len(pattern) == 2:
        # Pair: "AA", "KK", etc.
        return (pattern[0], pattern[1], "pair")
    elif len(pattern) == 3:
        if pattern[2] == "s":
            return (pattern[0], pattern[1], "suited")
        else:  # "o"
            return (pattern[0], pattern[1], "offsuit")
    else:
        raise ValueError(f"Invalid hand pattern: {pattern}")


def _generate_combos(pattern: str) -> list[tuple[str, str]]:
    """
    Generate all specific card combinations for a hand pattern.
    
    Args:
        pattern: Hand pattern like "AKs", "AKo", or "AA"
        
    Returns:
        List of (card1, card2) tuples, e.g., [("Ah", "Kh"), ("As", "Ks"), ...]
    """
    r1, r2, hand_type = _parse_hand_pattern(pattern)
    
    combos = []
    
    if hand_type == "pair":
        # 6 combinations for a pair
        for s1, s2 in combinations(SUITS, 2):
            combos.append((f"{r1}{s1}", f"{r2}{s2}"))
    elif hand_type == "suited":
        # 4 combinations for suited hands
        for s in SUITS:
            combos.append((f"{r1}{s}", f"{r2}{s}"))
    else:  # offsuit
        # 12 combinations for offsuit hands
        for s1 in SUITS:
            for s2 in SUITS:
                if s1 != s2:
                    combos.append((f"{r1}{s1}", f"{r2}{s2}"))
    
    return combos


def _get_all_possible_hands() -> set[tuple[str, str]]:
    """Generate all 1326 possible hold'em starting hands."""
    all_cards = [f"{r}{s}" for r in RANKS for s in SUITS]
    hands = set()
    for c1, c2 in combinations(all_cards, 2):
        # Normalize order (higher card first by rank, then by suit)
        if (RANKS.index(c1[0]), SUITS.index(c1[1])) > (RANKS.index(c2[0]), SUITS.index(c2[1])):
            hands.add((c2, c1))
        else:
            hands.add((c1, c2))
    return hands


def _normalize_hand(card1: str, card2: str) -> tuple[str, str]:
    """Normalize hand to consistent order (higher rank first)."""
    r1, r2 = RANKS.index(card1[0]), RANKS.index(card2[0])
    if r1 < r2:  # Lower index = higher rank
        return (card1, card2)
    elif r1 > r2:
        return (card2, card1)
    else:
        # Same rank, order by suit
        s1, s2 = SUITS.index(card1[1]), SUITS.index(card2[1])
        if s1 <= s2:
            return (card1, card2)
        else:
            return (card2, card1)


def _hand_to_pattern(card1: str, card2: str) -> str:
    """
    Convert specific cards to a hand pattern.
    
    Args:
        card1, card2: Cards like "Ah", "Kd"
        
    Returns:
        Pattern like "AKo", "AKs", or "AA"
    """
    r1, s1 = card1[0], card1[1]
    r2, s2 = card2[0], card2[1]
    
    # Order by rank (higher first)
    if RANKS.index(r1) > RANKS.index(r2):
        r1, r2 = r2, r1
        s1, s2 = s2, s1
    
    if r1 == r2:
        return f"{r1}{r2}"  # Pair
    elif s1 == s2:
        return f"{r1}{r2}s"  # Suited
    else:
        return f"{r1}{r2}o"  # Offsuit


# ============================================================================
# Build lookup tables
# ============================================================================

# Map from hand pattern to bucket
_PATTERN_TO_BUCKET: dict[str, str] = {}

# Map from bucket to all hand patterns in it
_BUCKET_TO_PATTERNS: dict[str, list[str]] = {name: [] for name in BUCKET_NAMES}

# Build mappings from defined buckets
for bucket_name, patterns in BUCKETS.items():
    if bucket_name == "trash":
        continue  # Handle trash bucket separately
    for pattern in patterns:
        _PATTERN_TO_BUCKET[pattern] = bucket_name
        _BUCKET_TO_PATTERNS[bucket_name].append(pattern)

# Generate all possible hand patterns and assign unassigned ones to trash
def _generate_all_patterns() -> set[str]:
    """Generate all 169 hand patterns (13 pairs + 78 suited + 78 offsuit)."""
    patterns = set()
    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            if i < j:
                patterns.add(f"{r1}{r2}s")
                patterns.add(f"{r1}{r2}o")
            elif i == j:
                patterns.add(f"{r1}{r2}")
    return patterns

_all_patterns = _generate_all_patterns()
_assigned_patterns = set(_PATTERN_TO_BUCKET.keys())
_trash_patterns = _all_patterns - _assigned_patterns

for pattern in _trash_patterns:
    _PATTERN_TO_BUCKET[pattern] = "trash"
    _BUCKET_TO_PATTERNS["trash"].append(pattern)

# Update BUCKETS with trash hands
BUCKETS["trash"] = sorted(_BUCKET_TO_PATTERNS["trash"])


# ============================================================================
# Public API
# ============================================================================

def hand_to_bucket(hole_cards: list[str], board: list[str] | None = None) -> str:
    """
    Map a specific hand to its bucket.
    
    Args:
        hole_cards: List of 2 cards, e.g., ["Ah", "Kd"]
        board: Optional board cards (not used in current implementation,
               but included for future hand-strength-based bucketing)
               
    Returns:
        Bucket name, e.g., "premium_broadway"
    """
    if len(hole_cards) != 2:
        raise ValueError(f"Expected 2 hole cards, got {len(hole_cards)}")
    
    pattern = _hand_to_pattern(hole_cards[0], hole_cards[1])
    return _PATTERN_TO_BUCKET.get(pattern, "trash")


def get_all_hands_in_bucket(bucket: str) -> list[tuple[str, str]]:
    """
    Get all possible card combinations in a bucket.
    
    Args:
        bucket: Bucket name
        
    Returns:
        List of (card1, card2) tuples
    """
    if bucket not in _BUCKET_TO_PATTERNS:
        raise ValueError(f"Unknown bucket: {bucket}")
    
    hands = []
    for pattern in _BUCKET_TO_PATTERNS[bucket]:
        hands.extend(_generate_combos(pattern))
    return hands


def get_valid_hands_for_bucket(
    bucket: str,
    blockers: list[str] | None = None,
) -> list[tuple[str, str]]:
    """
    Get all hands in a bucket that don't conflict with blockers.
    
    Args:
        bucket: Bucket name
        blockers: Cards that can't be in opponent's hand (hero's cards + board)
        
    Returns:
        List of valid (card1, card2) tuples
    """
    all_hands = get_all_hands_in_bucket(bucket)
    
    if blockers is None:
        return all_hands
    
    blocker_set = set(blockers)
    valid = []
    for c1, c2 in all_hands:
        if c1 not in blocker_set and c2 not in blocker_set:
            valid.append((c1, c2))
    
    return valid


def get_bucket_counts(blockers: list[str] | None = None) -> dict[str, int]:
    """
    Get the number of valid hands in each bucket.
    
    Args:
        blockers: Cards that can't be in opponent's hand
        
    Returns:
        Dict mapping bucket name to count of valid hands
    """
    return {
        bucket: len(get_valid_hands_for_bucket(bucket, blockers))
        for bucket in BUCKET_NAMES
    }


def get_bucket_prior(blockers: list[str] | None = None) -> dict[str, float]:
    """
    Get uniform prior probability over buckets (weighted by hand count).
    
    Args:
        blockers: Cards that can't be in opponent's hand
        
    Returns:
        Dict mapping bucket name to prior probability
    """
    counts = get_bucket_counts(blockers)
    total = sum(counts.values())
    
    if total == 0:
        # No valid hands (shouldn't happen)
        return {bucket: 0.0 for bucket in BUCKET_NAMES}
    
    return {bucket: count / total for bucket, count in counts.items()}


# ============================================================================
# Bucket Scheme Configurations (for ablation studies)
# ============================================================================

@dataclass
class BucketScheme:
    """Configuration for a bucket scheme."""
    name: str
    buckets: dict[str, list[str]]
    bucket_names: list[str]
    description: str


# 7-bucket coarse scheme
BUCKETS_7 = {
    "premium": ["AA", "KK", "QQ", "AKs", "AKo"],
    "strong_pairs": ["JJ", "TT", "99", "88", "77", "66"],
    "small_pairs": ["55", "44", "33", "22"],
    "broadway": [
        "AQs", "AQo", "AJs", "AJo", "ATs", "ATo",
        "KQs", "KQo", "KJs", "KJo", "KTs", "KTo",
        "QJs", "QJo", "QTs", "QTo", "JTs", "JTo",
    ],
    "suited": [],  # Computed: all suited hands not in above
    "offsuit_connectors": ["T9o", "98o", "87o", "76o", "65o", "54o"],
    "trash": [],  # Computed: everything else
}

BUCKET_NAMES_7 = [
    "premium", "strong_pairs", "small_pairs", 
    "broadway", "suited", "offsuit_connectors", "trash"
]


# 30-bucket fine-grained scheme (splits by high card)
BUCKETS_30 = {
    "AA": ["AA"],
    "KK": ["KK"],
    "QQ": ["QQ"],
    "JJ_TT": ["JJ", "TT"],
    "99_66": ["99", "88", "77", "66"],
    "55_22": ["55", "44", "33", "22"],
    "AK": ["AKs", "AKo"],
    "AQ": ["AQs", "AQo"],
    "AJ_AT": ["AJs", "AJo", "ATs", "ATo"],
    "A9_A2_suited": ["A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s"],
    "KQ": ["KQs", "KQo"],
    "KJ_KT": ["KJs", "KJo", "KTs", "KTo"],
    "K9_K2_suited": ["K9s", "K8s", "K7s", "K6s", "K5s", "K4s", "K3s", "K2s"],
    "QJ_QT": ["QJs", "QJo", "QTs", "QTo"],
    "Q9_Q2_suited": ["Q9s", "Q8s", "Q7s", "Q6s", "Q5s", "Q4s", "Q3s", "Q2s"],
    "JT": ["JTs", "JTo"],
    "J9_suited": ["J9s", "J8s", "J7s", "J6s", "J5s", "J4s", "J3s", "J2s"],
    "T9": ["T9s", "T9o"],
    "T8_suited": ["T8s", "T7s", "T6s", "T5s", "T4s", "T3s", "T2s"],
    "98": ["98s", "98o"],
    "97_suited": ["97s", "96s", "95s", "94s", "93s", "92s"],
    "87": ["87s", "87o"],
    "86_suited": ["86s", "85s", "84s", "83s", "82s"],
    "76": ["76s", "76o"],
    "75_suited": ["75s", "74s", "73s", "72s"],
    "65": ["65s", "65o"],
    "64_suited": ["64s", "63s", "62s"],
    "54_43": ["54s", "54o", "43s", "43o"],
    "low_suited": ["53s", "52s", "42s", "32s"],
    "trash": [],  # Computed
}

BUCKET_NAMES_30 = list(BUCKETS_30.keys())


def get_bucket_scheme(scheme: str = "default") -> BucketScheme:
    """
    Get a bucket scheme configuration.
    
    Args:
        scheme: One of "default" (14), "coarse" (7), "fine" (30)
        
    Returns:
        BucketScheme configuration
    """
    if scheme in ("default", "14", "buckets_14"):
        return BucketScheme(
            name="buckets_14",
            buckets=BUCKETS,
            bucket_names=BUCKET_NAMES,
            description="14 semantic buckets (default)",
        )
    elif scheme in ("coarse", "7", "buckets_7"):
        return BucketScheme(
            name="buckets_7",
            buckets=BUCKETS_7,
            bucket_names=BUCKET_NAMES_7,
            description="7 coarse buckets for quick analysis",
        )
    elif scheme in ("fine", "30", "buckets_30"):
        return BucketScheme(
            name="buckets_30",
            buckets=BUCKETS_30,
            bucket_names=BUCKET_NAMES_30,
            description="30 fine-grained buckets for detailed analysis",
        )
    else:
        raise ValueError(f"Unknown bucket scheme: {scheme}. Use 'default', 'coarse', or 'fine'")
