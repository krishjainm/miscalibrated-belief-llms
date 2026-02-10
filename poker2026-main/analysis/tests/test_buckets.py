"""Tests for hand bucket functionality."""

import pytest
from analysis.buckets import (
    BUCKET_NAMES,
    BUCKETS,
    hand_to_bucket,
    get_all_hands_in_bucket,
    get_valid_hands_for_bucket,
    get_bucket_prior,
)


class TestBucketDefinitions:
    """Tests for bucket definitions."""
    
    def test_all_buckets_defined(self):
        """All bucket names have definitions."""
        for name in BUCKET_NAMES:
            assert name in BUCKETS
    
    def test_bucket_count(self):
        """Should have expected number of buckets."""
        assert len(BUCKET_NAMES) == 14
    
    def test_all_169_hands_covered(self):
        """All 169 starting hands should be in some bucket."""
        covered = set()
        for bucket, patterns in BUCKETS.items():
            covered.update(patterns)
        
        # 169 = 13 pairs + 78 suited + 78 offsuit
        assert len(covered) == 169


class TestHandToBucket:
    """Tests for hand_to_bucket mapping."""
    
    def test_premium_pairs(self):
        """Premium pairs should be identified."""
        assert hand_to_bucket(["Ah", "As"]) == "premium_pairs"
        assert hand_to_bucket(["Kd", "Kc"]) == "premium_pairs"
        assert hand_to_bucket(["Qh", "Qs"]) == "premium_pairs"
    
    def test_suited_connectors(self):
        """Suited connectors should be identified."""
        assert hand_to_bucket(["Ts", "9s"]) == "suited_connectors"
        assert hand_to_bucket(["8h", "7h"]) == "suited_connectors"
    
    def test_premium_broadway(self):
        """Premium broadway hands should be identified."""
        assert hand_to_bucket(["Ah", "Kh"]) == "premium_broadway"  # AKs
        assert hand_to_bucket(["Ac", "Kd"]) == "premium_broadway"  # AKo
    
    def test_suited_aces(self):
        """Suited aces should be identified."""
        assert hand_to_bucket(["As", "5s"]) == "suited_aces"
        assert hand_to_bucket(["Ah", "2h"]) == "suited_aces"
    
    def test_trash_hands(self):
        """Weak hands should go to trash."""
        # 72o is the classic worst hand
        assert hand_to_bucket(["7d", "2c"]) == "trash"


class TestGetHandsInBucket:
    """Tests for retrieving hands from buckets."""
    
    def test_premium_pairs_count(self):
        """Premium pairs should have 18 combos (3 pairs × 6 combos)."""
        hands = get_all_hands_in_bucket("premium_pairs")
        assert len(hands) == 18
    
    def test_suited_connectors_count(self):
        """Each suited hand has 4 combos."""
        hands = get_all_hands_in_bucket("suited_connectors")
        # 6 suited connectors × 4 suits
        assert len(hands) == 24
    
    def test_hands_are_tuples(self):
        """Hands should be (card1, card2) tuples."""
        hands = get_all_hands_in_bucket("premium_pairs")
        for hand in hands:
            assert isinstance(hand, tuple)
            assert len(hand) == 2


class TestBlockerFiltering:
    """Tests for blocker-aware hand retrieval."""
    
    def test_blocker_removes_hands(self):
        """Blockers should reduce available hands."""
        all_hands = get_all_hands_in_bucket("premium_pairs")
        
        # Holding an ace blocks some AA combos
        filtered = get_valid_hands_for_bucket("premium_pairs", ["Ah", "Kd"])
        assert len(filtered) < len(all_hands)
    
    def test_no_blockers_returns_all(self):
        """No blockers should return all hands."""
        all_hands = get_all_hands_in_bucket("premium_pairs")
        filtered = get_valid_hands_for_bucket("premium_pairs", [])
        assert len(filtered) == len(all_hands)
    
    def test_blocker_filters_correctly(self):
        """Specific blockers should filter specific combos."""
        # If we have Ah, opponent can't have any hand with Ah
        filtered = get_valid_hands_for_bucket("premium_pairs", ["Ah"])
        
        for hand in filtered:
            assert "Ah" not in hand


class TestBucketPrior:
    """Tests for bucket prior computation."""
    
    def test_prior_sums_to_one(self):
        """Prior should be valid probability distribution."""
        prior = get_bucket_prior()
        assert abs(sum(prior.values()) - 1.0) < 1e-6
    
    def test_all_buckets_have_prior(self):
        """All buckets should have prior probability."""
        prior = get_bucket_prior()
        for bucket in BUCKET_NAMES:
            assert bucket in prior
    
    def test_blockers_affect_prior(self):
        """Blockers should change bucket probabilities."""
        prior_no_blocker = get_bucket_prior()
        prior_with_blocker = get_bucket_prior(["Ah", "Kh"])
        
        # Should be different
        assert prior_no_blocker != prior_with_blocker
