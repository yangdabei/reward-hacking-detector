"""Tests for the detection pipeline."""

import pytest

from src.detection.policy_divergence import compute_kl_divergence
from src.detection.trajectory_analyzer import TrajectoryAnalyzer


def test_kl_divergence_identical_policies():
    """KL divergence of identical policies should be 0."""
    policy = {(0, 0): 0, (0, 1): 1, (1, 0): 2}
    kl = compute_kl_divergence(policy, policy, list(policy.keys()))
    assert abs(kl) < 1e-6


def test_kl_divergence_opposite_policies():
    """Policies that always disagree should have high KL divergence."""
    reference = {(0, 0): 0, (0, 1): 0}   # always UP
    learned = {(0, 0): 1, (0, 1): 1}     # always DOWN
    kl = compute_kl_divergence(learned, reference, list(reference.keys()))
    assert kl > 0.5


def test_kl_divergence_skips_missing_states():
    """States absent from learned_policy should be skipped without error."""
    reference = {(0, 0): 0, (1, 1): 1, (2, 2): 2}
    learned = {(0, 0): 0}  # only one state
    kl = compute_kl_divergence(learned, reference, list(reference.keys()))
    assert isinstance(kl, float)


def test_extract_features_reaches_goal():
    """reached_goal should be True when the trajectory ends at the goal."""
    analyzer = TrajectoryAnalyzer(goal_position=(2, 2), coin_position=None)
    trajectory = [
        ((0, 0), 3, -0.1),
        ((0, 1), 3, -0.1),
        ((0, 2), 1, -0.1),
        ((1, 2), 1, -0.1),
        ((2, 2), 1, 10.0),
    ]
    features = analyzer.extract_features(trajectory)
    assert features.reached_goal is True
    assert features.total_reward == pytest.approx(10.0 - 0.4)
    assert features.path_length == 5
    assert features.distance_to_goal_final == 0.0


def test_extract_features_coin_visited():
    """visited_coin should be True when the coin position appears in the trajectory."""
    analyzer = TrajectoryAnalyzer(goal_position=(4, 4), coin_position=(1, 1))
    trajectory = [
        ((0, 0), 1, -0.1),
        ((1, 0), 3, -0.1),
        ((1, 1), 1, 5.0),   # coin collected here
        ((2, 1), 3, -0.1),
    ]
    features = analyzer.extract_features(trajectory)
    assert features.visited_coin is True
    assert features.distance_to_coin_min == 0.0


def test_extract_features_no_coin():
    """distance_to_coin_min should be inf when no coin is configured."""
    analyzer = TrajectoryAnalyzer(goal_position=(4, 4), coin_position=None)
    trajectory = [((0, 0), 1, -0.1), ((1, 0), 1, -0.1)]
    features = analyzer.extract_features(trajectory)
    assert features.visited_coin is False
    assert features.distance_to_coin_min == float("inf")
