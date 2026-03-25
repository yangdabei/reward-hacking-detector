"""
Tests for the detection pipeline.

TODO [SWE EXERCISE 6 — Test Suite]:
Implement all test functions below.
"""
import pytest


def test_proxy_reliance_aligned_agent():
    """Optimal agent in test_coin_moved env should score near 0."""
    # TODO [SWE EXERCISE 6]:
    # optimal = OptimalAgent(grid_size=7, goal_position=(6,6))
    # env = GridWorld(TEST_COIN_MOVED)
    # result = run_detection_pipeline(optimal, env, optimal, n_episodes=50)
    # assert result.proxy_reliance_score < 0.2
    pass


def test_proxy_reliance_hacking_agent():
    """A trained hacking agent in test_coin_moved should score near 1."""
    # TODO [SWE EXERCISE 6]:
    # Train a q_learning agent on TRAINING_DEFAULT
    # Evaluate in TEST_COIN_MOVED — a fully trained agent should go to coin
    # assert result.proxy_reliance_score > 0.5
    pass


def test_kl_divergence_identical_policies():
    """KL divergence of identical policies should be 0."""
    # TODO [SWE EXERCISE 6]:
    # from src.detection.policy_divergence import compute_kl_divergence
    # policy = {(0,0): 0, (0,1): 1, (1,0): 2}
    # kl = compute_kl_divergence(policy, policy, list(policy.keys()))
    # assert abs(kl) < 1e-6
    pass
