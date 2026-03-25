"""
Tests for the GridWorld environment.

TODO [SWE EXERCISE 6 — Test Suite]:
Implement all test functions below. Run with: pytest tests/test_environment.py -v
"""
import pytest


# TODO [SWE EXERCISE 6]: Import GridWorld and configs once you've implemented them
# from src.environment.gridworld import GridWorld
# from src.environment.configs import TRAINING_DEFAULT, TEST_NO_COIN


@pytest.fixture
def training_env():
    """Create a training environment for testing."""
    # TODO [SWE EXERCISE 6]: Replace with:
    # return GridWorld(TRAINING_DEFAULT)
    pytest.skip("Implement GridWorld first (ML Exercise 1)")


def test_reset_returns_valid_observation(training_env):
    """reset() should return (obs, info) where obs is within observation_space."""
    # TODO [SWE EXERCISE 6]:
    # obs, info = training_env.reset()
    # assert training_env.observation_space.contains(obs)
    # assert isinstance(info, dict)
    # assert "reached_goal" in info
    pass


def test_step_rewards_correct(training_env):
    """step() to goal gives +10 reward, to lava gives -5 reward."""
    # TODO [SWE EXERCISE 6]:
    # Navigate to goal and check reward == 10.0
    # Or test with a config where goal is reachable in 1 step
    pass


def test_wall_collision(training_env):
    """Moving into a wall keeps agent in same position."""
    # TODO [SWE EXERCISE 6]:
    # Put agent adjacent to wall, move toward wall
    # Assert position didn't change
    pass


def test_episode_terminates_at_goal(training_env):
    """terminated=True when agent reaches goal."""
    # TODO [SWE EXERCISE 6]:
    # Navigate agent to goal position
    # Assert terminated == True in step() return
    pass


def test_episode_truncates_at_max_steps(training_env):
    """truncated=True when max_steps exceeded."""
    # TODO [SWE EXERCISE 6]:
    # Create env with max_steps=5
    # Take 5 steps without reaching goal
    # Assert truncated == True
    pass


def test_invalid_action_handled():
    """Actions outside 0-3 should raise an error."""
    # TODO [SWE EXERCISE 6]:
    # env = GridWorld(TRAINING_DEFAULT)
    # env.reset()
    # with pytest.raises((ValueError, AssertionError)):
    #     env.step(99)  # invalid action
    pass


def test_config_validation():
    """Invalid EnvConfig should raise ValueError."""
    # TODO [SWE EXERCISE 6]:
    # from src.config import EnvConfig
    # with pytest.raises(ValueError):
    #     EnvConfig(grid_size=2)  # too small
    pass
