"""Tests for the GridWorld environment."""

import pytest

from src.config import EnvConfig, RewardConfig
from src.environment.configs import TRAINING_DEFAULT
from src.environment.gridworld import GridWorld


@pytest.fixture
def training_env():
    """Create a training environment for testing."""
    return GridWorld(TRAINING_DEFAULT)


def test_reset_returns_valid_observation(training_env):
    """reset() should return (obs, info) where obs is within observation_space."""
    obs, info = training_env.reset()
    assert training_env.observation_space.contains(obs)
    assert isinstance(info, dict)
    assert "reached_goal" in info


def test_step_rewards_correct():
    """step() to the goal gives the goal reward; step reward is applied each move."""
    config = EnvConfig(
        grid_size=3,
        agent_start=(0, 0),
        goal_position=(0, 1),
        coin_position=None,
        coin_terminal=False,
        lava_positions=[],
        wall_positions=[],
        max_steps=10,
        rewards=RewardConfig(goal=10.0, step=-0.1),
    )
    env = GridWorld(config)
    env.reset()
    _, reward, terminated, _, _ = env.step(3)  # RIGHT — reaches goal
    assert reward == 10.0
    assert terminated is True


def test_wall_collision():
    """Moving into a wall keeps the agent in the same position."""
    config = EnvConfig(
        grid_size=5,
        agent_start=(0, 0),
        goal_position=(4, 4),
        coin_position=None,
        coin_terminal=False,
        lava_positions=[],
        wall_positions=[(0, 1)],
        max_steps=50,
        rewards=RewardConfig(),
    )
    env = GridWorld(config)
    env.reset()
    env.step(3)  # RIGHT — blocked by wall at (0,1)
    assert env.agent_pos == (0, 0)


def test_episode_terminates_at_goal():
    """terminated=True when the agent reaches the goal."""
    config = EnvConfig(
        grid_size=3,
        agent_start=(0, 0),
        goal_position=(0, 1),
        coin_position=None,
        coin_terminal=False,
        lava_positions=[],
        wall_positions=[],
        max_steps=10,
        rewards=RewardConfig(),
    )
    env = GridWorld(config)
    env.reset()
    _, _, terminated, _, _ = env.step(3)  # RIGHT — reaches goal
    assert terminated is True


def test_episode_truncates_at_max_steps():
    """truncated=True when max_steps is exceeded without reaching the goal."""
    config = EnvConfig(
        grid_size=5,
        agent_start=(0, 0),
        goal_position=(4, 4),
        coin_position=None,
        coin_terminal=False,
        lava_positions=[],
        wall_positions=[],
        max_steps=3,
        rewards=RewardConfig(),
    )
    env = GridWorld(config)
    env.reset()
    for _ in range(2):
        _, _, _, truncated, _ = env.step(0)  # UP — bounces off boundary
        assert not truncated
    _, _, _, truncated, _ = env.step(0)
    assert truncated is True


def test_out_of_bounds_action_ignored():
    """Stepping out of bounds leaves the agent in the same position."""
    env = GridWorld(TRAINING_DEFAULT)
    env.reset()
    env.step(0)  # UP from (0,0) — out of bounds, should stay
    assert env.agent_pos == (0, 0)


def test_config_validation():
    """Invalid EnvConfig should raise ValueError."""
    with pytest.raises(ValueError):
        EnvConfig(
            grid_size=2,  # below minimum of 3
            agent_start=(0, 0),
            goal_position=(1, 1),
            coin_position=None,
            coin_terminal=False,
            lava_positions=[],
            wall_positions=[],
            max_steps=10,
            rewards=RewardConfig(),
        )
