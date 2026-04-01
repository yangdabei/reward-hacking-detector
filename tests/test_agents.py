"""Tests for the RL agents."""

import pytest

from src.agents.q_learning import QLearningAgent
from src.config import AgentConfig, EnvConfig, RewardConfig
from src.environment.gridworld import GridWorld


@pytest.fixture
def config():
    """Default AgentConfig for tests."""
    return AgentConfig()


@pytest.fixture
def simple_env():
    """Create a simple 5x5 GridWorld for fast testing."""
    env_config = EnvConfig(
        grid_size=5,
        agent_start=(0, 0),
        goal_position=(4, 4),
        coin_position=None,
        coin_terminal=False,
        lava_positions=[],
        wall_positions=[],
        max_steps=100,
        rewards=RewardConfig(),
    )
    return GridWorld(config=env_config)


def test_q_table_update(config):
    """After one Q-learning update, the Q-value for the taken action should change."""
    agent = QLearningAgent(config, grid_size=7)
    initial_q = agent.q_table[(0, 0)][0]
    agent.update(state=(0, 0), action=0, reward=1.0, next_state=(0, 1), done=False)
    assert agent.q_table[(0, 0)][0] != initial_q


def test_epsilon_decay():
    """Epsilon should decrease after decay_epsilon() and never drop below epsilon_end."""
    config = AgentConfig(epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9)
    agent = QLearningAgent(config, grid_size=7)
    initial_eps = agent.epsilon
    agent.decay_epsilon()
    assert agent.epsilon < initial_eps
    assert agent.epsilon >= config.epsilon_end


def test_greedy_policy_matches_q_values(config):
    """get_policy() should return the argmax of the Q-table for each visited state."""
    agent = QLearningAgent(config, grid_size=7)
    agent.q_table[(0, 0)] = [0.1, 0.5, 0.2, 0.3]
    agent.q_table[(1, 1)] = [0.9, 0.1, 0.1, 0.1]
    policy = agent.get_policy()
    assert policy[(0, 0)] == 1  # argmax of [0.1, 0.5, 0.2, 0.3]
    assert policy[(1, 1)] == 0  # argmax of [0.9, 0.1, 0.1, 0.1]


def test_agent_save_load(config, tmp_path):
    """Saving and loading should produce identical Q-tables."""
    agent = QLearningAgent(config, grid_size=7)
    agent.q_table[(0, 0)] = [1.0, 2.0, 3.0, 4.0]
    agent.q_table[(1, 1)] = [0.5, 0.5, 0.5, 0.5]

    path = tmp_path / "q_table.json"
    agent.save(path)

    loaded = QLearningAgent(config, grid_size=7)
    loaded.load(path)

    assert loaded.q_table[(0, 0)] == agent.q_table[(0, 0)]
    assert loaded.q_table[(1, 1)] == agent.q_table[(1, 1)]


@pytest.mark.parametrize("lr", [0.01, 0.1, 0.5])
def test_q_learning_converges(lr, simple_env):
    """Q-learning should reach the goal reliably after sufficient training."""
    config = AgentConfig(learning_rate=lr, epsilon_end=0.05)
    agent = QLearningAgent(config, grid_size=5)
    agent.train(simple_env, num_episodes=500)

    agent.epsilon = 0.0  # greedy evaluation
    goal_count = 0
    for _ in range(20):
        obs, _ = simple_env.reset()
        terminated, truncated = False, False
        while not terminated and not truncated:
            action = agent.select_action(obs)
            obs, _, terminated, truncated, info = simple_env.step(action)
        if info["reached_goal"]:
            goal_count += 1

    assert goal_count / 20 >= 0.8
