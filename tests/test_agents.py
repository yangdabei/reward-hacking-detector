"""
Tests for the RL agents.

TODO [SWE EXERCISE 6 — Test Suite]:
Implement all test functions below.
"""
import pytest


@pytest.fixture
def simple_env():
    """Create a simple 5x5 environment for fast testing."""
    # TODO [SWE EXERCISE 6]: Return a small GridWorld
    pytest.skip("Implement GridWorld first (ML Exercise 1)")


def test_q_table_update():
    """After one Q-learning update, Q-value should change correctly."""
    # TODO [SWE EXERCISE 6]:
    # agent = QLearningAgent(config, grid_size=7)
    # initial_q = agent.q_table[(0,0)][0]
    # agent.update(state=(0,0), action=0, reward=1.0, next_state=(0,1), done=False)
    # assert agent.q_table[(0,0)][0] != initial_q
    pass


def test_epsilon_decay():
    """Epsilon should decrease after each call to decay_epsilon()."""
    # TODO [SWE EXERCISE 6]:
    # agent = QLearningAgent(config, grid_size=7)
    # initial_eps = agent.epsilon
    # agent.decay_epsilon()
    # assert agent.epsilon < initial_eps
    # assert agent.epsilon >= config.epsilon_end
    pass


def test_greedy_policy_matches_q_values():
    """get_policy() should return argmax of Q-table for visited states."""
    # TODO [SWE EXERCISE 6]:
    # Manually set Q-table values, call get_policy(), verify argmax matches
    pass


def test_agent_save_load(tmp_path):
    """Saving and loading should produce identical Q-tables."""
    # TODO [SWE EXERCISE 6]:
    # agent = QLearningAgent(config, grid_size=7)
    # # Train briefly, save, load into new agent, compare q_tables
    pass


@pytest.mark.parametrize("lr", [0.01, 0.1, 0.5])
def test_q_learning_converges(lr, simple_env):
    """Q-learning should reach the goal reliably after training."""
    # TODO [SWE EXERCISE 6]:
    # agent = QLearningAgent(AgentConfig(learning_rate=lr), grid_size=5)
    # agent.train(simple_env, num_episodes=500)
    # Evaluate for 20 episodes, assert goal_rate > 0.8
    pass
