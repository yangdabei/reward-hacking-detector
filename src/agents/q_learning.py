"""Tabular Q-learning agent."""

from __future__ import annotations

import ast
import json
import logging
import pathlib
import random
from collections import defaultdict
from typing import Any

import numpy as np

from src.config import AgentConfig

logger = logging.getLogger(__name__)


class QLearningAgent:
    """Tabular Q-learning agent for grid-world environments.

    Maintains a Q-table Q(s, a) and learns via the Bellman update:

        Q(s, a) <- Q(s, a) + lr * (reward + gamma * max_a' Q(s', a') - Q(s, a))

    Exploration is handled with an epsilon-greedy strategy where epsilon
    decays over training.

    References:
        Watkins & Dayan (1992) — Q-learning, Machine Learning 8(3-4), 279-292.
    """

    def __init__(self, config: AgentConfig, grid_size: int, n_actions: int = 4) -> None:
        """Initialise the Q-learning agent.

        Args:
            config: AgentConfig with fields epsilon_start, epsilon_end,
                    epsilon_decay, learning_rate, gamma.
            grid_size: Side length of the square grid.
            n_actions: Number of discrete actions (default 4).
        """
        self.config = config
        self.grid_size = grid_size
        self.n_actions = n_actions
        self.q_table: dict[tuple, list[float]] = defaultdict(lambda: [0.0] * n_actions)
        self.epsilon: float = config.epsilon_start

    def select_action(self, state: tuple[int, int]) -> int:
        """Epsilon-greedy action selection.

        Args:
            state: Current (row, col) grid position.

        Returns:
            Selected action integer in [0, n_actions).
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int],
        done: bool,
    ) -> None:
        """Q-learning (off-policy TD) update rule.

        Applies the Bellman equation:

            Q(s, a) <- Q(s, a) + lr * (reward + gamma * max_a' Q(s', a') - Q(s, a))

        When done is True, the bootstrap term is omitted (terminal state).

        Args:
            state: State s — (row, col).
            action: Action a taken at state s.
            reward: Scalar reward received.
            next_state: State s' reached after action a.
            done: True if the episode ended after this transition.
        """
        current_q = self.q_table[state][action]
        target = reward if done else reward + self.config.gamma * max(self.q_table[next_state])
        self.q_table[state][action] += self.config.learning_rate * (target - current_q)

    def decay_epsilon(self) -> None:
        """Multiply epsilon by the decay rate and clip to epsilon_end."""
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    def train(self, env: Any, num_episodes: int) -> list[float]:
        """Run the full Q-learning training loop.

        Args:
            env: Gymnasium-compatible environment.
            num_episodes: Total number of episodes to train for.

        Returns:
            List of total rewards, one entry per episode.
        """
        rewards = []
        for episode in range(num_episodes):
            obs, info = env.reset()
            terminated, truncated = False, False
            episode_reward = 0.0
            while not terminated and not truncated:
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                self.update(obs, action, reward, next_obs, terminated or truncated)
                episode_reward += reward
                obs = next_obs
            self.decay_epsilon()
            rewards.append(episode_reward)
            if episode % 100 == 0:
                logger.info(
                    f"Episode {episode}/{num_episodes}, avg_reward={np.mean(rewards[-100:]):.2f}"
                )
        return rewards

    def get_policy(self) -> dict[tuple[int, int], int]:
        """Return the greedy policy (argmax Q) for all visited states.

        Returns:
            Dict mapping each visited (row, col) state to the greedy action.
        """
        return {state: int(np.argmax(q)) for state, q in self.q_table.items()}

    def get_action_probabilities(
        self, state: tuple[int, int], epsilon: float = 0.05
    ) -> np.ndarray:
        """Return softened action probability vector for a given state.

        Places (1 - epsilon) on the greedy action and epsilon / (n_actions - 1)
        on each remaining action. Used for KL divergence computation.

        Args:
            state: Current (row, col) position.
            epsilon: Smoothing factor (default 0.05).

        Returns:
            4-element numpy array of action probabilities.
        """
        greedy_action = int(np.argmax(self.q_table[state]))
        probs = np.full(self.n_actions, epsilon / (self.n_actions - 1))
        probs[greedy_action] = 1.0 - epsilon
        return probs

    def save(self, path: pathlib.Path) -> None:
        """Serialise the Q-table to a JSON file at *path*.

        Args:
            path: Destination file path (e.g. checkpoints/q_table.json).
        """
        serialisable = {str(state): q for state, q in self.q_table.items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(serialisable, indent=2))

    def load(self, path: pathlib.Path) -> None:
        """Load Q-table from a JSON file previously created by save().

        Args:
            path: Source file path.
        """
        data = json.loads(path.read_text())
        for key, value in data.items():
            self.q_table[ast.literal_eval(key)] = value
