"""Tabular Q-learning agent. TODO: Implement the core methods (ML Exercise 2)."""

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

    def __init__(self, config: "AgentConfig", grid_size: int, n_actions: int = 4) -> None:
        """Initialise the Q-learning agent.

        Args:
            config: AgentConfig with fields epsilon_start, epsilon_end,
                    epsilon_decay, learning_rate (lr), gamma.
            grid_size: Side length of the square grid (used for state space info).
            n_actions: Number of discrete actions (default 4).
        """
        self.config = config
        self.grid_size = grid_size
        self.n_actions = n_actions
        self.q_table: dict[tuple, list[float]] = defaultdict(lambda: [0.0] * n_actions)
        self.epsilon: float = config.epsilon_start

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def select_action(self, state: tuple[int, int]) -> int:
        """Epsilon-greedy action selection.

        With probability epsilon, choose a random action; otherwise choose
        the action with the highest Q-value for the current state.

        Args:
            state: Current (row, col) grid position.

        Returns:
            Selected action integer in [0, n_actions).
        """
        # TODO [ML EXERCISE 2 — Q-Learning Agent]:
        # Implement epsilon-greedy action selection:
        # - With probability self.epsilon: return random action (random.randint(0, self.n_actions-1))
        # - Otherwise: return argmax of self.q_table[state]
        # Hint: use numpy.argmax
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
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
        # TODO [ML EXERCISE 2 — Q-Learning Agent]:
        # Implement the Q-learning update rule:
        #
        #   Q(s, a) <- Q(s, a) + lr * (reward + gamma * max_a' Q(s', a') - Q(s, a))
        #
        # Steps:
        # 1. current_q  = self.q_table[state][action]
        # 2. If done:  target = reward
        #    Else:     target = reward + self.config.gamma * max(self.q_table[next_state])
        # 3. td_error  = target - current_q
        # 4. self.q_table[state][action] += self.config.lr * td_error
        current_q = self.q_table[state][action]
        if done:
            target = reward
        else:
            target = reward + self.config.gamma * max(self.q_table[next_state])
        td_error = target - current_q
        self.q_table[state][action] += self.config.learning_rate * td_error

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
        # TODO [ML EXERCISE 2 — Q-Learning Agent]:
        # Implement the training loop:
        # 1. For each episode:
        #    a. obs, info = env.reset()
        #    b. Loop until terminated or truncated:
        #       - action = self.select_action(obs)
        #       - next_obs, reward, terminated, truncated, info = env.step(action)
        #       - self.update(obs, action, reward, next_obs, terminated or truncated)
        #       - obs = next_obs
        #    c. self.decay_epsilon()
        #    d. Track episode reward
        # 2. Every 100 episodes: logger.info(f"Episode {ep}/{num_episodes}, avg_reward={...:.2f}")
        # 3. Use logger.debug() for individual episode details
        # Return: list of episode total rewards
        rewards = []
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            terminated, truncated = False, False
            episode_reward = 0.0
            while not terminated and not truncated:
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                self.update(obs, action, reward, next_obs, (terminated or truncated))
                episode_reward += reward
                obs = next_obs
            self.decay_epsilon()
            rewards.append(episode_reward)

            if episode % 100 == 0:
                logger.info(f"Episode {episode}/{num_episodes}, avg_reward={np.average(rewards):.2f}")
        
        return rewards

    def get_policy(self) -> dict[tuple[int, int], int]:
        """Return the greedy policy (argmax Q) for all visited states.

        Returns:
            Dict mapping each visited (row, col) state to the greedy action.
        """
        # TODO [ML EXERCISE 2 — Q-Learning Agent]:
        # Implement get_policy():
        # - Iterate over self.q_table.keys()
        # - For each state, return int(np.argmax(self.q_table[state]))
        # Return the resulting dict.
        actions = {}
        for state in self.q_table.keys():
            actions[state] = int(np.argmax(self.q_table[state]))
        return actions

    # ------------------------------------------------------------------
    # Provided helper — no changes needed
    # ------------------------------------------------------------------

    def get_action_probabilities(
        self, state: tuple[int, int], epsilon: float = 0.05
    ) -> np.ndarray:
        """Return softened action probability vector for a given state.

        Places (1 - epsilon) on the greedy action and epsilon / 3 on each
        of the remaining actions.  Used for KL divergence computation.

        Args:
            state: Current (row, col) position.
            epsilon: Smoothing factor (default 0.05).

        Returns:
            4-element numpy array of action probabilities.
        """
        q_values = self.q_table[state]
        greedy_action = int(np.argmax(q_values))
        probs = np.full(self.n_actions, epsilon / (self.n_actions - 1))
        probs[greedy_action] = 1.0 - epsilon
        return probs

    # ------------------------------------------------------------------
    # Persistence — implement these for ML Exercise 2
    # ------------------------------------------------------------------

    def save(self, path: pathlib.Path) -> None:
        """Serialise the Q-table to a JSON file at *path*.

        Args:
            path: Destination file path (e.g. checkpoints/q_table.json).
        """
        # TODO [ML EXERCISE 2 — Q-Learning Agent]:
        # Serialise self.q_table to JSON.
        # Hint: JSON keys must be strings — convert tuple keys with str(key).
        # Steps:
        # 1. Build a plain dict: {str(state): q_values for state, q_values in self.q_table.items()}
        # 2. path.parent.mkdir(parents=True, exist_ok=True)
        # 3. path.write_text(json.dumps(serialisable, indent=2))
        serialisable = {str(state): q_values for state, q_values in self.q_table.items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(serialisable, indent=2))

    def load(self, path: pathlib.Path) -> None:
        """Load Q-table from a JSON file previously created by save().

        Args:
            path: Source file path.
        """
        # TODO [ML EXERCISE 2 — Q-Learning Agent]:
        # Deserialise the Q-table from JSON.
        # Hint: JSON keys are strings — convert back with ast.literal_eval(key).
        # Steps:
        # 1. data = json.loads(path.read_text())
        # 2. For each key, value in data.items():
        #    - Convert key string back to tuple, e.g. import ast; ast.literal_eval(key)
        #    - self.q_table[tuple_key] = value
        data = json.loads(path.read_text())
        for key, value in data.items():
            tuple_key = ast.literal_eval(key)
            self.q_table[tuple_key] = value
                
