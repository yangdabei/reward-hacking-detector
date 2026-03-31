"""Deep Q-Network (DQN) agent."""

import logging
import pathlib
import random
from collections import deque
from typing import Any, Optional

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------


class QNetwork(nn.Module):
    """Multi-layer perceptron with 2 hidden layers for Q-value approximation.

    Maps a flat state representation to Q-values for each action.
    """

    def __init__(self, input_size: int, n_actions: int, hidden_size: int = 64) -> None:
        """Initialise the Q-network.

        Args:
            input_size: Dimension of the input state vector.
            n_actions: Number of discrete actions (output dimension).
            hidden_size: Width of each hidden layer (default 64).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Q-value tensor of shape (batch_size, n_actions).
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Experience replay
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Fixed-size experience replay buffer backed by a deque.

    Stores (state, action, reward, next_state, done) transitions and
    supports uniform random sampling for training mini-batches.
    """

    def __init__(self, maxlen: int = 10000) -> None:
        """Initialise the replay buffer.

        Args:
            maxlen: Maximum number of transitions to store (oldest are dropped).
        """
        self.buffer = deque(maxlen=maxlen)

    def add(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        """Add a transition to the buffer.

        Args:
            state: Observed state (e.g. one-hot numpy array).
            action: Integer action taken.
            reward: Scalar reward received.
            next_state: Next observed state.
            done: True if this transition ended the episode.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list:
        """Randomly sample a mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            List of (state, action, reward, next_state, done) tuples.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN agent
# ---------------------------------------------------------------------------


class DQNAgent:
    """DQN agent with a target network and experience replay.

    Implements the algorithm from Mnih et al. (2015):
      - Online network updated every step via TD loss.
      - Target network updated periodically (hard copy).
      - Experience replay for decorrelated mini-batch training.
    """

    def __init__(self, config: Any, grid_size: int, n_actions: int = 4) -> None:
        """Initialise the DQN agent.

        Args:
            config: AgentConfig with fields: epsilon_start, epsilon_end,
                    epsilon_decay, learning_rate, gamma, batch_size,
                    target_update_freq, replay_buffer_size.
            grid_size: Side length of the square grid.
            n_actions: Number of discrete actions (default 4).
        """
        self.config = config
        self.grid_size = grid_size
        self.n_actions = n_actions
        self.epsilon = config.epsilon_start
        self.steps_done = 0

        input_size = grid_size * grid_size
        self.online_net = QNetwork(input_size, n_actions)
        self.target_net = QNetwork(input_size, n_actions)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.buffer = ReplayBuffer(maxlen=config.replay_buffer_size)
        self.optimizer = t.optim.Adam(
            self.online_net.parameters(), lr=config.learning_rate
        )

    def select_action(self, state: tuple[int, int], grid_as_onehot: np.ndarray) -> int:
        """Epsilon-greedy action selection using the online Q-network.

        Args:
            state: Current (row, col) position.
            grid_as_onehot: Flat one-hot encoding of the current state.

        Returns:
            Selected action integer.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        state_tensor = t.tensor(grid_as_onehot, dtype=t.float32).unsqueeze(0)
        with t.no_grad():
            q_values = self.online_net(state_tensor)
        return int(q_values.argmax().item())

    def update(self, batch_size: int = 32) -> Optional[float]:
        """Sample a mini-batch and perform one gradient descent step.

        TD target: reward + gamma * max_a' Q_target(s', a')  (zero for terminal states)
        Loss: MSE(TD_target - Q_online(s, a))

        Args:
            batch_size: Mini-batch size.

        Returns:
            Scalar loss value, or None if the buffer has fewer than batch_size samples.
        """
        if len(self.buffer) < batch_size:
            return None

        batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = t.from_numpy(np.stack(states))
        next_states_t = t.from_numpy(np.stack(next_states))
        actions_t = t.tensor(actions, dtype=t.long)
        rewards_t = t.tensor(rewards, dtype=t.float32)
        dones_t = t.tensor(dones, dtype=t.float32)

        q_current = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with t.no_grad():
            q_next = self.target_net(next_states_t).max(1).values
            q_target = rewards_t + self.config.gamma * q_next * (1 - dones_t)

        loss = F.mse_loss(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def train(self, env: Any, num_episodes: int) -> list[float]:
        """Full DQN training loop.

        Args:
            env: Gymnasium-compatible environment.
            num_episodes: Number of training episodes.

        Returns:
            List of total rewards per episode.
        """
        rewards = []
        for episode in range(num_episodes):
            obs, info = env.reset()
            terminated, truncated = False, False
            episode_reward = 0.0

            while not terminated and not truncated:
                onehot = make_onehot(obs, self.grid_size)
                action = self.select_action(obs, onehot)
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_onehot = make_onehot(next_obs, self.grid_size)
                self.buffer.add(onehot, action, reward, next_onehot, terminated or truncated)
                self.update(self.config.batch_size)
                episode_reward += reward
                obs = next_obs

            self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
            rewards.append(episode_reward)

            if episode % 100 == 0:
                logger.info(
                    f"Episode {episode}/{num_episodes}, avg_reward={np.mean(rewards[-100:]):.2f}"
                )

        return rewards

    def save(self, path: pathlib.Path) -> None:
        """Save online network weights to *path*.

        Args:
            path: Destination .pt file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        t.save(self.online_net.state_dict(), path)

    def load(self, path: pathlib.Path) -> None:
        """Load weights from *path* into both online and target networks.

        Args:
            path: Source .pt file path.
        """
        state_dict = t.load(path)
        self.online_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def make_onehot(state: tuple[int, int], grid_size: int) -> np.ndarray:
    """Convert a (row, col) position to a flattened one-hot vector.

    Args:
        state: (row, col) grid position.
        grid_size: Side length of the square grid.

    Returns:
        1-D numpy float32 array of length grid_size * grid_size.
    """
    vec = np.zeros(grid_size * grid_size, dtype=np.float32)
    vec[state[0] * grid_size + state[1]] = 1.0
    return vec
