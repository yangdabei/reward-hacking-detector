"""Deep Q-Network (DQN) agent. TODO: Implement (ML Exercise 3)."""

import numpy as np
from collections import deque
import random
import pathlib
import logging
from typing import Any, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
    F = None      # type: ignore[assignment]
    raise ImportError(
        "PyTorch is required for the DQN agent but was not found.\n"
        "Install it with:  pip install torch\n"
        "See https://pytorch.org/get-started/locally/ for platform-specific instructions."
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Multi-layer perceptron with 2 hidden layers for Q-value approximation.

    Maps a flat state representation to Q-values for each action.
    """

    # TODO [ML EXERCISE 3 — DQN Agent]:
    # Implement QNetwork(nn.Module):
    # - __init__(self, input_size: int, n_actions: int, hidden_size: int = 64)
    #   Layers: Linear(input_size, 64), ReLU, Linear(64, 64), ReLU, Linear(64, n_actions)
    # - forward(self, x: torch.Tensor) -> torch.Tensor
    #   Pass x through layers, return Q-values for each action

    def __init__(self, input_size: int, n_actions: int, hidden_size: int = 64) -> None:
        """Initialise the Q-network.

        Args:
            input_size: Dimension of the input state vector.
            n_actions: Number of discrete actions (output dimension).
            hidden_size: Width of each hidden layer (default 64).
        """
        super().__init__()
        raise NotImplementedError(
            "Implement QNetwork.__init__() — see TODO above"
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Q-value tensor of shape (batch_size, n_actions).
        """
        raise NotImplementedError(
            "Implement QNetwork.forward() — see TODO above"
        )


# ---------------------------------------------------------------------------
# Experience replay
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size experience replay buffer backed by a deque.

    Stores (state, action, reward, next_state, done) transitions and
    supports uniform random sampling for training mini-batches.
    """

    # TODO [ML EXERCISE 3 — DQN Agent]:
    # Implement ReplayBuffer:
    # - __init__(self, maxlen: int = 10000): use collections.deque(maxlen=maxlen)
    # - add(self, state, action, reward, next_state, done): append tuple to deque
    # - sample(self, batch_size: int) -> list: random.sample from deque
    # - __len__(self): return len(deque)

    def __init__(self, maxlen: int = 10000) -> None:
        """Initialise the replay buffer.

        Args:
            maxlen: Maximum number of transitions to store (oldest are dropped).
        """
        raise NotImplementedError(
            "Implement ReplayBuffer.__init__() — see TODO above"
        )

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
        raise NotImplementedError(
            "Implement ReplayBuffer.add() — see TODO above"
        )

    def sample(self, batch_size: int) -> list:
        """Randomly sample a mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            List of (state, action, reward, next_state, done) tuples.
        """
        raise NotImplementedError(
            "Implement ReplayBuffer.sample() — see TODO above"
        )

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        raise NotImplementedError(
            "Implement ReplayBuffer.__len__() — see TODO above"
        )


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

    # TODO [ML EXERCISE 3 — DQN Agent]:
    # Implement DQNAgent:
    # - __init__(self, config, grid_size, n_actions=4):
    #     Create QNetwork (online) and target network (copy)
    #     Create ReplayBuffer
    #     Create Adam optimizer
    # - select_action(self, state, grid_as_onehot) -> int: epsilon-greedy on Q-values
    # - update(self, batch_size=32) -> float | None:
    #     Sample from replay buffer, compute TD loss, backprop
    #     Loss: MSE((reward + gamma * max_a' Q_target(s', a')) - Q(s, a))
    # - train(self, env, num_episodes) -> list[float]: training loop
    # - save(self, path: Path): torch.save model weights
    # - load(self, path: Path): load model weights

    def __init__(self, config: Any, grid_size: int, n_actions: int = 4) -> None:
        """Initialise the DQN agent.

        Args:
            config: Configuration object with fields: epsilon_start, epsilon_end,
                    epsilon_decay, lr, gamma, batch_size, target_update_freq.
            grid_size: Side length of the square grid.
            n_actions: Number of discrete actions (default 4).
        """
        raise NotImplementedError(
            "Implement DQNAgent.__init__() — see TODO above"
        )

    def select_action(self, state: tuple[int, int], grid_as_onehot: np.ndarray) -> int:
        """Epsilon-greedy action selection using the online Q-network.

        Args:
            state: Current (row, col) position.
            grid_as_onehot: Flat one-hot encoding of the current state.

        Returns:
            Selected action integer.
        """
        raise NotImplementedError(
            "Implement DQNAgent.select_action() — see TODO above"
        )

    def update(self, batch_size: int = 32) -> Optional[float]:
        """Sample a mini-batch and perform one gradient descent step.

        TD target: reward + gamma * max_a' Q_target(s', a')  (zero for terminal states)
        Loss: MSE(TD_target - Q_online(s, a))

        Args:
            batch_size: Mini-batch size.

        Returns:
            Scalar loss value, or None if the buffer has fewer than batch_size samples.
        """
        raise NotImplementedError(
            "Implement DQNAgent.update() — see TODO above"
        )

    def train(self, env: Any, num_episodes: int) -> list[float]:
        """Full DQN training loop.

        Args:
            env: Gymnasium-compatible environment.
            num_episodes: Number of training episodes.

        Returns:
            List of total rewards per episode.
        """
        raise NotImplementedError(
            "Implement DQNAgent.train() — see TODO above"
        )

    def save(self, path: pathlib.Path) -> None:
        """Save online network weights to *path* using torch.save.

        Args:
            path: Destination .pt file path.
        """
        raise NotImplementedError(
            "Implement DQNAgent.save() — see TODO above"
        )

    def load(self, path: pathlib.Path) -> None:
        """Load online (and target) network weights from *path*.

        Args:
            path: Source .pt file path.
        """
        raise NotImplementedError(
            "Implement DQNAgent.load() — see TODO above"
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def make_onehot(state: tuple[int, int], grid_size: int) -> np.ndarray:
    """Convert a (row, col) position to a flattened one-hot vector.

    The vector has length grid_size * grid_size.  The element at index
    row * grid_size + col is set to 1.0; all others are 0.0.

    Args:
        state: (row, col) grid position.
        grid_size: Side length of the square grid.

    Returns:
        1-D numpy float32 array of length grid_size * grid_size.
    """
    # TODO [ML EXERCISE 3 — DQN Agent]:
    # Implement make_onehot:
    # 1. Create zeros array: vec = np.zeros(grid_size * grid_size, dtype=np.float32)
    # 2. Compute flat index: idx = state[0] * grid_size + state[1]
    # 3. Set vec[idx] = 1.0
    # 4. Return vec
    raise NotImplementedError("Implement make_onehot() — see TODO above")
