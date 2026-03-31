"""Trajectory analysis and feature extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryFeatures:
    """Seven scalar features summarising a single agent trajectory.

    Attributes:
        total_reward: Sum of all rewards received during the trajectory.
        path_length: Number of steps in the trajectory.
        reached_goal: True if the final state equals the goal position.
        visited_coin: True if the coin position was visited at any step.
        distance_to_goal_final: Manhattan distance from the final state to the goal.
        distance_to_coin_min: Minimum Manhattan distance to the coin across all steps.
            Set to float('inf') when no coin exists.
        directness: Ratio of straight-line (Manhattan) start-to-goal distance divided
            by actual path length. 1.0 means the agent took the most direct route.
    """

    total_reward: float
    path_length: int
    reached_goal: bool
    visited_coin: bool
    distance_to_goal_final: float
    distance_to_coin_min: float
    directness: float

    def to_array(self) -> np.ndarray:
        """Return all features as a 1-D numpy float64 array (7 elements).

        Order matches the field declaration order in the dataclass.
        """
        return np.array(
            [
                self.total_reward,
                float(self.path_length),
                float(self.reached_goal),
                float(self.visited_coin),
                self.distance_to_goal_final,
                self.distance_to_coin_min,
                self.directness,
            ],
            dtype=np.float64,
        )

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return ordered list of feature names corresponding to to_array()."""
        return [
            "total_reward",
            "path_length",
            "reached_goal",
            "visited_coin",
            "distance_to_goal_final",
            "distance_to_coin_min",
            "directness",
        ]


class TrajectoryAnalyzer:
    """Extracts and analyses features from agent trajectories."""

    def __init__(
        self,
        goal_position: tuple[int, int],
        coin_position: tuple[int, int] | None,
        start_position: tuple[int, int] = (0, 0),
    ) -> None:
        """Initialise the analyser.

        Args:
            goal_position: (row, col) of the goal cell.
            coin_position: (row, col) of the coin cell, or None if absent.
            start_position: (row, col) of the agent's starting cell (default (0, 0)).
        """
        self.goal_position = goal_position
        self.coin_position = coin_position
        self.start_position = start_position

    def extract_features(
        self, trajectory: list[tuple[tuple[int, int], int, float]]
    ) -> TrajectoryFeatures:
        """Extract 7 scalar features from a single trajectory.

        Args:
            trajectory: List of (state, action, reward) tuples where
                state = (row, col), action = int, reward = float.

        Returns:
            A TrajectoryFeatures dataclass instance.
        """
        states, _, rewards = zip(*trajectory)
        states_arr = np.array(states)  # shape (n, 2)
        final_state = states[-1]

        gr, gc = self.goal_position
        sr, sc = self.start_position

        total_reward = float(np.sum(rewards))
        path_length = len(trajectory)
        reached_goal = final_state == self.goal_position
        distance_to_goal_final = float(abs(final_state[0] - gr) + abs(final_state[1] - gc))
        directness = (abs(sr - gr) + abs(sc - gc)) / path_length

        if self.coin_position is None:
            visited_coin = False
            distance_to_coin_min = float("inf")
        else:
            coin = np.array(self.coin_position)
            distances = np.abs(states_arr - coin).sum(axis=1)
            visited_coin = bool(distances.min() == 0)
            distance_to_coin_min = float(distances.min())

        return TrajectoryFeatures(
            total_reward=total_reward,
            path_length=path_length,
            reached_goal=reached_goal,
            visited_coin=visited_coin,
            distance_to_goal_final=distance_to_goal_final,
            distance_to_coin_min=distance_to_coin_min,
            directness=directness,
        )

    def extract_batch(self, trajectories: list) -> np.ndarray:
        """Extract features for a batch of trajectories.

        Args:
            trajectories: List of trajectories, each in the format accepted
                by extract_features().

        Returns:
            2-D numpy array of shape (len(trajectories), 7).
        """
        return np.stack([self.extract_features(traj).to_array() for traj in trajectories])

    def compute_proxy_reliance_direction(self, trajectory: list) -> float:
        """Compute the fraction of steps that move toward the coin.

        Args:
            trajectory: List of (state, action, reward) tuples.

        Returns:
            Fraction of steps that moved toward the coin. Returns 0.0
            if there is no coin or the trajectory has fewer than 2 steps.
        """
        if self.coin_position is None or len(trajectory) < 2:
            return 0.0

        states_arr = np.array([s for s, _, _ in trajectory])  # shape (n, 2)
        coin = np.array(self.coin_position)
        distances = np.abs(states_arr - coin).sum(axis=1)
        toward = int(np.sum(distances[1:] < distances[:-1]))
        return toward / (len(trajectory) - 1)
