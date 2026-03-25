"""Trajectory analysis and feature extraction. TODO: Implement extract_features (ML Exercise 5)."""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryFeatures:
    """Seven scalar features summarising a single agent trajectory.

    Attributes:
        total_reward: Sum of all rewards received during the trajectory.
        path_length: Number of steps (transitions) in the trajectory.
        reached_goal: True if the final state equals the goal position.
        visited_coin: True if the coin position was visited at any step.
        distance_to_goal_final: Manhattan distance from the final state to
            the goal position.
        distance_to_coin_min: Minimum Manhattan distance to the coin
            across all steps.  Set to float('inf') when no coin exists.
        directness: Ratio of the straight-line (Manhattan) distance from
            start to goal divided by the actual path length.  A value of
            1.0 means the agent took the most direct route possible.
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
        # TODO [ML EXERCISE 5 — Trajectory Feature Extraction]:
        #
        # Trajectory format: list of (state, action, reward) tuples
        # where state = (row, col), action = int, reward = float
        #
        # Compute these features:
        # - total_reward: sum of all rewards in trajectory
        # - path_length: len(trajectory)
        # - reached_goal: True if final state == self.goal_position
        # - visited_coin: True if self.coin_position in [s for s,a,r in trajectory]
        # - distance_to_goal_final: Manhattan distance from final state to goal_position
        #   = abs(final_row - goal_row) + abs(final_col - goal_col)
        # - distance_to_coin_min: minimum Manhattan distance to coin during trajectory
        #   = min(abs(s[0]-coin[0]) + abs(s[1]-coin[1]) for s,a,r in trajectory)
        #   If no coin: set to float('inf')
        # - directness: (Manhattan distance start->goal) / path_length
        #   = (abs(start[0]-goal[0]) + abs(start[1]-goal[1])) / len(trajectory)
        #
        # Return a TrajectoryFeatures dataclass instance.
        raise NotImplementedError("Implement extract_features() — see TODO above")

    def extract_batch(self, trajectories: list) -> np.ndarray:
        """Extract features for a batch of trajectories.

        Args:
            trajectories: List of trajectories, each in the format accepted
                by extract_features().

        Returns:
            2-D numpy array of shape (len(trajectories), 7).
        """
        feature_arrays = [self.extract_features(traj).to_array() for traj in trajectories]
        return np.stack(feature_arrays, axis=0)

    def compute_proxy_reliance_direction(self, trajectory: list) -> float:
        """Compute the fraction of steps that move toward the coin.

        For each consecutive pair of states, the step is counted as
        "toward coin" if the Manhattan distance to the coin strictly
        decreased.  Returns a value in [0, 1].

        Args:
            trajectory: List of (state, action, reward) tuples.

        Returns:
            Fraction of steps that moved toward the coin.  Returns 0.0
            if there is no coin or the trajectory has fewer than 2 steps.
        """
        if self.coin_position is None or len(trajectory) < 2:
            return 0.0

        coin_row, coin_col = self.coin_position
        toward_coin_count = 0
        total_steps = len(trajectory) - 1  # number of transitions between consecutive states

        for i in range(total_steps):
            state_now, _, _ = trajectory[i]
            state_next, _, _ = trajectory[i + 1]

            dist_now = abs(state_now[0] - coin_row) + abs(state_now[1] - coin_col)
            dist_next = abs(state_next[0] - coin_row) + abs(state_next[1] - coin_col)

            if dist_next < dist_now:
                toward_coin_count += 1

        return toward_coin_count / total_steps if total_steps > 0 else 0.0
