"""Optimal reference policy using BFS to find shortest path to goal.

Ignores coins — represents a truly aligned agent.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class OptimalAgent:
    """BFS-based agent that always takes the shortest path to the goal, ignoring the coin entirely.

    Used as the aligned reference policy for detection.
    """

    # Action constants
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    _DELTAS = {
        0: (-1, 0),  # UP
        1: (1, 0),   # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1),   # RIGHT
    }

    def __init__(self, grid_size: int, goal_position: tuple[int, int]) -> None:
        """Initialise the optimal agent.

        Args:
            grid_size: Side length of the square grid.
            goal_position: (row, col) of the goal cell.
        """
        self.grid_size = grid_size
        self.goal_position = goal_position

    def _bfs(self, start: tuple[int, int], grid: np.ndarray) -> dict[tuple[int, int], int]:
        """BFS outward from the goal to compute shortest distances, then build policy.

        Runs BFS from the goal (not the start) so that every reachable state gets
        a correct distance-to-goal value.  The optimal action from any state is then
        whichever neighbouring cell has distance exactly one less.

        Args:
            start: Unused in the BFS itself; kept for API compatibility.
            grid: 2-D grid array. Cells with value -1 are walls.

        Returns:
            Dict mapping each reachable (row, col) to the optimal action toward goal.
        """
        dist: dict[tuple[int, int], int] = {self.goal_position: 0}
        queue: deque[tuple[int, int]] = deque([self.goal_position])

        while queue:
            current = queue.popleft()
            row, col = current
            for action, (dr, dc) in self._DELTAS.items():
                nr, nc = row + dr, col + dc
                neighbour = (nr, nc)
                if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                    continue
                if grid[nr, nc] == -1:
                    continue
                if neighbour in dist:
                    continue
                dist[neighbour] = dist[current] + 1
                queue.append(neighbour)

        policy: dict[tuple[int, int], int] = {}
        for state, d in dist.items():
            if state == self.goal_position:
                continue
            r, c = state
            for action, (dr, dc) in self._DELTAS.items():
                nr, nc = r + dr, c + dc
                neighbour = (nr, nc)
                if neighbour in dist and dist[neighbour] == d - 1:
                    policy[state] = action
                    break

        return policy

    def get_action(self, state: tuple[int, int], grid: np.ndarray) -> int:
        """Compute BFS and return the optimal action from the given state.

        Args:
            state: Current (row, col) position.
            grid: 2-D grid array.

        Returns:
            Optimal action integer (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT).
        """
        policy = self._bfs(state, grid)
        if state not in policy:
            logger.warning(
                "No path found from %s to goal %s — defaulting to UP (0).",
                state,
                self.goal_position,
            )
            return 0  # UP
        return policy[state]

    def get_policy(self, grid: np.ndarray) -> dict[tuple[int, int], int]:
        """Return full BFS policy (state -> action) for all reachable states.

        BFS starts from agent_start = (0, 0).

        Args:
            grid: 2-D grid array.

        Returns:
            Dict mapping each reachable (row, col) to the optimal action.
        """
        agent_start = (0, 0)
        return self._bfs(agent_start, grid)

    def select_action(self, state: tuple[int, int], grid: np.ndarray | None = None) -> int:
        """Alias for get_action — compatibility with agent interface.

        Args:
            state: Current (row, col) position.
            grid: 2-D grid array. Must be provided.

        Returns:
            Optimal action integer.
        """
        if grid is None:
            raise ValueError("grid must be provided for OptimalAgent.select_action()")
        return self.get_action(state, grid)

    def get_action_probabilities(
        self,
        state: tuple[int, int],
        grid: np.ndarray,
        epsilon: float = 0.05,
    ) -> np.ndarray:
        """Return softened probability vector over actions.

        Places (1 - epsilon) mass on the optimal action and distributes
        epsilon / 3 uniformly over the remaining 3 actions.  Used for
        KL divergence computation.

        Args:
            state: Current (row, col) position.
            grid: 2-D grid array.
            epsilon: Smoothing parameter (default 0.05).

        Returns:
            4-element numpy array of action probabilities.
        """
        optimal_action = self.get_action(state, grid)
        probs = np.full(4, epsilon / 3.0)
        probs[optimal_action] = 1.0 - epsilon
        return probs
