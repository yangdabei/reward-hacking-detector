"""Optimal reference policy using BFS to find shortest path to goal. Ignores coins — represents a truly aligned agent."""

from __future__ import annotations

from collections import deque
import numpy as np
import logging

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
        """BFS from start to goal; returns dict mapping state -> optimal action.

        Args:
            start: Starting (row, col) position.
            grid: 2-D grid array. Cells with value -1 are walls.

        Returns:
            Dict mapping each reachable (row, col) to the first action that
            should be taken from that state to reach the goal optimally.
        """
        # parent[state] = (parent_state, action_taken_from_parent)
        parent: dict[tuple[int, int], tuple[tuple[int, int], int] | None] = {start: None}
        queue: deque[tuple[int, int]] = deque([start])

        while queue:
            current = queue.popleft()

            if current == self.goal_position:
                break

            row, col = current
            for action, (dr, dc) in self._DELTAS.items():
                nr, nc = row + dr, col + dc
                neighbour = (nr, nc)

                # Bounds check
                if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
                    continue
                # Wall check
                if grid[nr, nc] == -1:
                    continue
                # Already visited
                if neighbour in parent:
                    continue

                parent[neighbour] = (current, action)
                queue.append(neighbour)

        # Back-trace to assign first action from each state
        policy: dict[tuple[int, int], int] = {}

        for state, info in parent.items():
            if info is None:
                # This is the start; action will be computed by forward pass
                continue
            # Walk back to find the first step taken from `start`
            node = state
            while parent[node] is not None:
                prev_state, action = parent[node]
                if prev_state == start:
                    policy[start] = action
                    break
                node = prev_state

        # Forward BFS policy: for every reachable state, record the action
        # that continues toward goal.  Re-run a cleaner forward pass.
        policy = {}
        # second pass: for each reachable node record the action taken FROM that node
        # We need to reconstruct: for each state, what action do we take?
        # It's easier to trace backwards from goal.
        # Build reverse path map: state -> action to reach goal's direction
        # Actually the cleanest way: store, for each node, the action taken from
        # its *parent* to arrive there.  Then reconstruct first-step for each state.

        # action_to_reach[state] = action taken from parent to reach state
        action_to_reach: dict[tuple[int, int], int] = {}
        for state, info in parent.items():
            if info is not None:
                _, action = info
                action_to_reach[state] = action

        # For each reachable state, find the optimal action by tracing path to goal
        # and finding the first step FROM that state.
        # More efficient: build next_step[state] = next state on shortest path to goal.
        next_step: dict[tuple[int, int], tuple[int, int]] = {}
        for state, info in parent.items():
            if info is not None:
                parent_state, _ = info
                # parent_state -> state is one step; reverse: state's parent is parent_state
                # We want next_step going TOWARD goal, not away.
                pass

        # Simplest correct approach: for each state, trace forward along parent chain
        # to goal and record the first action.
        for state in parent:
            if state == self.goal_position:
                continue
            # Trace from state toward goal using parent pointers (reversed)
            # parent pointers go from child back toward start, so we need
            # to trace from goal back to state.
            # Build path from start to state first, then first step is policy[state].
            path_actions: list[int] = []
            node = state
            while parent[node] is not None:
                prev_state, action = parent[node]
                path_actions.append(action)
                node = prev_state
            # path_actions is reversed (from state back to start)
            # The last element is the action taken FROM start toward state
            # For state == start neighbour, len==1 and that action IS what we want
            # For deeper states, we want the action taken from `state` toward goal,
            # which is the reverse of the last action in path_actions... this is wrong.
            # We need a different approach.
            pass

        # Correct approach: build next_node_toward_goal for each state.
        # parent[child] = (parent_node, action_from_parent_to_child)
        # So from parent_node we take action_from_parent_to_child to reach child.
        # We want: from state, what action takes us one step closer to goal?
        # = the action taken from state to reach state's child on path to goal.
        # Build: for each state, its child on the path to goal.
        child_toward_goal: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
        for state, info in parent.items():
            if info is not None:
                parent_state, action = info
                # parent_state --action--> state
                child_toward_goal[parent_state] = (state, action)

        for state, (child, action) in child_toward_goal.items():
            policy[state] = action

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
