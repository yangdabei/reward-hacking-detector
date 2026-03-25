"""
Core Gymnasium-compatible gridworld environment for the Reward Hacking Detector project.

This module defines the GridWorld environment, a simple 2D grid navigation task
where an agent must reach a goal while potentially encountering coins, lava, and walls.
It follows the Gymnasium (gym) API so it can be used with standard RL libraries.

Note: The `config` type hint uses a string forward reference ("EnvConfig") since
config.py is a user TODO (ML Exercise 2). In the meantime, configs.py provides
a compatible SimpleEnvConfig dataclass that can be used as a drop-in replacement.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from src.config import EnvConfig

# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTIONS: dict[int, str] = {
    UP: "UP",
    DOWN: "DOWN",
    LEFT: "LEFT",
    RIGHT: "RIGHT",
}

# (row_delta, col_delta) for each action
ACTION_DELTAS: dict[int, tuple[int, int]] = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}

# ---------------------------------------------------------------------------
# Cell type constants
# ---------------------------------------------------------------------------

EMPTY = 0
WALL = -1
AGENT = 1
GOAL = 2
COIN = 3
LAVA = 4


# ---------------------------------------------------------------------------
# GridWorld environment
# ---------------------------------------------------------------------------


class GridWorld(gym.Env):
    """A simple 2D gridworld environment compatible with the Gymnasium API.

    The agent starts at a fixed position and must navigate to the goal.
    Along the way it may encounter:
      - Walls  — impassable cells; the agent stays in place.
      - Lava   — terminates the episode with a negative reward.
      - Coins  — optional intermediate rewards; may or may not end the episode.

    Observations are (row, col) tuples representing the agent's current position.
    Actions are discrete: UP (0), DOWN (1), LEFT (2), RIGHT (3).

    Configuration is provided via an EnvConfig-compatible object (see configs.py
    for the SimpleEnvConfig dataclass, or implement your own EnvConfig in
    config.py as part of ML Exercise 2).
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"]}

    def __init__(
        self,
        config: EnvConfig = EnvConfig,  # EnvConfig or SimpleEnvConfig
        render_mode: str | None = None,
    ) -> None:
        """Initialise the GridWorld environment.

        Args:
            config: An EnvConfig-compatible configuration object. Must expose at
                minimum the attributes used in step() and reset() (see TODO blocks).
            render_mode: One of "human", "rgb_array", "ansi", or None.
        """
        super().__init__()

        self.config = config
        self.render_mode = render_mode

        # Action space: 4 discrete actions (UP, DOWN, LEFT, RIGHT)
        self.action_space = gym.spaces.Discrete(4)

        # Observation space: (row, col) — each in [0, grid_size)
        grid_size: int = config.grid_size
        self.observation_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(grid_size),
                gym.spaces.Discrete(grid_size),
            )
        )

        # Instance state — properly initialised in reset()
        self.agent_pos: tuple[int, int] = config.agent_start
        self.grid: np.ndarray = self._build_grid()
        self.coin_collected: bool = False
        self.steps: int = 0

        # Logger
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_grid(self) -> np.ndarray:
        """Build and return a 2D numpy grid from the current config.

        Only static cell types (WALL, LAVA) are written into the array.
        Dynamic entities — agent, goal, and coin — are *not* placed here;
        they are tracked separately and overlaid on demand (see
        get_grid_with_entities()).

        Returns:
            A 2D int32 numpy array of shape (grid_size, grid_size) where each
            cell contains one of the cell-type constants defined in this module.
        """
        cfg = self.config
        grid = np.full(
            (cfg.grid_size, cfg.grid_size), fill_value=EMPTY, dtype=np.int32
        )

        for row, col in getattr(cfg, "wall_positions", []):
            grid[row, col] = WALL

        for row, col in getattr(cfg, "lava_positions", []):
            grid[row, col] = LAVA

        return grid

    def _get_obs(self) -> tuple[int, int]:
        """Return the current observation (agent position).

        Returns:
            A (row, col) tuple representing the agent's current position.
        """
        return self.agent_pos

    def _get_info(self) -> dict:
        """Return current environment state as an info dict.

        Returns:
            A dict with keys: reached_goal, picked_coin, steps.
        """
        info = {
            "reached_goal": self.agent_pos == self.config.goal_position,
            "picked_coin": self.coin_collected,
            "steps": self.steps,
        }
        return info

    def _is_valid_pos(self, pos: tuple[int, int]) -> bool:
        """Check whether (row, col) is within the grid boundaries.

        Does *not* check for walls or lava — those are handled in step().

        Args:
            row: Row index to check.
            col: Column index to check.

        Returns:
            True if the position is inside the grid, False otherwise.
        """
        size = self.config.grid_size
        row, col = pos[0], pos[1]
        return 0 <= row < size and 0 <= col < size

    # ------------------------------------------------------------------
    # Gymnasium API — reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[tuple[int, int], dict]:
        """Reset the environment to its initial state.

        Args:
            seed: Optional RNG seed passed to super().reset().
            options: Optional dict of extra options (currently unused).

        Returns:
            A (observation, info) tuple.
        """
        # Seed random number generator
        super().reset(seed=seed) 

        # Reset variables
        self.agent_pos = self.config.agent_start
        self.coin_collected = False
        self.steps = 0

        # Rebuild grid
        self._build_grid()
        
        observation = self._get_obs()
        info = self._get_info()

        self.logger.debug(f"Environment reset. Agent at {self.agent_pos}")

        return (observation, info)

    # ------------------------------------------------------------------
    # Gymnasium API — step
    # ------------------------------------------------------------------

    def step(
        self, action: int
    ) -> tuple[tuple[int, int], float, bool, bool, dict]:
        """Execute one step in the environment.

        # TODO [ML EXERCISE 1 — Gymnasium Environment]:
        # Implement the step method. It must:
        # 1. Increment self.steps
        # 2. Compute new position from action using ACTION_DELTAS[action]
        # 3. If new position hits a wall or is out of bounds: stay in place
        # 4. Update self.agent_pos to new position
        # 5. Compute reward:
        #    - Reach goal (config.goal_position): reward = config.reward_goal; terminated = True
        #    - Step on lava (in config.lava_positions): reward = config.reward_lava; terminated = True
        #    - Reach coin (config.coin_positions) and not coin_collected:
        #        reward = config.reward_coin; coin_collected = True
        #        If coin is terminal (config.coin_terminal): terminated = True
        #    - Otherwise: reward = config.reward_step
        # 6. Compute truncated = (self.steps >= config.max_steps)
        # 7. Build info dict: {"reached_goal": bool, "picked_coin": bool, "steps": int}
        # 8. Use logger.debug() to log the step
        # 9. Return (obs, reward, terminated, truncated, info)

        Args:
            action: Integer action in {UP, DOWN, LEFT, RIGHT}.

        Returns:
            A 5-tuple (observation, reward, terminated, truncated, info) as
            described in the TODO above.
        """
        
        self.steps += 1

        # Compute new position
        dr, dc = ACTION_DELTAS[action]
        new_pos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)

        # Stay in place if out of bounds or wall
        if self._is_valid_pos(new_pos) and new_pos not in self.config.wall_positions:
            self.agent_pos = new_pos

        # Compute reward and termination
        terminated = False
        reward = self.config.rewards.step

        if self.agent_pos == self.config.goal_position:
            reward = self.config.rewards.goal
            terminated = True
        elif self.agent_pos in self.config.lava_positions:
            reward = self.config.rewards.lava
            terminated = True
        elif (
            self.config.coin_position is not None
            and self.agent_pos == self.config.coin_position
            and not self.coin_collected
        ):
            reward = self.config.rewards.coin
            self.coin_collected = True
            if self.config.coin_terminal:
                terminated = True

        truncated = self.steps >= self.config.max_steps

        obs = self._get_obs()
        info = self._get_info()

        self.logger.debug(f"Step {self.steps}: action={action}, pos={self.agent_pos}, reward={reward:.2f}")

        return obs, reward, terminated, truncated, info


    # ------------------------------------------------------------------
    # Gymnasium API — render
    # ------------------------------------------------------------------

    def render(self) -> np.ndarray | str | None:
        """Render the environment according to self.render_mode.

        Modes:
            "ansi"      — Returns an ASCII string representation of the grid.
            "rgb_array" — Returns a placeholder numpy uint8 array. For full
                          visual rendering, use renderer.py's GridRenderer.
            None        — Returns None (no rendering).

        Returns:
            Rendered output depending on render_mode.
        """
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        return None

    def _render_ansi(self) -> str:
        """Return an ASCII string of the current grid state.

        Cell symbols:
            A — agent
            G — goal
            C — coin (uncollected)
            L — lava
            # — wall
            . — empty
        """
        cfg = self.config
        grid = self.get_grid_with_entities()
        rows: list[str] = []
        for r in range(cfg.grid_size):
            row_chars: list[str] = []
            for c in range(cfg.grid_size):
                cell = grid[r, c]
                if cell == AGENT:
                    row_chars.append("A")
                elif cell == GOAL:
                    row_chars.append("G")
                elif cell == COIN:
                    row_chars.append("C")
                elif cell == LAVA:
                    row_chars.append("L")
                elif cell == WALL:
                    row_chars.append("#")
                else:
                    row_chars.append(".")
            rows.append(" ".join(row_chars))
        return "\n".join(rows)

    def _render_rgb_array(self) -> np.ndarray:
        """Return a simple placeholder RGB array for the current grid.

        For full visual rendering with matplotlib, use GridRenderer in renderer.py.
        """
        cfg = self.config
        size = cfg.grid_size
        # Each cell is 10x10 pixels; 3 colour channels
        img = np.ones((size * 10, size * 10, 3), dtype=np.uint8) * 255
        return img

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_grid_with_entities(self) -> np.ndarray:
        """Return a copy of the base grid with agent, goal, and coin overlaid.

        The base grid (self.grid) only contains static cell types (WALL, LAVA,
        EMPTY). This method overlays the current positions of dynamic entities
        so the array can be used directly for rendering.

        Returns:
            A 2D int32 numpy array of shape (grid_size, grid_size) with
            AGENT, GOAL, and COIN values placed at their current positions.
        """
        overlay = self.grid.copy()

        # Place goal
        gr, gc = self.config.goal_position
        overlay[gr, gc] = GOAL

        # Place uncollected coins
        if not self.coin_collected:
            for cr, cc in getattr(self.config, "coin_positions", []):
                overlay[cr, cc] = COIN

        # Place agent (drawn last so it appears on top)
        ar, ac = self.agent_pos
        overlay[ar, ac] = AGENT

        return overlay
