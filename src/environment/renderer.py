"""
Matplotlib-based grid renderer for the Reward Hacking Detector project.

This module provides GridRenderer, a class that produces publication-quality
matplotlib figures of GridWorld environments.  It supports single-frame renders,
trajectory overlays, state-visitation heatmaps, and side-by-side comparison
plots for aligned vs reward-hacking agents.

Typical usage::

    from src.environment.configs import get_config
    from src.environment.renderer import GridRenderer

    renderer = GridRenderer(cell_size=80)
    fig = renderer.render_grid(get_config("training_default"), agent_pos=(0, 0))
    renderer.save_figure(fig, pathlib.Path("output/grid.png"))
"""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, Any

import matplotlib.axes
import matplotlib.figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

if TYPE_CHECKING:
    from src.environment.configs import SimpleEnvConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cell-type constants (mirrored from gridworld.py to avoid circular import)
# ---------------------------------------------------------------------------

_EMPTY = 0
_WALL = -1
_AGENT = 1
_GOAL = 2
_COIN = 3
_LAVA = 4

# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------

_COLORS: dict[str, str] = {
    "empty": "#FFFFFF",   # white
    "wall": "#5A5A5A",    # dark grey
    "goal": "#27AE60",    # green
    "coin": "#F1C40F",    # gold
    "lava": "#E74C3C",    # red
    "agent": "#2980B9",   # blue
    "path": "#85C1E9",    # light blue (path line)
    "grid_line": "#CCCCCC",
}

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_grid_array(config: "SimpleEnvConfig") -> np.ndarray:
    """Build a 2D integer numpy array from a SimpleEnvConfig.

    Only static cell types (walls and lava) are encoded.  Dynamic entities
    (agent, goal, coin) are *not* included — overlay them separately when
    needed.

    Args:
        config: A SimpleEnvConfig (or compatible) object.

    Returns:
        A 2D int32 array of shape (grid_size, grid_size).
    """
    grid = np.zeros((config.grid_size, config.grid_size), dtype=np.int32)
    for r, c in getattr(config, "wall_positions", []):
        grid[r, c] = _WALL
    for r, c in getattr(config, "lava_positions", []):
        grid[r, c] = _LAVA
    return grid


# ---------------------------------------------------------------------------
# GridRenderer
# ---------------------------------------------------------------------------


class GridRenderer:
    """Matplotlib-based renderer for GridWorld environments.

    Produces figures showing the grid layout, agent position, entity labels,
    optional path overlays, and state-visitation heatmaps.

    Args:
        cell_size: Logical size of each grid cell in pixels (used to determine
            figure dimensions).  Defaults to 80.
    """

    def __init__(self, cell_size: int = 80) -> None:
        self.cell_size = cell_size
        self.colors = dict(_COLORS)  # local mutable copy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_grid(
        self,
        config: "SimpleEnvConfig",
        agent_pos: tuple[int, int] | None = None,
        path: list[tuple[int, int]] | None = None,
        title: str = "GridWorld",
        ax: matplotlib.axes.Axes | None = None,
    ) -> matplotlib.figure.Figure:
        """Render the gridworld as a matplotlib figure.

        Draws each cell with its appropriate background colour, then overlays
        text labels for the goal (G), coins (C), and lava (L), a filled circle
        for the agent, and an optional path polyline.

        Args:
            config: Environment configuration defining the grid layout.
            agent_pos: (row, col) of the agent.  If None the agent is not drawn.
            path: Optional list of (row, col) positions to draw as a line.
            title: Figure / axes title string.
            ax: Existing matplotlib Axes to draw into.  If None a new figure
                and axes are created.

        Returns:
            The matplotlib Figure containing the render.
        """
        size = config.grid_size
        fig, ax = self._get_fig_ax(ax, size)

        grid = make_grid_array(config)

        # Draw cell backgrounds
        for r in range(size):
            for c in range(size):
                cell = grid[r, c]
                color = self._cell_color(cell)
                rect = mpatches.FancyBboxPatch(
                    (c, size - 1 - r),
                    1,
                    1,
                    boxstyle="square,pad=0",
                    facecolor=color,
                    edgecolor=self.colors["grid_line"],
                    linewidth=0.5,
                )
                ax.add_patch(rect)

        # Draw goal
        gr, gc = config.goal_position
        self._draw_label(ax, gr, gc, "G", size, color=self.colors["goal"], fontsize=14)

        # Draw coins
        for cr, cc in getattr(config, "coin_positions", []):
            self._draw_label(ax, cr, cc, "C", size, color=self.colors["coin"], fontsize=14)

        # Draw lava labels (background already coloured, add text for clarity)
        for lr, lc in getattr(config, "lava_positions", []):
            self._draw_label(ax, lr, lc, "L", size, color="#FFFFFF", fontsize=11)

        # Draw path
        if path and len(path) > 1:
            path_cols = [c + 0.5 for _, c in path]
            path_rows = [size - 1 - r + 0.5 for r, _ in path]
            ax.plot(
                path_cols,
                path_rows,
                color=self.colors["path"],
                linewidth=2,
                zorder=3,
                alpha=0.8,
            )

        # Draw agent
        if agent_pos is not None:
            ar, ac = agent_pos
            circle = plt.Circle(
                (ac + 0.5, size - 1 - ar + 0.5),
                0.35,
                color=self.colors["agent"],
                zorder=4,
            )
            ax.add_patch(circle)

        self._finalise_ax(ax, size, title)
        logger.debug("render_grid: config grid_size=%d title=%r", size, title)
        return fig

    def render_path(
        self,
        config: "SimpleEnvConfig",
        trajectory: list[tuple[tuple[int, int], int, float]],
        title: str = "Agent Trajectory",
        ax: matplotlib.axes.Axes | None = None,
    ) -> matplotlib.figure.Figure:
        """Render the gridworld with a full agent trajectory overlaid.

        Args:
            config: Environment configuration.
            trajectory: A list of (position, action, reward) tuples recorded
                during an episode.  Each ``position`` is a (row, col) tuple.
            title: Figure title.
            ax: Optional existing Axes.

        Returns:
            The matplotlib Figure.
        """
        path = [pos for pos, _action, _reward in trajectory]
        agent_pos = path[-1] if path else None
        logger.debug("render_path: %d steps", len(path))
        return self.render_grid(config, agent_pos=agent_pos, path=path, title=title, ax=ax)

    def render_heatmap(
        self,
        config: "SimpleEnvConfig",
        visit_counts: dict[tuple[int, int], int],
        title: str = "State Visitation Heatmap",
        ax: matplotlib.axes.Axes | None = None,
    ) -> matplotlib.figure.Figure:
        """Render a heatmap of state-visitation counts overlaid on the grid.

        Args:
            config: Environment configuration.
            visit_counts: Mapping from (row, col) to visit frequency.
            title: Figure title.
            ax: Optional existing Axes.

        Returns:
            The matplotlib Figure.
        """
        size = config.grid_size
        fig, ax = self._get_fig_ax(ax, size)

        # Build density matrix
        density = np.zeros((size, size), dtype=float)
        max_count = max(visit_counts.values()) if visit_counts else 1
        for (r, c), count in visit_counts.items():
            density[r, c] = count / max_count

        # Custom colormap: white → deep blue
        heatmap_cmap = LinearSegmentedColormap.from_list(
            "visit_heat", ["#FFFFFF", "#1A5276"], N=256
        )

        # Draw base grid cells, tinting by density
        base_grid = make_grid_array(config)
        for r in range(size):
            for c in range(size):
                cell = base_grid[r, c]
                if cell == _WALL:
                    color = self.colors["wall"]
                elif cell == _LAVA:
                    color = self.colors["lava"]
                else:
                    rgba = heatmap_cmap(density[r, c])
                    color = rgba
                rect = mpatches.FancyBboxPatch(
                    (c, size - 1 - r),
                    1,
                    1,
                    boxstyle="square,pad=0",
                    facecolor=color,
                    edgecolor=self.colors["grid_line"],
                    linewidth=0.5,
                )
                ax.add_patch(rect)

        # Overlay entity labels
        gr, gc = config.goal_position
        self._draw_label(ax, gr, gc, "G", size, color=self.colors["goal"], fontsize=14)
        for cr, cc in getattr(config, "coin_positions", []):
            self._draw_label(ax, cr, cc, "C", size, color=self.colors["coin"], fontsize=14)

        # Colourbar
        sm = plt.cm.ScalarMappable(cmap=heatmap_cmap, norm=plt.Normalize(0, max_count))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="Visit count")

        self._finalise_ax(ax, size, title)
        logger.debug("render_heatmap: %d unique cells visited", len(visit_counts))
        return fig

    def render_comparison(
        self,
        config_train: "SimpleEnvConfig",
        config_test: "SimpleEnvConfig",
        traj_aligned: list,
        traj_hacking: list,
        save_path: pathlib.Path | None = None,
    ) -> matplotlib.figure.Figure:
        """Create a 2×2 comparison figure.

        Layout::

            [Aligned  | Training env]   [Hacking | Training env]
            [Aligned  | Test env    ]   [Hacking | Test env    ]

        Args:
            config_train: Training environment configuration.
            config_test: Test environment configuration.
            traj_aligned: Trajectory list for the aligned agent (same format as
                render_path's ``trajectory`` argument).
            traj_hacking: Trajectory list for the reward-hacking agent.
            save_path: If provided, save the figure here.

        Returns:
            The matplotlib Figure.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(
            "Aligned Agent vs Reward-Hacking Agent\nTraining (top) and Test (bottom) Environments",
            fontsize=13,
            fontweight="bold",
        )

        aligned_path_train = [pos for pos, _, _ in traj_aligned] if traj_aligned else []
        hacking_path_train = [pos for pos, _, _ in traj_hacking] if traj_hacking else []

        # Row 0: training environment
        self.render_grid(
            config_train,
            agent_pos=aligned_path_train[-1] if aligned_path_train else None,
            path=aligned_path_train,
            title="Aligned Agent — Training",
            ax=axes[0, 0],
        )
        self.render_grid(
            config_train,
            agent_pos=hacking_path_train[-1] if hacking_path_train else None,
            path=hacking_path_train,
            title="Hacking Agent — Training",
            ax=axes[0, 1],
        )

        # Row 1: test environment (trajectories may not perfectly transfer, so
        # show the path positions clipped to the test grid)
        self.render_grid(
            config_test,
            agent_pos=aligned_path_train[-1] if aligned_path_train else None,
            path=aligned_path_train,
            title="Aligned Agent — Test",
            ax=axes[1, 0],
        )
        self.render_grid(
            config_test,
            agent_pos=hacking_path_train[-1] if hacking_path_train else None,
            path=hacking_path_train,
            title="Hacking Agent — Test",
            ax=axes[1, 1],
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path is not None:
            self.save_figure(fig, save_path)

        logger.debug("render_comparison: saved=%s", save_path)
        return fig

    def save_figure(
        self,
        fig: matplotlib.figure.Figure,
        path: pathlib.Path,
    ) -> None:
        """Save a figure to disk with tight layout applied.

        Args:
            fig: The matplotlib Figure to save.
            path: Destination file path.  The file format is inferred from the
                extension (e.g. .png, .pdf, .svg).
        """
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight", dpi=150)
        logger.info("Figure saved to %s", path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_fig_ax(
        self,
        ax: matplotlib.axes.Axes | None,
        size: int,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Return (fig, ax), creating a new figure if ax is None."""
        if ax is not None:
            return ax.get_figure(), ax
        fig_size = max(4, size * self.cell_size / 80)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        return fig, ax

    def _finalise_ax(
        self,
        ax: matplotlib.axes.Axes,
        size: int,
        title: str,
    ) -> None:
        """Apply axis limits, ticks, aspect ratio, and title."""
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect("equal")
        ax.set_xticks(range(size + 1))
        ax.set_yticks(range(size + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    def _cell_color(self, cell_type: int) -> str:
        """Return the background hex colour string for a cell type constant."""
        mapping = {
            _EMPTY: self.colors["empty"],
            _WALL: self.colors["wall"],
            _LAVA: self.colors["lava"],
            _GOAL: self.colors["goal"],
            _COIN: self.colors["coin"],
            _AGENT: self.colors["agent"],
        }
        return mapping.get(cell_type, self.colors["empty"])

    def _draw_label(
        self,
        ax: matplotlib.axes.Axes,
        row: int,
        col: int,
        label: str,
        grid_size: int,
        color: str = "#000000",
        fontsize: int = 12,
    ) -> None:
        """Draw a text label centred in a grid cell."""
        ax.text(
            col + 0.5,
            grid_size - 1 - row + 0.5,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=color,
            zorder=5,
        )
