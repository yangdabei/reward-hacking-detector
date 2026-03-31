"""Matplotlib-based grid renderer for the Reward Hacking Detector project.

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
from typing import TYPE_CHECKING

import matplotlib.axes
import matplotlib.figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

if TYPE_CHECKING:
    from src.config import EnvConfig

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
# Sprite assets
# ---------------------------------------------------------------------------

_SPRITE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "grid"

# Module-level cache so sprites are read from disk at most once per process.
_sprite_cache: dict[str, np.ndarray] = {}


def _load_sprite(name: str) -> np.ndarray | None:
    """Load a PNG sprite from assets/grid/, returning an RGBA float32 array.

    Results are cached in ``_sprite_cache`` so each file is read only once.

    Args:
        name: Sprite name without extension (e.g. ``"coin"``).

    Returns:
        A (H, W, 4) float32 array with values in [0, 1], or None if the file
        does not exist.
    """
    if name in _sprite_cache:
        return _sprite_cache[name]
    path = _SPRITE_DIR / f"{name}.png"
    if not path.exists():
        return None
    img = plt.imread(str(path))  # float32, shape (H, W, 3 or 4)
    if img.ndim == 2:
        # Greyscale → RGBA
        rgba = np.ones((*img.shape, 4), dtype=np.float32)
        rgba[:, :, :3] = img[:, :, np.newaxis]
    elif img.shape[2] == 3:
        rgba = np.ones((*img.shape[:2], 4), dtype=np.float32)
        rgba[:, :, :3] = img
    else:
        rgba = img.astype(np.float32)
    _sprite_cache[name] = rgba
    return rgba


# ---------------------------------------------------------------------------
# Colour scheme (fallback when sprites are absent)
# ---------------------------------------------------------------------------

_COLORS: dict[str, str] = {
    "empty": "#FFFFFF",
    "wall": "#5A5A5A",
    "goal": "#27AE60",
    "coin": "#F1C40F",
    "lava": "#E74C3C",
    "agent": "#2980B9",
    "path": "#85C1E9",
    "grid_line": "#CCCCCC",
}

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_grid_array(config: "EnvConfig") -> np.ndarray:
    """Build a 2D integer numpy array from an EnvConfig.

    Only static cell types (walls and lava) are encoded.  Dynamic entities
    (agent, goal, coin) are *not* included — overlay them separately when
    needed.

    Args:
        config: An EnvConfig (or compatible) object.

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
    optional path overlays, and state-visitation heatmaps.  When PNG sprite
    assets are present in ``assets/grid/`` they are used instead of plain
    coloured patches.

    Args:
        cell_size: Logical size of each grid cell in pixels (used to determine
            figure dimensions).  Defaults to 80.
    """

    def __init__(self, cell_size: int = 80) -> None:
        self.cell_size = cell_size
        self.colors = dict(_COLORS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_grid(
        self,
        config: "EnvConfig",
        agent_pos: tuple[int, int] | None = None,
        path: list[tuple[int, int]] | None = None,
        title: str = "GridWorld",
        ax: matplotlib.axes.Axes | None = None,
    ) -> matplotlib.figure.Figure:
        """Render the gridworld as a matplotlib figure.

        Uses PNG sprite assets when available, falling back to coloured
        patches with text labels.  Draws the goal, coin, and agent on top
        of the base grid.

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

        self._draw_background(ax, grid, size)

        # Entity sprites / labels
        gr, gc = config.goal_position
        self._draw_entity(ax, gr, gc, "goal", "G", size, fallback_color=self.colors["goal"])

        coin_pos = getattr(config, "coin_position", None)
        if coin_pos is not None:
            cr, cc = coin_pos
            self._draw_entity(ax, cr, cc, "coin", "C", size, fallback_color=self.colors["coin"])

        # Path overlay
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

        # Agent
        if agent_pos is not None:
            ar, ac = agent_pos
            self._draw_entity(ax, ar, ac, "agent", "A", size, fallback_color=self.colors["agent"])

        self._finalise_ax(ax, size, title)
        logger.debug("render_grid: config grid_size=%d title=%r", size, title)
        return fig

    def render_path(
        self,
        config: "EnvConfig",
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
        config: "EnvConfig",
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

        density = np.zeros((size, size), dtype=float)
        max_count = max(visit_counts.values()) if visit_counts else 1
        for (r, c), count in visit_counts.items():
            density[r, c] = count / max_count

        heatmap_cmap = LinearSegmentedColormap.from_list(
            "visit_heat", ["#FFFFFF", "#1A5276"], N=256
        )

        base_grid = make_grid_array(config)
        for r in range(size):
            for c in range(size):
                cell = base_grid[r, c]
                if cell == _WALL:
                    color = self.colors["wall"]
                elif cell == _LAVA:
                    color = self.colors["lava"]
                else:
                    color = heatmap_cmap(density[r, c])
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

        gr, gc = config.goal_position
        self._draw_label(ax, gr, gc, "G", size, color=self.colors["goal"], fontsize=14)

        coin_pos = getattr(config, "coin_position", None)
        if coin_pos is not None:
            cr, cc = coin_pos
            self._draw_label(ax, cr, cc, "C", size, color=self.colors["coin"], fontsize=14)

        sm = plt.cm.ScalarMappable(cmap=heatmap_cmap, norm=plt.Normalize(0, max_count))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="Visit count")

        self._finalise_ax(ax, size, title)
        logger.debug("render_heatmap: %d unique cells visited", len(visit_counts))
        return fig

    def render_comparison(
        self,
        config_train: "EnvConfig",
        config_test: "EnvConfig",
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

        aligned_path = [pos for pos, _, _ in traj_aligned] if traj_aligned else []
        hacking_path = [pos for pos, _, _ in traj_hacking] if traj_hacking else []

        self.render_grid(
            config_train,
            agent_pos=aligned_path[-1] if aligned_path else None,
            path=aligned_path,
            title="Aligned Agent — Training",
            ax=axes[0, 0],
        )
        self.render_grid(
            config_train,
            agent_pos=hacking_path[-1] if hacking_path else None,
            path=hacking_path,
            title="Hacking Agent — Training",
            ax=axes[0, 1],
        )
        self.render_grid(
            config_test,
            agent_pos=aligned_path[-1] if aligned_path else None,
            path=aligned_path,
            title="Aligned Agent — Test",
            ax=axes[1, 0],
        )
        self.render_grid(
            config_test,
            agent_pos=hacking_path[-1] if hacking_path else None,
            path=hacking_path,
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

    def _draw_background(
        self,
        ax: matplotlib.axes.Axes,
        grid: np.ndarray,
        size: int,
    ) -> None:
        """Draw the base grid background using sprites when available.

        Attempts to build a single composite RGBA image from cell sprites.
        Falls back to individual FancyBboxPatch rectangles if sprites are
        not found.

        Args:
            ax: Axes to draw into.
            grid: 2D int32 array from ``make_grid_array``.
            size: Grid dimension.
        """
        empty_sprite = _load_sprite("empty")
        if empty_sprite is None:
            self._draw_background_patches(ax, grid, size)
            return

        sh, sw = empty_sprite.shape[:2]
        composite = np.ones((size * sh, size * sw, 4), dtype=np.float32)

        cell_sprite_map = {_WALL: "wall", _LAVA: "lava"}
        for r in range(size):
            for c in range(size):
                name = cell_sprite_map.get(int(grid[r, c]), "empty")
                loaded = _load_sprite(name)
                sprite = loaded if loaded is not None else empty_sprite
                composite[r * sh : (r + 1) * sh, c * sw : (c + 1) * sw] = sprite

        ax.imshow(
            composite,
            extent=[0, size, 0, size],
            origin="upper",
            aspect="auto",
            zorder=1,
            interpolation="bilinear",
        )

    def _draw_background_patches(
        self,
        ax: matplotlib.axes.Axes,
        grid: np.ndarray,
        size: int,
    ) -> None:
        """Fallback: draw plain coloured FancyBboxPatch rectangles."""
        for r in range(size):
            for c in range(size):
                rect = mpatches.FancyBboxPatch(
                    (c, size - 1 - r),
                    1,
                    1,
                    boxstyle="square,pad=0",
                    facecolor=self._cell_color(int(grid[r, c])),
                    edgecolor=self.colors["grid_line"],
                    linewidth=0.5,
                )
                ax.add_patch(rect)

    def _draw_entity(
        self,
        ax: matplotlib.axes.Axes,
        row: int,
        col: int,
        sprite_name: str,
        fallback_label: str,
        grid_size: int,
        fallback_color: str = "#000000",
    ) -> None:
        """Draw an entity (goal, coin, or agent) using a sprite or a text label.

        Args:
            ax: Axes to draw into.
            row: Grid row of the entity.
            col: Grid column of the entity.
            sprite_name: Name of the sprite file (without extension).
            fallback_label: Single character label used when the sprite is absent.
            grid_size: Size of the grid (for coordinate flipping).
            fallback_color: Colour for the fallback text label.
        """
        sprite = _load_sprite(sprite_name)
        if sprite is not None:
            # Place sprite exactly within the 1×1 data-unit cell.
            # extent=[xmin, xmax, ymin, ymax] with origin='upper':
            #   image row 0 → ymax, image row -1 → ymin.
            ymin = grid_size - 1 - row
            ymax = grid_size - row
            ax.imshow(
                sprite,
                extent=[col, col + 1, ymin, ymax],
                origin="upper",
                aspect="auto",
                zorder=4,
                interpolation="bilinear",
            )
        else:
            self._draw_label(
                ax, row, col, fallback_label, grid_size, color=fallback_color, fontsize=14
            )
