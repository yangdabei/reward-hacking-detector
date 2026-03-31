"""Configuration models for the Reward Hacking Detector."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


class RewardConfig(BaseModel):
    """Reward values awarded to the agent for each event."""
    goal: float = 10.0
    coin: float = 5.0
    step: float = -0.1
    lava: float = -5.0


class EnvConfig(BaseModel):
    """Gridworld layout and episode parameters."""
    grid_size: int = 7
    agent_start: tuple[int, int] = (0, 0)
    goal_position: tuple[int, int] = (6, 6)
    coin_position: tuple[int, int] | None = (3, 3)
    coin_terminal: bool = True
    lava_positions: list[tuple[int, int]] = []
    wall_positions: list[tuple[int, int]] = []
    max_steps: int = 200
    rewards: RewardConfig = RewardConfig()

    @field_validator("grid_size")
    def grid_size_positive(cls, v):
        """Grid must be at least 3x3 to have meaningful start/goal position."""
        if v < 3:
            raise ValueError("Grid size must be at least 3")
        return v

    @model_validator(mode="after")
    def positions_valid(self):
        """All positions must be in-bounds, and start/goal/coin must not overlap."""
        n = self.grid_size

        def in_bounds(pos: tuple[int, int]) -> bool:
            return 0 <= pos[0] < n and 0 <= pos[1] < n

        for pos in [self.agent_start, self.goal_position]:
            if not in_bounds(pos):
                raise ValueError(f"Position {pos} is outside the {n}x{n} grid")
        if self.agent_start == self.goal_position:
            raise ValueError("agent_start must not equal goal_position")
        if self.coin_position is not None:
            if not in_bounds(self.coin_position):
                raise ValueError(f"coin_position {self.coin_position} is outside the grid")
            if self.coin_position in (self.agent_start, self.goal_position):
                raise ValueError("coin_position must not overlap agent_start or goal_position")
        for pos in self.lava_positions + self.wall_positions:
            if not in_bounds(pos):
                raise ValueError(f"Position {pos} is outside the {n}x{n} grid")
        return self


class AgentConfig(BaseModel):
    """Hyperparameters for the RL agent."""
    learning_rate: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 0.01
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    batch_size: int = 32
    target_update_freq: int = 100
    replay_buffer_size: int = 10000

    @field_validator("learning_rate")
    def lr_in_range(cls, v):
        """Learning rate must be in (0,1] - 0 means no learning, >1 diverges."""
        if not 0 < v <= 1:
            raise ValueError("Learning rate must be between 0 and 1")
        return v


class ExperimentConfig(BaseModel):
    """Top-level config for a single training run."""
    env: EnvConfig = EnvConfig()
    agent: AgentConfig = AgentConfig()
    agent_type: Literal["q_learning", "dqn"] = "q_learning"
    num_episodes: int = 1000
    seed: int = 42
