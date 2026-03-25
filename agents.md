# Agent Coordination Guide — Reward Hacking Detector

This document is the authoritative reference for which agent owns which file, what each agent must fully implement, and which files must be left as stubs for the user. Read this before writing a single line of code.

---

## Agent A: Environment & Visualisation

Agent A is responsible for everything a human observer interacts with directly: the environment itself, its visual rendering, and the live dashboard.

### Files Owned (full implementation required)

#### `src/environment/gridworld.py` — GridWorld Environment (partial)

Agent A provides:
- The full class definition with all private attributes documented.
- `__init__(self, config: GridWorldConfig)` — complete implementation including observation/action space construction.
- Private helpers: `_build_observation()`, `_is_valid_position()`, `_manhattan_distance()`, `_get_coin_positions()`, and any other internals.
- Class-level docstring explaining the observation format, action space, and reward structure.

Agent A must NOT implement:
- `step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]` — leave as stub with full docstring (ML Exercise 1).
- `reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]` — leave as stub with full docstring (ML Exercise 1).

Stub format for user exercises:
```python
def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
    """Run one timestep of the environment.

    Args:
        action: Integer in [0, 3] representing UP, RIGHT, DOWN, LEFT.

    Returns:
        A 5-tuple (observation, reward, terminated, truncated, info) following
        the Gymnasium 0.29+ API. ``terminated`` is True when the agent reaches
        the goal. ``truncated`` is True when ``max_steps`` is exceeded.
        ``info`` must contain at least ``{"proxy_reward": float, "true_reward": float}``.

    Raises:
        ValueError: If action is outside the valid action space.
    """
    # TODO (ML Exercise 1): Implement the environment step logic.
    # Steps to complete:
    #   1. Validate the action against self.action_space.
    #   2. Compute the proposed new position from self._agent_pos and action.
    #   3. Clamp to grid bounds (walls are solid).
    #   4. Check if the new position contains a coin; if so, collect it and
    #      add self.config.coin_reward to the proxy reward.
    #   5. Check if the new position is the goal; if so, add self.config.goal_reward
    #      and set terminated = True.
    #   6. Increment self._step_count; set truncated = True if >= self.config.max_steps.
    #   7. Build and return the full 5-tuple using self._build_observation().
    raise NotImplementedError("ML Exercise 1: implement GridWorld.step()")
```

#### `src/environment/configs.py` — Environment Configurations

Full implementation of all six named `GridWorldConfig` instances:

| Name | Purpose | Proxy coin reward | Notes |
|------|---------|-------------------|-------|
| `training_default` | Baseline training | 0.5 | Coin aligned with goal path |
| `training_high_proxy` | Strong hacking signal | 5.0 | Coin reward dominates goal reward |
| `training_low_proxy` | Weak hacking signal | 0.1 | Agent should ignore coin |
| `test_coin_moved` | Generalisation test | 0.5 | Coin in opposite corner from training |
| `test_no_coin` | True objective test | 0.0 | No coin at all; measures true goal-seeking |
| `test_adversarial` | Worst-case test | 5.0 | Coin blocks shortest path to goal |

Each config is a `GridWorldConfig` Pydantic model. All six instances are exported from the module and collected in a `CONFIG_REGISTRY: dict[str, GridWorldConfig]` mapping.

#### `src/environment/renderer.py` — Matplotlib Renderer

Full implementation:
- `GridRenderer` class with `__init__(self, config: GridWorldConfig)`.
- `render_frame(self, obs: np.ndarray, ax: matplotlib.axes.Axes | None = None) -> matplotlib.figure.Figure` — draws a single grid frame.
- `render_trajectory(self, trajectory: list[np.ndarray], output_path: pathlib.Path) -> None` — saves an animated GIF of a full episode.
- `render_policy_heatmap(self, q_table: np.ndarray, ax: matplotlib.axes.Axes | None = None) -> matplotlib.figure.Figure` — colour-map of the greedy policy over all states.

Uses only `matplotlib`; no extra dependencies.

#### `dashboard/streamlit_app.py` — Interactive Dashboard

Full implementation:
- Sidebar: experiment selector (loads from SQLite via `src.storage`), environment config selector.
- Tab 1 "Live Rollout": renders a real-time policy rollout using `GridRenderer`; play/pause/step controls.
- Tab 2 "Training Curves": line charts of episode rewards for all selected experiments.
- Tab 3 "Detection Results": bar chart of Proxy Reliance Scores per agent type; heatmap of KL divergence across (agent, coin_reward) grid.
- Tab 4 "Trajectory Clusters": scatter plot of trajectory embeddings coloured by cluster label.

Uses only `streamlit`, `matplotlib`, and local `src` imports.

---

## Agent B: Training & Evaluation Infrastructure

Agent B owns the experiment loop — the machinery that trains agents, stores results, and generates the data that the detection pipeline and dashboard consume.

### Files Owned (full implementation required)

#### `src/agents/optimal.py` — BFS Reference Agent

Full implementation of `OptimalAgent`:
- `__init__(self, config: GridWorldConfig)` — stores config; no learning.
- `select_action(self, state: np.ndarray) -> int` — runs BFS from current agent position to goal (ignoring coins) and returns the first action on the shortest path.
- `train(self, env: gymnasium.Env, num_episodes: int) -> list[float]` — no-op training loop (BFS needs no training); runs `num_episodes` evaluation episodes and returns per-episode rewards.

The BFS implementation must be deterministic and handle all edge cases (goal unreachable → raise; already at goal → return no-op action).

#### `src/experiments/train.py` — Training Loop

Full implementation:
- `train_agent(agent, env, config: TrainingConfig, experiment_id: str) -> TrainingResult` — outer training loop.
- Calls `agent.train(env, config.num_episodes)` and captures per-episode rewards.
- Logs a structured JSON record after every `config.log_interval` episodes via `logger`.
- Saves agent checkpoint to `config.checkpoint_dir / experiment_id / "checkpoint.pt"` (or `.pkl` for Q-learning).
- Persists the `TrainingResult` to SQLite via `src.storage`.
- Returns a `TrainingResult` Pydantic model with `experiment_id`, `agent_type`, `env_config_name`, `episode_rewards`, `total_steps`, `wall_time_seconds`.

#### `src/experiments/evaluate.py` — Evaluation Pipeline

Full implementation:
- `evaluate_agent(agent, env_configs: list[str], num_episodes: int, experiment_id: str) -> EvaluationResult` — runs the agent greedily (no exploration) on each named test environment.
- Records per-environment mean reward, goal-reach rate, coin-collection rate, and mean episode length.
- Persists `EvaluationResult` to SQLite.
- Returns an `EvaluationResult` Pydantic model.

#### `src/experiments/sweep.py` — Parameter Sweep

Full implementation:
- `run_sweep(sweep_config: SweepConfig) -> list[str]` — takes a `SweepConfig` (agent types × coin reward values × seeds) and fans out training jobs using `concurrent.futures.ProcessPoolExecutor`.
- Returns a list of experiment IDs for downstream use.
- Gracefully handles worker exceptions (log and continue; do not crash the whole sweep).

---

## Agent C: Detection & Analysis

Agent C owns the scientific core of the project: computing and interpreting the signals that indicate reward hacking.

### Files Owned (full implementation required)

#### `src/detection/metrics.py` — Detection Metrics

Full implementation:
- `compute_proxy_reliance_score(eval_result: EvaluationResult) -> float` — implements the Proxy Reliance Score formula (see METHODOLOGY.md Section 3).
- `decompose_rewards(episode_log: list[dict]) -> RewardDecomposition` — separates per-episode rewards into proxy component and true-goal component by reading the `info["proxy_reward"]` and `info["true_reward"]` fields logged during evaluation.
- `run_detection_pipeline(experiment_id: str) -> DetectionReport` — orchestrates all detection metrics (PRS, KL divergence, trajectory clusters) into a single `DetectionReport` Pydantic model with a boolean `is_hacking` verdict and per-metric scores.

Agent C must NOT implement:
- `policy_divergence.py` — KL divergence computation (ML Exercise 4, user).
- `trajectory_analyzer.py` — feature extraction and clustering (ML Exercise 5, user).

Stubs for both files (with full docstrings and `raise NotImplementedError`) must be present so that `metrics.py` can import them without error at module load time. Use `try/except NotImplementedError` guards in `run_detection_pipeline` so the pipeline degrades gracefully when the exercises are not yet implemented.

#### `notebooks/analysis.ipynb` — Analysis Notebook

Full implementation of the notebook scaffold:
- Cell 1: imports and configuration (loads experiment IDs from SQLite).
- Cell 2: training curve plots for all sweep experiments.
- Cell 3: Proxy Reliance Score bar chart.
- Cell 4: KL divergence heatmap (calls `policy_divergence.compute_kl_divergence`; will error until ML 4 is done — note this in a markdown cell).
- Cell 5: Trajectory cluster scatter plot (calls `trajectory_analyzer.cluster_trajectories`; same caveat).
- Cell 6: Markdown interpretation template for students to fill in.

---

## User Learning Exercises

The following files and functions must NOT be implemented by any agent. Every file should exist as a stub with complete docstrings, type signatures, and `raise NotImplementedError` bodies.

### SWE Exercises

| ID    | File path                        | Description                                                           |
|-------|----------------------------------|-----------------------------------------------------------------------|
| SWE 1 | `src/config.py`                  | Pydantic `BaseSettings` models for training, eval, API, and storage config |
| SWE 2 | `src/logging_config.py`          | Structured JSON logging setup; log-level driven by config             |
| SWE 3 | `src/storage.py`                 | SQLite schema, experiment/result CRUD operations via `sqlite3`        |
| SWE 4 | `src/api/app.py`                 | FastAPI route handlers: list experiments, get result, trigger train, run detect |
| SWE 5 | `src/cli.py`                     | CLI with `train`, `evaluate`, `detect` subcommands                    |
| SWE 6 | `tests/`                         | Full pytest suite: unit tests, integration tests, async API tests     |
| SWE 7 | `Dockerfile` + `docker-compose.yml` | Container build and multi-service orchestration                    |
| SWE 8 | `.github/workflows/ci.yml`       | GitHub Actions: lint (ruff), type-check (mypy), test (pytest)         |

### ML Exercises

| ID   | File path                                  | Description                                                           |
|------|--------------------------------------------|-----------------------------------------------------------------------|
| ML 1 | `src/environment/gridworld.py`             | `step()` and `reset()` — core Gymnasium environment dynamics          |
| ML 2 | `src/agents/q_learning.py`                 | `update()`, `select_action()`, `train()` — tabular Q-learning with epsilon-greedy |
| ML 3 | `src/agents/dqn.py`                        | `QNetwork` (nn.Module), `DQNAgent` with replay buffer and target network |
| ML 4 | `src/detection/policy_divergence.py`       | `compute_kl_divergence(ref_policy, learned_policy)` — KL divergence   |
| ML 5 | `src/detection/trajectory_analyzer.py`    | `extract_features(trajectory)` and `cluster_trajectories(features)`  |

---

## Implementation Order

The following sequence minimises blocking dependencies. Steps marked **USER TODO** must be completed by the user before the next agent step can proceed.

| Step | Owner | Deliverable | Blocks |
|------|-------|-------------|--------|
| 1 | Agent A | `gridworld.py` scaffold, `configs.py`, `renderer.py` | Step 2 |
| 2 | **USER TODO (ML 1)** | Implement `GridWorld.step()` and `GridWorld.reset()` | Steps 3–11 |
| 3 | Agent B | `optimal.py`, `train.py` (training loop) | Step 4 |
| 4 | **USER TODO (ML 2)** | Implement `QLearningAgent` in `q_learning.py` | Step 5 |
| 5 | Agent B | `evaluate.py` (evaluation pipeline) | Step 6 |
| 6 | **USER TODO (ML 3)** | Implement `DQNAgent` in `dqn.py` | Step 7 |
| 7 | Agent C + **USER TODO (ML 4, ML 5)** | `metrics.py` (full); stubs for `policy_divergence.py` and `trajectory_analyzer.py`; `analysis.ipynb` scaffold | Step 8 |
| 8 | Agent B | `sweep.py` (parameter sweep) | Step 9 |
| 9 | Agent A + Agent B | `streamlit_app.py` (dashboard); `src/api/app.py` stub for user | Step 10 |
| 10 | **USER TODO (SWE 1–8)** | `config.py`, `logging_config.py`, `storage.py`, `api/app.py`, `cli.py`, `tests/`, `Dockerfile`, `ci.yml` | Step 11 |
| 11 | Any agent | README refinement, docstring review, final notebook polish | — |

**Key dependency rule**: Steps 3 onward all depend on ML Exercise 1 being complete. If a user asks an agent to run experiments before implementing `step()`/`reset()`, the agent should clearly explain the dependency and point the user to the ML 1 stub in `gridworld.py`.
