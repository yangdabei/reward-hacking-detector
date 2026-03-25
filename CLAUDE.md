# Reward Hacking Detector — Claude Agent Guide

## Project Description

This project builds a controlled laboratory for studying reward hacking in toy reinforcement learning environments. A custom GridWorld environment presents agents with a proxy reward (collecting coins) that is correlated with, but not identical to, the true objective (reaching the goal). Agents trained under varying proxy-reward magnitudes learn to exploit the proxy at the expense of the true goal — a clean, reproducible demonstration of Goodhart's Law. A detection pipeline then measures policy divergence (KL divergence vs. a BFS-optimal reference), trajectory clustering, and reward decomposition to automatically flag hacking agents. The full system is exposed through a FastAPI backend, a Streamlit dashboard, and a CLI, with experiment results persisted in SQLite.

---

## Architecture Overview

```
reward_hacking/
├── src/
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── gridworld.py          # Custom Gymnasium env (ML 1 — step/reset are user exercises)
│   │   ├── configs.py            # 6 named environment configurations
│   │   └── renderer.py           # Matplotlib grid renderer
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── optimal.py            # BFS reference agent (Agent B)
│   │   ├── q_learning.py         # Tabular Q-learning (ML 2 — user exercise)
│   │   └── dqn.py                # Deep Q-Network (ML 3 — user exercise)
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── metrics.py            # Proxy reliance score, reward decomp (Agent C)
│   │   ├── policy_divergence.py  # KL divergence vs. reference (ML 4 — user exercise)
│   │   └── trajectory_analyzer.py# Feature extraction + clustering (ML 5 — user exercise)
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── train.py              # Training loop with structured logging (Agent B)
│   │   ├── evaluate.py           # Evaluation across test environments (Agent B)
│   │   └── sweep.py              # Parallel parameter sweep (Agent B)
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py                # FastAPI endpoints (SWE 4 — user exercise)
│   ├── config.py                 # Pydantic settings models (SWE 1 — user exercise)
│   ├── logging_config.py         # Structured logging setup (SWE 2 — user exercise)
│   ├── storage.py                # SQLite persistence layer (SWE 3 — user exercise)
│   └── cli.py                    # Typer/argparse CLI (SWE 5 — user exercise)
├── dashboard/
│   └── streamlit_app.py          # Interactive Streamlit dashboard (Agent A)
├── notebooks/
│   └── analysis.ipynb            # Exploratory analysis + plots (Agent C)
├── tests/                        # Full test suite (SWE 6 — user exercise)
├── docs/
│   └── METHODOLOGY.md
├── Dockerfile                    # SWE 7 — user exercise
├── docker-compose.yml            # SWE 7 — user exercise
├── .github/
│   └── workflows/
│       └── ci.yml                # SWE 8 — user exercise
├── pyproject.toml
├── requirements.txt
├── CLAUDE.md
├── agents.md
└── README.md
```

---

## Agent Roles

### Agent A — Environment & Visualisation

Agent A owns everything related to the environment definition and visual output. This includes:

- `src/environment/gridworld.py` — Provides the class skeleton, `__init__`, and all private helper methods. The public `step()` and `reset()` methods are intentionally left as stubs (ML Exercise 1) for the user to implement.
- `src/environment/configs.py` — All six named environment configurations (training envs with varying proxy-reward magnitudes, test envs with shifted coin placements).
- `src/environment/renderer.py` — Matplotlib-based renderer that draws the grid, agent position, coin, and goal for a single timestep or an animated trajectory.
- `dashboard/streamlit_app.py` — Full interactive Streamlit dashboard: live policy rollout viewer, experiment comparison charts, detection score heatmaps.

### Agent B — Training & Evaluation Infrastructure

Agent B owns the experiment lifecycle — running agents, collecting data, and computing aggregate statistics:

- `src/agents/optimal.py` — BFS-based optimal reference agent used as the baseline for divergence measurement.
- `src/experiments/train.py` — Training loop: iterates episodes, calls agent `train()`, logs structured episode data, saves checkpoints.
- `src/experiments/evaluate.py` — Evaluates a trained agent across all test environments, records per-episode rewards and terminal states.
- `src/experiments/sweep.py` — Launches a grid of (agent_type × coin_reward) experiments in parallel using `concurrent.futures`.

### Agent C — Detection & Analysis

Agent C owns the reward-hacking detection pipeline and post-hoc analysis:

- `src/detection/metrics.py` — Implements Proxy Reliance Score, reward decomposition (proxy vs. true reward per episode), and the full detection pipeline that aggregates all signals into a single hacking verdict.
- `notebooks/analysis.ipynb` — Jupyter notebook with pre-written analysis cells, visualisation helpers, and interpretation guidance; students fill in the results after running experiments.

---

## What Claude MUST NOT Implement

The following files and functions are **user learning exercises**. Claude agents must provide stubs, docstrings, and type signatures — but must leave the implementation body as `raise NotImplementedError` or a `# TODO` comment. Do not implement these under any circumstances, even if asked to "just fill it in quickly."

### SWE Exercises (software-engineering skills)

| ID   | File                        | What the user must implement                              |
|------|-----------------------------|-----------------------------------------------------------|
| SWE 1 | `src/config.py`            | Pydantic `BaseSettings` models for all configuration      |
| SWE 2 | `src/logging_config.py`    | Structured JSON logging setup, log-level from config      |
| SWE 3 | `src/storage.py`           | SQLite schema creation, experiment CRUD via `sqlite3`     |
| SWE 4 | `src/api/app.py`           | FastAPI route handlers (list, get, trigger, detect)       |
| SWE 5 | `src/cli.py`               | CLI entry points: `train`, `evaluate`, `detect`           |
| SWE 6 | `tests/`                   | Full pytest test suite (unit + integration + e2e)         |
| SWE 7 | `Dockerfile` + `docker-compose.yml` | Container definitions and service orchestration  |
| SWE 8 | `.github/workflows/ci.yml` | GitHub Actions CI pipeline (lint, type-check, test)       |

### ML Exercises (machine-learning skills)

| ID   | File                                  | What the user must implement                                        |
|------|---------------------------------------|---------------------------------------------------------------------|
| ML 1 | `src/environment/gridworld.py`        | `step(action)` and `reset(seed, options)` — core Gymnasium API      |
| ML 2 | `src/agents/q_learning.py`            | `update()`, `select_action()`, `train()` — tabular Q-learning       |
| ML 3 | `src/agents/dqn.py`                   | `QNetwork` (nn.Module), `DQNAgent` with replay buffer and target net |
| ML 4 | `src/detection/policy_divergence.py`  | KL divergence computation between reference and learned policies    |
| ML 5 | `src/detection/trajectory_analyzer.py`| Trajectory feature extraction and KMeans clustering                |

---

## Code Style Requirements

All code in this project must follow these conventions:

- **Python version**: 3.10+ — use `match`/`case`, `X | Y` union types, and `from __future__ import annotations` where helpful.
- **Type hints**: Every function and method signature must have complete type annotations on all parameters and the return type. No bare `Any` unless genuinely unavoidable.
- **Docstrings**: Google style. Every public class, method, and function must have a docstring with `Args:`, `Returns:`, and `Raises:` sections as applicable.
- **Function length**: Maximum 30 lines per function body (excluding docstring). Extract helpers if needed.
- **Logging**: Use `logger = logging.getLogger(__name__)` at module level. Never use `print()` for diagnostic output in library code.
- **Paths**: Use `pathlib.Path` throughout. Never use `os.path` string manipulation.
- **Configuration**: All tuneable parameters must be exposed as Pydantic model fields, not as bare module-level constants.
- **Imports**: Grouped as standard library / third-party / local, separated by blank lines (enforced by `ruff --select I`).

---

## Key Invariants

The following contracts must be maintained across all agents' code:

### Gymnasium API Compliance

Every environment must conform to the Gymnasium 0.29+ API:

```python
obs, info = env.reset(seed=42)          # reset returns (obs, info)
obs, reward, terminated, truncated, info = env.step(action)  # 5-tuple
```

Never use the old 4-tuple `step()` return. Always pass both `terminated` and `truncated` to the agent.

### Agent Interface

Every agent class must implement exactly these two public methods (duck-typed protocol):

```python
def select_action(self, state: np.ndarray) -> int:
    """Select an action for the given state (greedy / epsilon-greedy)."""
    ...

def train(self, env: gymnasium.Env, num_episodes: int) -> list[float]:
    """Run training loop; return list of per-episode total rewards."""
    ...
```

The `train()` method must return a list whose length equals `num_episodes`.

---

## Running the Project

Install dependencies:
```bash
pip install -r requirements.txt
```

Train an agent:
```bash
python -m src.cli train --config training_default --agent q_learning --episodes 1000
```

Evaluate a trained agent against test environments:
```bash
python -m src.cli evaluate --experiment-id <id> --test-env test_coin_moved
```

Run the reward-hacking detector:
```bash
python -m src.cli detect --experiment-id <id>
```

Launch the interactive dashboard:
```bash
streamlit run dashboard/streamlit_app.py
```

Start the REST API server:
```bash
uvicorn src.api.app:app --reload
```
