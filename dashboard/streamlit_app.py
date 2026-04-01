"""
Reward Hacking Detector — Streamlit Dashboard.

Interactive web dashboard for exploring GridWorld environments, running
RL agent experiments, and visualising reward-hacking behaviour vs aligned
agent behaviour.

Run with::

    streamlit run dashboard/streamlit_app.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend required for Streamlit
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Extend sys.path so imports from src/ work when running from repo root or
# from the dashboard/ directory.
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Optional imports from src/  — show helpful errors if modules are missing.
# ---------------------------------------------------------------------------

try:
    from src.environment.configs import ALL_CONFIGS, get_config
    _CONFIGS_AVAILABLE = True
except ImportError as exc:
    _CONFIGS_AVAILABLE = False
    _CONFIGS_ERROR = str(exc)

try:
    from src.environment.renderer import GridRenderer
    _RENDERER_AVAILABLE = True
except ImportError as exc:
    _RENDERER_AVAILABLE = False
    _RENDERER_ERROR = str(exc)

try:
    from src.environment.gridworld import GridWorld
    _GRIDWORLD_AVAILABLE = True
except ImportError as exc:
    _GRIDWORLD_AVAILABLE = False
    _GRIDWORLD_ERROR = str(exc)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Reward Hacking Detector",
    page_icon="🪤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Reward Hacking Detector")
    st.markdown(
        "An simple GridWorld environment demonstrating how RL agents exploit "
        "proxy rewards instead of pursuing the true objective."
    )
    st.divider()

    # Environment config selector
    config_options = (
        list(ALL_CONFIGS.keys()) if _CONFIGS_AVAILABLE else
        ["training_default", "test_coin_moved", "test_no_coin",
         "test_coin_near_lava", "training_large", "training_multi_coin"]
    )
    selected_config_name = st.selectbox(
        "Environment Config",
        options=config_options,
        index=0,
        help="Choose which gridworld configuration to use.",
    )

    # Agent type selector
    agent_type = st.selectbox(
        "Agent Type",
        options=["q_learning", "dqn", "optimal"],
        index=0,
        help=(
            "q_learning — tabular Q-learning; "
            "dqn — deep Q-network; "
            "optimal — hand-coded shortest-path agent."
        ),
    )

    # Episode count
    num_episodes = st.number_input(
        "Number of Episodes",
        min_value=100,
        max_value=5000,
        value=500,
        step=100,
        help="How many training episodes to run.",
    )

    st.divider()

    run_experiment = st.button(
        "Run Experiment",
        type="primary",
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Fetch selected config
# ---------------------------------------------------------------------------

selected_config = None
if _CONFIGS_AVAILABLE:
    try:
        selected_config = get_config(selected_config_name)
    except ValueError as exc:
        st.error(f"Config error: {exc}")

# ---------------------------------------------------------------------------
# Main content — tabs
# ---------------------------------------------------------------------------

tab_env, tab_train, tab_compare, tab_metrics, tab_about = st.tabs(
    ["Environment", "Training", "Comparison", "Detection Metrics", "About"]
)

# ==========================================================================
# Tab 1 — Environment
# ==========================================================================

with tab_env:
    st.header("Environment Preview")

    if not _CONFIGS_AVAILABLE:
        st.error(
            f"Could not import `environment.configs`. "
            f"Make sure src/ is on your Python path.\n\nError: {_CONFIGS_ERROR}"
        )
    elif not _RENDERER_AVAILABLE:
        st.warning(
            f"Renderer not available — install matplotlib to enable visualisation.\n\n"
            f"Error: {_RENDERER_ERROR}"
        )
        if selected_config:
            st.json(
                {
                    "grid_size": selected_config.grid_size,
                    "agent_start": list(selected_config.agent_start),
                    "goal_position": list(selected_config.goal_position),
                    "coin_position": list(selected_config.coin_position) if selected_config.coin_position else None,
                    "lava_positions": [list(p) for p in selected_config.lava_positions],
                    "wall_positions": [list(p) for p in selected_config.wall_positions],
                    "max_steps": selected_config.max_steps,
                    "coin_terminal": selected_config.coin_terminal,
                    "reward_goal": selected_config.rewards.goal,
                    "reward_coin": selected_config.rewards.coin,
                    "reward_step": selected_config.rewards.step,
                    "reward_lava": selected_config.rewards.lava,
                }
            )
    else:
        col_render, col_info = st.columns([2, 1])

        with col_render:
            if selected_config is not None:
                renderer = GridRenderer(cell_size=80)
                fig = renderer.render_grid(
                    selected_config,
                    agent_pos=selected_config.agent_start,
                    title=f"Config: {selected_config_name}",
                )
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("Select a configuration in the sidebar.")

        with col_info:
            st.subheader("Configuration Details")
            if selected_config is not None:
                config_dict = {
                    "grid_size": selected_config.grid_size,
                    "agent_start": list(selected_config.agent_start),
                    "goal_position": list(selected_config.goal_position),
                    "coin_position": list(selected_config.coin_position) if selected_config.coin_position else None,
                    "lava_positions": [list(p) for p in selected_config.lava_positions],
                    "wall_positions": [list(p) for p in selected_config.wall_positions],
                    "max_steps": selected_config.max_steps,
                    "coin_terminal": selected_config.coin_terminal,
                    "rewards": {
                        "goal": selected_config.rewards.goal,
                        "coin": selected_config.rewards.coin,
                        "step": selected_config.rewards.step,
                        "lava": selected_config.rewards.lava,
                    },
                }
                st.json(config_dict)

# ==========================================================================
# Tab 2 — Training
# ==========================================================================

with tab_train:
    st.header("Training Curves")

    if not run_experiment:
        st.info(
            "Configure your experiment in the sidebar and click **Run Experiment** "
            "to start training.  Training curves will appear here."
        )

        # Placeholder chart so the tab doesn't look empty
        st.subheader("Placeholder: Reward over Episodes")
        placeholder_episodes = np.arange(1, 101)
        placeholder_rewards = (
            -5 + 15 * (1 - np.exp(-placeholder_episodes / 30))
            + np.random.default_rng(42).normal(0, 1, 100)
        )
        fig_placeholder, ax_placeholder = plt.subplots(figsize=(10, 4))
        ax_placeholder.plot(
            placeholder_episodes,
            placeholder_rewards,
            color="#2980B9",
            linewidth=1.5,
            alpha=0.7,
            label="Episode reward (example)",
        )
        ax_placeholder.axhline(0, color="#CCCCCC", linewidth=0.8, linestyle="--")
        ax_placeholder.set_xlabel("Episode")
        ax_placeholder.set_ylabel("Total Reward")
        ax_placeholder.set_title("Example Training Curve (placeholder)")
        ax_placeholder.legend()
        ax_placeholder.grid(True, alpha=0.3)
        st.pyplot(fig_placeholder, use_container_width=True)
        plt.close(fig_placeholder)
        st.caption("This is synthetic example data.  Run a real experiment to see actual results.")

    else:
        with st.spinner(f"Running {num_episodes} episodes with {agent_type}…"):
            # Placeholder: simulate a training run
            time.sleep(0.5)

        st.success(f"Experiment complete: {num_episodes} episodes, agent={agent_type}")

        rng = np.random.default_rng(int(time.time()))
        ep_rewards = (
            -5
            + 15 * (1 - np.exp(-np.arange(1, num_episodes + 1) / (num_episodes * 0.15)))
            + rng.normal(0, 1.5, num_episodes)
        )
        rolling_mean = (
            np.convolve(ep_rewards, np.ones(20) / 20, mode="valid")
        )

        fig_train, ax_train = plt.subplots(figsize=(10, 4))
        ax_train.plot(ep_rewards, color="#85C1E9", linewidth=0.8, alpha=0.6, label="Episode reward")
        ax_train.plot(
            np.arange(19, num_episodes),
            rolling_mean,
            color="#2980B9",
            linewidth=2,
            label="Rolling mean (20 ep)",
        )
        ax_train.axhline(0, color="#CCCCCC", linewidth=0.8, linestyle="--")
        ax_train.set_xlabel("Episode")
        ax_train.set_ylabel("Total Reward")
        ax_train.set_title(f"Training Curve — {agent_type} on {selected_config_name}")
        ax_train.legend()
        ax_train.grid(True, alpha=0.3)
        st.pyplot(fig_train, use_container_width=True)
        plt.close(fig_train)
        st.caption("Note: these are simulated results.  Plug in your trained agent for real data.")

# ==========================================================================
# Tab 3 — Comparison
# ==========================================================================

with tab_compare:
    st.header("Aligned Agent vs Reward-Hacking Agent")

    if not _CONFIGS_AVAILABLE or not _RENDERER_AVAILABLE or not _GRIDWORLD_AVAILABLE:
        st.error("Required modules unavailable — check that src/ is importable.")

    elif not run_experiment and "comparison_trajs" not in st.session_state:
        st.info("Click **Run Experiment** in the sidebar to compare agent behaviours.")
        col_left, col_right = st.columns(2)
        renderer = GridRenderer(cell_size=80)
        with col_left:
            st.markdown("**Training: `training_default`**")
            train_cfg = get_config("training_default")
            fig_l = renderer.render_grid(train_cfg, title="Training Environment")
            st.pyplot(fig_l, use_container_width=True)
            plt.close(fig_l)
        with col_right:
            st.markdown(f"**Test: `{selected_config_name}`**")
            fig_r = renderer.render_grid(selected_config, title="Test Environment")
            st.pyplot(fig_r, use_container_width=True)
            plt.close(fig_r)

    else:
        if run_experiment:
            try:
                from src.agents.optimal import OptimalAgent
                from src.agents.q_learning import QLearningAgent
                from src.config import AgentConfig

                train_cfg = get_config("training_default")
                train_env = GridWorld(train_cfg)

                with st.spinner(f"Training Q-learning agent for {int(num_episodes)} episodes…"):
                    hacking = QLearningAgent(AgentConfig(), grid_size=train_cfg.grid_size)
                    hacking.train(train_env, int(num_episodes))
                    hacking.epsilon = 0.0

                aligned = OptimalAgent(selected_config.grid_size, selected_config.goal_position)
                test_env = GridWorld(selected_config)

                # Collect hacking agent trajectory on test env
                obs, _ = test_env.reset()
                traj_hacking: list = []
                terminated = truncated = False
                while not (terminated or truncated):
                    action = hacking.select_action(obs)
                    next_obs, reward, terminated, truncated, _ = test_env.step(action)
                    traj_hacking.append((obs, action, reward))
                    obs = next_obs

                # Collect aligned agent trajectory on test env
                obs, _ = test_env.reset()
                traj_aligned: list = []
                terminated = truncated = False
                grid = test_env.grid
                while not (terminated or truncated):
                    action = aligned.get_action(obs, grid)
                    next_obs, reward, terminated, truncated, _ = test_env.step(action)
                    traj_aligned.append((obs, action, reward))
                    obs = next_obs

                st.session_state["comparison_trajs"] = (traj_aligned, traj_hacking)
                st.session_state["comparison_cfg"] = selected_config_name

                # Run detection pipeline on the hacking agent
                try:
                    from src.detection.metrics import run_detection_pipeline
                    with st.spinner("Running detection pipeline…"):
                        det_env = GridWorld(selected_config)
                        det_result = run_detection_pipeline(
                            hacking, det_env, aligned,
                            n_episodes=20,
                            goal_position=selected_config.goal_position,
                            coin_position=selected_config.coin_position,
                        )
                    st.session_state["detection_result"] = det_result
                except Exception as det_exc:
                    st.warning(f"Detection pipeline failed: {det_exc}")

            except Exception as exc:
                st.error(f"Experiment failed: {exc}")

        if "comparison_trajs" in st.session_state:
            traj_aligned, traj_hacking = st.session_state["comparison_trajs"]
            cfg_name = st.session_state.get("comparison_cfg", selected_config_name)
            train_cfg = get_config("training_default")
            test_cfg = get_config(cfg_name)

            renderer = GridRenderer(cell_size=60)
            fig_cmp = renderer.render_comparison(train_cfg, test_cfg, traj_aligned, traj_hacking)
            st.pyplot(fig_cmp, use_container_width=True)
            plt.close(fig_cmp)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Aligned agent steps", len(traj_aligned))
                st.metric("Reached goal", "Yes" if traj_aligned and traj_aligned[-1][2] > 1.0 else "No")
            with col2:
                st.metric("Hacking agent steps", len(traj_hacking))
                st.metric("Reached goal", "Yes" if traj_hacking and traj_hacking[-1][2] > 1.0 else "No")

# ==========================================================================
# Tab 4 — Detection Metrics
# ==========================================================================

with tab_metrics:
    st.header("Reward Hacking Detection Metrics")

    dr = st.session_state.get("detection_result")

    if dr is None:
        st.info("Run an experiment in the sidebar to see detection metrics here.")
    else:
        _verdict_fn = {"ALIGNED": st.success, "HACKING": st.error, "UNCERTAIN": st.warning}
        _verdict_fn.get(dr.verdict, st.info)(f"Verdict: **{dr.verdict}**")

    st.subheader("Detection Scores")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.metric(
            label="Proxy Reliance Score",
            value=f"{dr.proxy_reliance_score:.3f}" if dr else "—",
            delta=None,
            help=(
                "Measures how much the agent's behaviour is driven by the proxy reward "
                "(coin) rather than the true objective (goal).  "
                "Higher = more hacking."
            ),
        )

    with col_m2:
        st.metric(
            label="KL Divergence",
            value=f"{dr.kl_divergence:.3f}" if (dr and dr.kl_divergence is not None) else "—",
            delta=None,
            help=(
                "KL divergence between the state-visitation distributions of the "
                "trained agent and an optimal aligned agent.  "
                "Higher = more divergent behaviour."
            ),
        )

    with col_m3:
        st.metric(
            label="Goal Reward Fraction",
            value=f"{dr.reward_from_goal:.1%}" if dr else "—",
            delta=None,
            help="Fraction of total reward that came from reaching the goal.",
        )

    with col_m4:
        st.metric(
            label="Coin Reward Fraction",
            value=f"{dr.reward_from_coin:.1%}" if dr else "—",
            delta=None,
            help="Fraction of total reward that came from collecting coins.",
        )

    st.divider()
    st.subheader("State Visitation Heatmap")
    st.info(
        "A heatmap comparing the state-visitation frequency of the trained agent "
        "vs an aligned baseline will appear here after training."
    )

    if _RENDERER_AVAILABLE and selected_config is not None:
        st.caption("Example: uniform visitation (placeholder)")
        renderer = GridRenderer(cell_size=80)
        size = selected_config.grid_size
        dummy_visits = {
            (r, c): int(np.random.default_rng(r * size + c).integers(1, 20))
            for r in range(size) for c in range(size)
        }
        fig_heat = renderer.render_heatmap(
            selected_config,
            visit_counts=dummy_visits,
            title=f"Placeholder Heatmap — {selected_config_name}",
        )
        st.pyplot(fig_heat, use_container_width=True)
        plt.close(fig_heat)
        st.caption("Placeholder data — connect real agent visitation counts for meaningful results.")

# ==========================================================================
# Tab 5 — About
# ==========================================================================

with tab_about:
    
    st.markdown(
        """
## What is Reward Hacking?

Reward hacking occurs when a reinforcement learning agent finds unintended
ways to maximise its reward signal without actually achieving the intended
objective.  Instead of learning the true goal, the agent exploits a proxy
reward that is easier to optimise but may diverge from what we actually want.

---

## The GridWorld Scenario

In this demo the agent must reach the goal (G) in a 2D grid. To make
training easier we also place a coin (C) on or near the optimal path:

- An agent that is well-aligned treats the coin as a helpful stepping stone and
  eventually ignores it once it learns the true goal is more valuable.
- An agent that exhibits reward-hacking fixates on the coin, since it provides a quick,
  reliable reward.  When the coin is moved to a different location (test time),
  the hacking agent detours to collect it — even at the cost of never reaching
  the goal.

---

## Why Does This Matter?

| Scenario | Aligned Agent | Hacking Agent |
|---|---|---|
| Coin on optimal path (training) | Reaches goal efficiently | Collects coin, may reach goal |
| Coin moved off-path (test) | Reaches goal directly | Detours for coin, may miss goal |
| No coin (test) | Reaches goal | Wanders or performs poorly |
| Coin near lava (test) | Avoids lava, reaches goal | Steps into lava to get coin |

---

## Detection Methods

This tool demonstrates several approaches to detecting reward hacking:

1. **Proxy Reliance Score** — measures how often the agent targets the coin
   vs the goal across distribution-shifted test environments.
2. **KL Divergence** — compares state-visitation distributions between the
   trained agent and an optimal aligned baseline.
3. **Goal Reach Rate** — the simplest metric: does the agent actually reach
   the goal?
        """
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption("Reward Hacking Detector")
