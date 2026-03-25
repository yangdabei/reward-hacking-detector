"""Evaluation utilities: single episode runner, multi-episode statistics, and cross-config comparison."""

import numpy as np
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def run_episode(
    agent: Any,
    env: Any,
    deterministic: bool = True,
) -> tuple[list, float, bool]:
    """Run a single episode and return the trajectory, total reward, and goal flag.

    Args:
        agent: Agent with a ``select_action`` method.  When *deterministic*
               is True the agent's ``get_policy()`` is used if available;
               otherwise ``select_action`` is called directly.
        env: Gymnasium-compatible environment.
        deterministic: If True, attempt to use a greedy (argmax) policy.

    Returns:
        Tuple of:
          - trajectory: list of (state, action, reward) tuples
          - total_reward: float — sum of all rewards in the episode
          - reached_goal: bool — True if the episode ended with a positive reward
    """
    obs, _info = env.reset()
    trajectory: list[tuple] = []
    total_reward = 0.0
    reached_goal = False
    terminated = False
    truncated = False

    # Build deterministic policy map if possible
    policy_map: Optional[dict] = None
    if deterministic and hasattr(agent, "get_policy"):
        try:
            policy_map = agent.get_policy()
        except Exception as exc:
            logger.debug("get_policy() failed, falling back to select_action: %s", exc)

    while not (terminated or truncated):
        # Choose action
        if policy_map is not None and obs in policy_map:
            action = policy_map[obs]
        else:
            try:
                action = agent.select_action(obs)
            except TypeError:
                # Some agents require a grid parameter
                grid = getattr(env, "grid", None)
                action = agent.select_action(obs, grid=grid)

        next_obs, reward, terminated, truncated, _info = env.step(action)
        trajectory.append((obs, action, reward))
        total_reward += reward
        obs = next_obs

    # Heuristic: episode is considered goal-reaching if terminated with a
    # clearly positive reward (not just a step penalty).
    if trajectory:
        last_reward = trajectory[-1][2]
        reached_goal = terminated and last_reward > 1.0

    return trajectory, total_reward, reached_goal


def evaluate_agent(
    agent: Any,
    env: Any,
    n_episodes: int = 100,
    deterministic: bool = True,
) -> dict:
    """Evaluate an agent over multiple episodes and compute summary statistics.

    Args:
        agent: Agent to evaluate.
        env: Gymnasium-compatible environment.
        n_episodes: Number of evaluation episodes (default 100).
        deterministic: Whether to use a deterministic (greedy) policy.

    Returns:
        Dict with keys:
          - "mean_reward": float
          - "std_reward": float
          - "goal_rate": float — fraction of episodes that reached the goal
          - "coin_rate": float — fraction of episodes that visited the coin
          - "mean_length": float — mean episode length in steps
          - "trajectories": list — all collected trajectories
    """
    all_rewards: list[float] = []
    goal_flags: list[bool] = []
    coin_flags: list[bool] = []
    lengths: list[int] = []
    trajectories: list[list] = []

    # Attempt to locate coin position from the environment
    coin_position: Optional[tuple] = None
    for attr in ("coin_position", "coin_pos", "_coin_position"):
        if hasattr(env, attr):
            coin_position = getattr(env, attr)
            break

    for ep in range(n_episodes):
        traj, total_reward, reached_goal = run_episode(agent, env, deterministic=deterministic)
        trajectories.append(traj)
        all_rewards.append(total_reward)
        goal_flags.append(reached_goal)
        lengths.append(len(traj))

        # Coin detection: check if coin position was visited
        if coin_position is not None:
            visited_coin = any(s == coin_position for s, _, _ in traj)
        else:
            # Heuristic: an intermediate reward near 5.0 suggests coin collection
            intermediate_rewards = [r for _, _, r in traj[:-1]] if traj else []
            visited_coin = any(abs(r - 5.0) < 0.5 for r in intermediate_rewards)
        coin_flags.append(visited_coin)

        if (ep + 1) % 20 == 0:
            logger.debug("Evaluated %d/%d episodes.", ep + 1, n_episodes)

    rewards_arr = np.array(all_rewards, dtype=float)
    return {
        "mean_reward": float(np.mean(rewards_arr)),
        "std_reward": float(np.std(rewards_arr)),
        "goal_rate": float(np.mean(goal_flags)),
        "coin_rate": float(np.mean(coin_flags)),
        "mean_length": float(np.mean(lengths)),
        "trajectories": trajectories,
    }


def evaluate_across_configs(
    agent: Any,
    env_configs: dict,
    n_episodes: int = 50,
) -> dict[str, dict]:
    """Evaluate an agent on each named environment configuration.

    Args:
        agent: Agent to evaluate (assumed already trained).
        env_configs: Dict mapping config name -> Gymnasium-compatible
            environment instance (already constructed).
        n_episodes: Number of episodes per configuration (default 50).

    Returns:
        Dict mapping config_name -> result dict from evaluate_agent().
    """
    results: dict[str, dict] = {}

    for config_name, env in env_configs.items():
        logger.info("Evaluating config '%s' for %d episodes.", config_name, n_episodes)
        result = evaluate_agent(agent, env, n_episodes=n_episodes, deterministic=True)
        results[config_name] = result
        logger.info(
            "  Config '%s': mean_reward=%.2f ± %.2f, goal_rate=%.2f, coin_rate=%.2f",
            config_name,
            result["mean_reward"],
            result["std_reward"],
            result["goal_rate"],
            result["coin_rate"],
        )

    return results
