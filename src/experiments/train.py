"""Training utilities: wraps agent training loops, handles checkpointing, and returns structured results."""

from __future__ import annotations

import logging
import pathlib
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


def run_training(
    agent: Any,
    env: Any,
    num_episodes: int,
    checkpoint_dir: Optional[pathlib.Path] = None,
    log_interval: int = 100,
) -> dict:
    """Run training for the given agent and environment.

    If the agent exposes a ``train(env, num_episodes)`` method it is called
    directly.  Otherwise a generic Gymnasium-compatible manual loop is used.
    Checkpoints are saved every 250 episodes when *checkpoint_dir* is given.

    Args:
        agent: Any agent with ``select_action``, optionally ``train``,
               ``decay_epsilon``, and ``save`` methods.
        env: Gymnasium-compatible training environment.
        num_episodes: Total number of training episodes.
        checkpoint_dir: Directory for periodic checkpoints.  Created if it
            does not exist.  Pass None to skip checkpointing.
        log_interval: Log average reward every this many episodes.

    Returns:
        Dict with keys:
          - "episode_rewards": list[float] — total reward per episode
          - "final_epsilon": float — epsilon value after training
          - "total_time_s": float — wall-clock training time in seconds
    """
    start_time = time.time()

    if checkpoint_dir is not None:
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Delegate to agent.train() if available ---
    if hasattr(agent, "train") and callable(agent.train):
        logger.info("Delegating to agent.train() for %d episodes.", num_episodes)
        episode_rewards: list[float] = agent.train(env, num_episodes)
    else:
        # --- Manual training loop ---
        logger.info("Running manual training loop for %d episodes.", num_episodes)
        episode_rewards = []
        checkpoint_interval = 250

        for ep in range(num_episodes):
            obs, _info = env.reset()
            ep_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)

                if hasattr(agent, "update"):
                    agent.update(obs, action, reward, next_obs, terminated or truncated)

                obs = next_obs
                ep_reward += reward

            episode_rewards.append(ep_reward)

            if hasattr(agent, "decay_epsilon"):
                agent.decay_epsilon()

            if (ep + 1) % log_interval == 0:
                window = episode_rewards[max(0, ep + 1 - log_interval):]
                avg = sum(window) / len(window)
                epsilon = getattr(agent, "epsilon", float("nan"))
                logger.info(
                    "Episode %d/%d — avg_reward=%.2f, epsilon=%.4f",
                    ep + 1,
                    num_episodes,
                    avg,
                    epsilon,
                )

            # Periodic checkpointing
            if (
                checkpoint_dir is not None
                and hasattr(agent, "save")
                and (ep + 1) % checkpoint_interval == 0
            ):
                ckpt_path = checkpoint_dir / f"checkpoint_ep{ep + 1}.json"
                try:
                    agent.save(ckpt_path)
                    logger.info("Checkpoint saved to %s", ckpt_path)
                except Exception as exc:
                    logger.warning("Checkpoint save failed at episode %d: %s", ep + 1, exc)

    total_time = time.time() - start_time
    final_epsilon = float(getattr(agent, "epsilon", 0.0))

    logger.info(
        "Training complete: %d episodes in %.1f s (final epsilon=%.4f)",
        num_episodes,
        total_time,
        final_epsilon,
    )

    # Save final checkpoint
    if checkpoint_dir is not None and hasattr(agent, "save"):
        final_ckpt = checkpoint_dir / "checkpoint_final.json"
        try:
            agent.save(final_ckpt)
            logger.info("Final checkpoint saved to %s", final_ckpt)
        except Exception as exc:
            logger.warning("Final checkpoint save failed: %s", exc)

    return {
        "episode_rewards": episode_rewards,
        "final_epsilon": final_epsilon,
        "total_time_s": total_time,
    }


def train_and_evaluate(
    agent: Any,
    train_env: Any,
    test_envs: dict,
    num_episodes: int,
    checkpoint_dir: Optional[pathlib.Path] = None,
) -> dict:
    """Train an agent and then evaluate it across multiple test environments.

    Args:
        agent: Agent to train and evaluate.
        train_env: Gymnasium-compatible training environment.
        test_envs: Dict mapping environment name -> environment instance.
        num_episodes: Number of training episodes.
        checkpoint_dir: Optional directory for training checkpoints.

    Returns:
        Dict containing:
          - "training": result dict from run_training()
          - "evaluation": dict mapping env_name -> evaluation summary dict
              Each summary has keys: "mean_reward", "std_reward",
              "goal_rate", "min_reward", "max_reward".
    """
    logger.info("Starting training for %d episodes.", num_episodes)
    training_result = run_training(
        agent=agent,
        env=train_env,
        num_episodes=num_episodes,
        checkpoint_dir=checkpoint_dir,
    )

    evaluation_results: dict[str, dict] = {}
    eval_episodes = 20

    for env_name, test_env in test_envs.items():
        logger.info("Evaluating on '%s' for %d episodes.", env_name, eval_episodes)
        rewards: list[float] = []
        goal_count = 0

        for _ in range(eval_episodes):
            obs, _info = test_env.reset()
            ep_reward = 0.0
            terminated = False
            truncated = False
            reached_goal = False

            while not (terminated or truncated):
                # Prefer deterministic greedy action during evaluation
                if hasattr(agent, "get_policy"):
                    try:
                        policy = agent.get_policy()
                        action = policy.get(obs, agent.select_action(obs))
                    except Exception:
                        action = agent.select_action(obs)
                else:
                    action = agent.select_action(obs)

                obs, reward, terminated, truncated, info = test_env.step(action)
                ep_reward += reward

                if terminated and reward > 0:
                    reached_goal = True

            rewards.append(ep_reward)
            if reached_goal:
                goal_count += 1

        arr = [float(r) for r in rewards]
        evaluation_results[env_name] = {
            "mean_reward": float(sum(arr) / len(arr)) if arr else 0.0,
            "std_reward": float(
                (sum((r - sum(arr) / len(arr)) ** 2 for r in arr) / len(arr)) ** 0.5
            ) if arr else 0.0,
            "goal_rate": goal_count / eval_episodes,
            "min_reward": min(arr) if arr else 0.0,
            "max_reward": max(arr) if arr else 0.0,
        }
        logger.info(
            "  %s — mean_reward=%.2f, goal_rate=%.2f",
            env_name,
            evaluation_results[env_name]["mean_reward"],
            evaluation_results[env_name]["goal_rate"],
        )

    return {
        "training": training_result,
        "evaluation": evaluation_results,
    }
