"""Parameter sweep utilities: config generation, parallel execution, and result summarisation."""

from __future__ import annotations

import concurrent.futures
import itertools
import json
import logging
import pathlib

logger = logging.getLogger(__name__)


def generate_sweep_configs(
    coin_rewards: list[float] | None = None,
    grid_sizes: list[int] | None = None,
    num_episodes_list: list[int] | None = None,
    epsilon_decays: list[float] | None = None,
    agent_type: str = "q_learning",
) -> list[dict]:
    """Generate the full Cartesian product of sweep hyperparameters.

    Args:
        coin_rewards: Coin reward values to sweep over.
            Default: [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
        grid_sizes: Grid side lengths to sweep over.
            Default: [5, 7, 10]
        num_episodes_list: Episode counts to sweep over.
            Default: [500, 1000, 2000]
        epsilon_decays: Epsilon decay rates to sweep over.
            Default: [0.995] (single value — extend as needed)
        agent_type: Agent type string to embed in every config
            (e.g. "q_learning" or "dqn").

    Returns:
        List of config dicts, one per hyperparameter combination.
        Each dict contains keys: coin_reward, grid_size, num_episodes,
        epsilon_decay, agent_type, and a unique sweep_id.
    """
    if coin_rewards is None:
        coin_rewards = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
    if grid_sizes is None:
        grid_sizes = [5, 7, 10]
    if num_episodes_list is None:
        num_episodes_list = [500, 1000, 2000]
    if epsilon_decays is None:
        epsilon_decays = [0.995]

    configs: list[dict] = []
    for sweep_id, (cr, gs, ne, ed) in enumerate(
        itertools.product(coin_rewards, grid_sizes, num_episodes_list, epsilon_decays)
    ):
        configs.append(
            {
                "sweep_id": sweep_id,
                "agent_type": agent_type,
                "coin_reward": cr,
                "grid_size": gs,
                "num_episodes": ne,
                "epsilon_decay": ed,
            }
        )

    logger.info("Generated %d sweep configurations.", len(configs))
    return configs


def run_single_experiment(sweep_config: dict) -> dict:
    """Run one experiment defined by *sweep_config* and return a result dict.

    This function is designed to be called inside a worker process (via
    ProcessPoolExecutor).  It imports heavy dependencies lazily to avoid
    pickling issues.

    Args:
        sweep_config: Dict produced by generate_sweep_configs().  Must
            contain at least: coin_reward, grid_size, num_episodes,
            epsilon_decay, agent_type.

    Returns:
        Dict containing all sweep_config keys plus:
          - "mean_reward": float
          - "goal_rate": float
          - "proxy_reliance_score": float
          - "verdict": str ("ALIGNED" | "HACKING" | "UNCERTAIN")
          - "error": str or None — error message if the experiment failed
    """
    result = dict(sweep_config)
    result["error"] = None

    try:
        # Lazy imports — these may not be available in all environments
        # and we want clean error reporting per experiment.
        from src.detection.metrics import compute_proxy_reliance_score, determine_verdict
        from src.experiments.evaluate import evaluate_agent

        # Build a minimal AgentConfig-like namespace
        class _Cfg:
            epsilon_start = 1.0
            epsilon_end = 0.05
            epsilon_decay = sweep_config.get("epsilon_decay", 0.995)
            lr = 0.1
            gamma = 0.99

        grid_size: int = sweep_config["grid_size"]
        num_episodes: int = sweep_config["num_episodes"]
        agent_type: str = sweep_config.get("agent_type", "q_learning")
        coin_reward: float = sweep_config["coin_reward"]

        # Attempt to build environment and agent
        # The exact env/agent constructors depend on what is available in
        # the project; we wrap in try/except so one broken config does not
        # abort the whole sweep.
        try:
            import gymnasium as gym
            env_id = f"RewardHacking-{grid_size}x{grid_size}-v0"
            env = gym.make(env_id, coin_reward=coin_reward)
        except Exception:
            # Fall back to any registered env or raise a clear error
            raise RuntimeError(
                f"Could not create environment for grid_size={grid_size}. "
                "Ensure the reward_hacking gymnasium environment is registered."
            )

        if agent_type == "q_learning":
            from src.agents.q_learning import QLearningAgent
            agent = QLearningAgent(config=_Cfg(), grid_size=grid_size)
        elif agent_type == "dqn":
            from src.agents.dqn import DQNAgent
            agent = DQNAgent(config=_Cfg(), grid_size=grid_size)
        else:
            raise ValueError(f"Unknown agent_type: {agent_type!r}")

        # Train
        from src.experiments.train import run_training
        train_result = run_training(agent, env, num_episodes)

        # Evaluate
        eval_result = evaluate_agent(agent, env, n_episodes=50)

        # Detection
        goal_pos = (grid_size - 1, grid_size - 1)
        coin_pos_attr = getattr(env, "coin_position", None)
        proxy_score = compute_proxy_reliance_score(
            eval_result["trajectories"], goal_pos, coin_pos_attr
        )
        verdict = determine_verdict(proxy_score)

        result.update(
            {
                "mean_reward": eval_result["mean_reward"],
                "goal_rate": eval_result["goal_rate"],
                "proxy_reliance_score": proxy_score,
                "verdict": verdict,
                "final_epsilon": train_result["final_epsilon"],
                "total_time_s": train_result["total_time_s"],
            }
        )

    except Exception as exc:
        logger.error(
            "Experiment sweep_id=%s failed: %s",
            sweep_config.get("sweep_id", "?"),
            exc,
        )
        result["error"] = str(exc)
        result.setdefault("mean_reward", float("nan"))
        result.setdefault("goal_rate", float("nan"))
        result.setdefault("proxy_reliance_score", float("nan"))
        result.setdefault("verdict", "ERROR")

    return result


def run_sweep(
    sweep_configs: list[dict],
    max_workers: int = 4,
    output_dir: pathlib.Path | None = None,
) -> list[dict]:
    """Execute a parameter sweep in parallel using multiple processes.

    Args:
        sweep_configs: List of config dicts from generate_sweep_configs().
        max_workers: Maximum number of parallel worker processes (default 4).
        output_dir: Directory to write sweep_results.json.  Created if it
            does not exist.  Pass None to skip saving.

    Returns:
        List of result dicts, one per sweep configuration, in the order
        they were submitted (not necessarily completion order).
    """
    logger.info(
        "Starting sweep with %d configs across up to %d workers.",
        len(sweep_configs),
        max_workers,
    )

    results: list[dict] = [{}] * len(sweep_configs)  # pre-allocate to preserve order

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(run_single_experiment, cfg): idx
            for idx, cfg in enumerate(sweep_configs)
        }

        completed = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.error("Future for config index %d raised: %s", idx, exc)
                results[idx] = {**sweep_configs[idx], "error": str(exc)}

            completed += 1
            if completed % 10 == 0 or completed == len(sweep_configs):
                logger.info("Sweep progress: %d/%d complete.", completed, len(sweep_configs))

    # Save results
    if output_dir is not None:
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "sweep_results.json"
        try:
            out_path.write_text(json.dumps(results, indent=2, default=str))
            logger.info("Sweep results saved to %s", out_path)
        except Exception as exc:
            logger.warning("Failed to save sweep results: %s", exc)

    return results


def summarize_sweep(results: list[dict]) -> dict:
    """Compute summary statistics across all sweep results.

    Ignores experiments that encountered errors.

    Args:
        results: List of result dicts returned by run_sweep().

    Returns:
        Dict with keys:
          - "n_total": total number of experiments
          - "n_errors": number of failed experiments
          - "n_hacking": number classified as HACKING
          - "n_aligned": number classified as ALIGNED
          - "n_uncertain": number classified as UNCERTAIN
          - "mean_proxy_reliance": mean proxy reliance across successful runs
          - "mean_goal_rate": mean goal rate across successful runs
          - "mean_reward": mean total reward across successful runs
          - "hacking_threshold_coin_reward": lowest coin_reward where majority
            verdict is HACKING (None if not found)
          - "key_findings": list[str] of human-readable summary bullets
    """
    n_total = len(results)
    successful = [r for r in results if r.get("error") is None]
    n_errors = n_total - len(successful)

    verdicts = [r.get("verdict", "ERROR") for r in successful]
    n_hacking = verdicts.count("HACKING")
    n_aligned = verdicts.count("ALIGNED")
    n_uncertain = verdicts.count("UNCERTAIN")

    proxy_scores = [r["proxy_reliance_score"] for r in successful if "proxy_reliance_score" in r]
    goal_rates = [r["goal_rate"] for r in successful if "goal_rate" in r]
    mean_rewards = [r["mean_reward"] for r in successful if "mean_reward" in r]

    mean_proxy = float(sum(proxy_scores) / len(proxy_scores)) if proxy_scores else float("nan")
    mean_goal = float(sum(goal_rates) / len(goal_rates)) if goal_rates else float("nan")
    mean_rew = float(sum(mean_rewards) / len(mean_rewards)) if mean_rewards else float("nan")

    # Find threshold coin_reward where HACKING becomes majority
    hacking_threshold: float | None = None
    by_coin_reward: dict[float, list[str]] = {}
    for r in successful:
        cr = r.get("coin_reward")
        if cr is not None:
            by_coin_reward.setdefault(cr, []).append(r.get("verdict", "UNCERTAIN"))

    for cr in sorted(by_coin_reward.keys()):
        verd_list = by_coin_reward[cr]
        if verd_list.count("HACKING") > len(verd_list) / 2:
            hacking_threshold = cr
            break

    key_findings = [
        f"Total experiments: {n_total} ({n_errors} errors)",
        f"Verdicts — HACKING: {n_hacking}, ALIGNED: {n_aligned}, UNCERTAIN: {n_uncertain}",
        f"Mean proxy reliance score: {mean_proxy:.4f}",
        f"Mean goal rate: {mean_goal:.4f}",
        f"Mean episode reward: {mean_rew:.2f}",
    ]
    if hacking_threshold is not None:
        key_findings.append(
            f"HACKING becomes majority verdict at coin_reward >= {hacking_threshold}"
        )
    else:
        key_findings.append("No clear coin_reward threshold for HACKING majority found.")

    return {
        "n_total": n_total,
        "n_errors": n_errors,
        "n_hacking": n_hacking,
        "n_aligned": n_aligned,
        "n_uncertain": n_uncertain,
        "mean_proxy_reliance": mean_proxy,
        "mean_goal_rate": mean_goal,
        "mean_reward": mean_rew,
        "hacking_threshold_coin_reward": hacking_threshold,
        "key_findings": key_findings,
    }
