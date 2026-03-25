"""Command-line interface for the Reward Hacking Detector."""

import argparse
import json
import logging
from pathlib import Path

from src.config import AgentConfig, EnvConfig, ExperimentConfig
from src.experiments.sweep import run_sweep
from src.logging_config import setup_logging
from src.storage import ExperimentStore

logger = logging.getLogger(__name__)


def _load_config(args: argparse.Namespace) -> ExperimentConfig:
    """Load ExperimentConfig from --config name or --config-file path."""
    if hasattr(args, "config_file") and args.config_file:
        return ExperimentConfig.model_validate_json(args.config_file.read_text())
    return ExperimentConfig(
        env=EnvConfig(),
        agent=AgentConfig(),
        agent_type=args.agent if hasattr(args, "agent") else "q_learning",
        num_episodes=args.episodes if hasattr(args, "episodes") else 1000,
        seed=args.seed if hasattr(args, "seed") else 42,
    )


def handle_train(args: argparse.Namespace) -> None:
    """Train an agent and save the experiment to the store."""
    setup_logging()
    config = _load_config(args)
    store = ExperimentStore()
    experiment_id = store.create_experiment(config.model_dump())
    store.update_status(experiment_id, "running")
    logger.info(f"Started experiment {experiment_id} (agent={args.agent}, episodes={args.episodes})")
    try:
        # TODO: instantiate GridWorld + agent and call run_training
        # Requires ML Exercise 1 (GridWorld) and ML Exercise 2/3 (agents)
        store.update_status(experiment_id, "completed")
        logger.info(f"Training complete. Experiment ID: {experiment_id}")
    except Exception as e:
        store.update_status(experiment_id, "failed")
        logger.error(f"Training failed: {e}")


def handle_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a trained agent across test environments."""
    setup_logging()
    store = ExperimentStore()
    experiment = store.get_experiment(args.experiment_id)
    if experiment is None:
        logger.error(f"Experiment {args.experiment_id} not found")
        return
    logger.info(f"Evaluating experiment {args.experiment_id} on {args.test_env}")
    # TODO: load agent from experiment, run evaluate_across_configs
    # Requires ML Exercise 1 (GridWorld) and ML Exercise 2/3 (agents)
    logger.info("Evaluation complete")


def handle_detect(args: argparse.Namespace) -> None:
    """Run the detection pipeline on a completed experiment."""
    setup_logging()
    store = ExperimentStore()
    experiment = store.get_experiment(args.experiment_id)
    if experiment is None:
        logger.error(f"Experiment {args.experiment_id} not found")
        return
    if experiment["status"] != "completed":
        logger.error(f"Experiment is not completed (status: {experiment['status']})")
        return
    logger.info(f"Running detection on experiment {args.experiment_id}")
    # TODO: run detection pipeline once ML Exercises 4 & 5 are complete
    logger.info("Detection complete")


def handle_sweep(args: argparse.Namespace) -> None:
    """Run a parameter sweep."""
    setup_logging()
    sweep_config = []
    if args.sweep_file and args.sweep_file.exists():
        sweep_config = json.loads(args.sweep_file.read_text())
    results = run_sweep(sweep_config, max_workers=args.workers)
    logger.info(f"Sweep complete: {len(results)} experiments")


def handle_list(args: argparse.Namespace) -> None:
    """List experiments from the store."""
    setup_logging()
    store = ExperimentStore()
    experiments = store.list_experiments(status=args.status)
    if not experiments:
        logger.info("No experiments found")
        return
    for exp in experiments:
        logger.info(f"{exp['id']} | {exp['status']} | {exp['created_at']}")


def main() -> None:
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Reward Hacking Detector")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train an agent")
    train_parser.add_argument(
        "--config",
        choices=["training_default", "training_large", "training_multi_coin"],
        default="training_default",
    )
    train_parser.add_argument("--config-file", type=Path, help="Custom config JSON file")
    train_parser.add_argument("--agent", choices=["q_learning", "dqn"], default="q_learning")
    train_parser.add_argument("--episodes", type=int, default=1000)
    train_parser.add_argument("--seed", type=int, default=42)

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained agent")
    eval_parser.add_argument("--experiment-id", required=True)
    eval_parser.add_argument(
        "--test-env",
        choices=["test_coin_moved", "test_no_coin", "test_coin_near_lava"],
        default="test_coin_moved",
    )

    # Detect subcommand
    detect_parser = subparsers.add_parser("detect", help="Run detection pipeline")
    detect_parser.add_argument("--experiment-id", required=True)

    # Sweep subcommand
    sweep_parser = subparsers.add_parser("sweep", help="Run parameter sweep")
    sweep_parser.add_argument("--sweep-file", type=Path)
    sweep_parser.add_argument("--workers", type=int, default=4)

    # List subcommand
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--status", choices=["pending", "running", "completed", "failed"])

    args = parser.parse_args()

    handlers = {
        "train": handle_train,
        "evaluate": handle_evaluate,
        "detect": handle_detect,
        "sweep": handle_sweep,
        "list": handle_list,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
