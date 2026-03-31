"""FastAPI server for the Reward Hacking Detector."""

import logging
from pathlib import Path

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException

from src.agents.dqn import DQNAgent
from src.agents.optimal import OptimalAgent
from src.agents.q_learning import QLearningAgent
from src.config import ExperimentConfig
from src.detection.metrics import run_detection_pipeline
from src.environment.configs import get_config
from src.environment.gridworld import GridWorld
from src.experiments.train import run_training
from src.storage import ExperimentStore

logger = logging.getLogger(__name__)

app = FastAPI(title="Reward Hacking Detector", version="0.1.0")
store = ExperimentStore()


def _build_agent(config: ExperimentConfig) -> QLearningAgent | DQNAgent:
    """Instantiate the correct agent class from an ExperimentConfig."""
    if config.agent_type == "q_learning":
        return QLearningAgent(config.agent, grid_size=config.env.grid_size)
    return DQNAgent(config.agent, grid_size=config.env.grid_size)


def _checkpoint_path(experiment_id: str, agent_type: str) -> Path:
    """Return the model checkpoint path for a given experiment."""
    ext = ".json" if agent_type == "q_learning" else ".pt"
    return Path(f"data/checkpoints/{experiment_id}/model{ext}")


def _background_train(experiment_id: str, config: ExperimentConfig) -> None:
    """Run training in the background and persist results."""
    store.update_status(experiment_id, "running")
    try:
        agent = _build_agent(config)
        env = GridWorld(config.env)
        checkpoint_dir = Path(f"data/checkpoints/{experiment_id}")
        result = run_training(agent, env, config.num_episodes, checkpoint_dir=checkpoint_dir)
        agent.save(_checkpoint_path(experiment_id, config.agent_type))
        store.save_results(experiment_id, {
            "mean_reward_last100": float(np.mean(result["episode_rewards"][-100:])),
            "final_epsilon": result["final_epsilon"],
            "total_time_s": result["total_time_s"],
        })
        store.update_status(experiment_id, "completed")
    except Exception as e:
        logger.error(f"Training failed for experiment {experiment_id}: {e}")
        store.update_status(experiment_id, "failed")


@app.get("/health")
async def health() -> dict:
    """Check that the server is running."""
    return {"status": "ok", "experiments_count": len(store.list_experiments())}


@app.post("/experiments", status_code=201)
async def create_experiment(config: ExperimentConfig, background_tasks: BackgroundTasks) -> dict:
    """Create a new experiment and start training in the background."""
    experiment_id = store.create_experiment(config.model_dump())
    store.update_status(experiment_id, "pending")
    background_tasks.add_task(_background_train, experiment_id, config)
    return {"experiment_id": experiment_id, "status": "pending"}


@app.get("/experiments")
async def list_experiments(status: str | None = None) -> list:
    """List all experiments, optionally filtered by status."""
    return store.list_experiments(status)


@app.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str) -> dict:
    """Get a single experiment by ID."""
    experiment = store.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@app.post("/experiments/{experiment_id}/detect")
async def run_detection(experiment_id: str) -> dict:
    """Run the reward-hacking detection pipeline on a completed experiment."""
    experiment = store.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if experiment["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Experiment is not completed (status: {experiment['status']})",
        )
    config = ExperimentConfig.model_validate(experiment["config"])
    agent = _build_agent(config)
    checkpoint = _checkpoint_path(experiment_id, config.agent_type)
    if checkpoint.exists():
        agent.load(checkpoint)
    reference = OptimalAgent(config.env.grid_size, config.env.goal_position)
    test_env = GridWorld(get_config("test_coin_moved"))
    result = run_detection_pipeline(
        agent, test_env, reference,
        goal_position=config.env.goal_position,
        coin_position=config.env.coin_position,
    )
    return {
        "verdict": result.verdict,
        "proxy_reliance_score": result.proxy_reliance_score,
        "kl_divergence": result.kl_divergence,
        "reward_from_goal": result.reward_from_goal,
        "reward_from_coin": result.reward_from_coin,
        "n_episodes": result.n_episodes,
    }
