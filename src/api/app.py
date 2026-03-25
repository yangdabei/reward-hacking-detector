"""FastAPI server for the Reward Hacking Detector."""

import logging

from fastapi import BackgroundTasks, FastAPI, HTTPException

from src.config import ExperimentConfig
from src.storage import ExperimentStore

logger = logging.getLogger(__name__)

app = FastAPI(title="Reward Hacking Detector", version="0.1.0")
store = ExperimentStore()


def _background_train(experiment_id: str, config: ExperimentConfig) -> None:
    """Bridge between API and training pipeline."""
    store.update_status(experiment_id, "running")
    try:
        # TODO: instantiate agent + env from config and call run_training
        # Requires ML Exercises 1 & 2 (GridWorld + QLearningAgent) to be complete
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
    """Run the detection pipeline on a completed experiment."""
    experiment = store.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    if experiment["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Experiment is not completed (status: {experiment['status']})")
    # TODO: run detection pipeline once ML Exercises 4 & 5 are complete
    return {"message": "Detection pipeline not yet implemented"}
