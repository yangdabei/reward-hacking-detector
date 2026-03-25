"""Experiment storage using SQLite."""

import json
import sqlite3
import uuid
from pathlib import Path


class ExperimentStore:
    """SQLite-backed storage for experiments, results, and trajectories."""

    def __init__(self, db_path: Path = Path("data/experiments.db")) -> None:
        """Initialise the store and create tables if they don't exist.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _create_tables(self) -> None:
        """Create the database schema if it doesn't already exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    config JSON NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL REFERENCES experiments(id),
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metadata JSON
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trajectories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL REFERENCES experiments(id),
                    env_config_name TEXT NOT NULL,
                    trajectory JSON NOT NULL
                )
            """)

    def create_experiment(self, config: dict) -> str:
        """Create a new experiment record and return its ID."""
        experiment_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO experiments (id, config) VALUES (?, ?)",
                (experiment_id, json.dumps(config)),
            )
        return experiment_id

    def update_status(self, experiment_id: str, status: str) -> None:
        """Update the status of an experiment."""
        with sqlite3.connect(self.db_path) as conn:
            if status == "completed":
                conn.execute(
                    "UPDATE experiments SET status = ?, completed_at = CURRENT_TIMESTAMP WHERE id = ?",  # noqa: E501
                    (status, experiment_id),
                )
            else:
                conn.execute(
                    "UPDATE experiments SET status = ? WHERE id = ?",
                    (status, experiment_id),
                )

    def save_results(self, experiment_id: str, metrics: dict[str, float]) -> None:
        """Save metric results for an experiment."""
        with sqlite3.connect(self.db_path) as conn:
            for metric_name, metric_value in metrics.items():
                conn.execute(
                    "INSERT INTO results (experiment_id, metric_name, metric_value) VALUES (?, ?, ?)",  # noqa: E501
                    (experiment_id, metric_name, metric_value),
                )

    def save_trajectory(self, experiment_id: str, env_name: str, trajectory: list) -> None:
        """Save a trajectory for an experiment."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO trajectories (experiment_id, env_config_name, trajectory) VALUES (?, ?, ?)",  # noqa: E501
                (experiment_id, env_name, json.dumps(trajectory)),
            )

    def get_experiment(self, experiment_id: str) -> dict | None:
        """Retrieve a single experiment by ID."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, config, status, created_at, completed_at FROM experiments WHERE id = ?",  # noqa: E501
                (experiment_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "config": json.loads(row[1]),
            "status": row[2],
            "created_at": row[3],
            "completed_at": row[4],
        }

    def list_experiments(self, status: str | None = None) -> list[dict]:
        """List all experiments, optionally filtered by status."""
        with sqlite3.connect(self.db_path) as conn:
            if status is None:
                rows = conn.execute(
                    "SELECT id, config, status, created_at, completed_at FROM experiments"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, config, status, created_at, completed_at FROM experiments WHERE status = ?",  # noqa: E501
                    (status,),
                ).fetchall()
        return [
            {
                "id": row[0],
                "config": json.loads(row[1]),
                "status": row[2],
                "created_at": row[3],
                "completed_at": row[4],
            }
            for row in rows
        ]
