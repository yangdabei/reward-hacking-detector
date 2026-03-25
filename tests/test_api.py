"""
Tests for the FastAPI server.

Run with: pytest tests/test_api.py -v
"""
import httpx
import pytest
from httpx import ASGITransport

from src.api.app import app

transport = ASGITransport(app=app)


@pytest.mark.asyncio
async def test_health_endpoint():
    """GET /health should return 200 with status ok."""
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_create_experiment():
    """POST /experiments should return 201 with experiment_id."""
    config = {"env": {}, "agent": {}, "agent_type": "q_learning", "num_episodes": 10, "seed": 42}
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/experiments", json=config)
    assert response.status_code == 201
    assert "experiment_id" in response.json()


@pytest.mark.asyncio
async def test_get_nonexistent_experiment():
    """GET /experiments/fake-id should return 404."""
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/experiments/nonexistent-id-12345")
    assert response.status_code == 404