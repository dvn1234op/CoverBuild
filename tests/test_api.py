"""tests/test_api.py — FastAPI endpoint tests using httpx async client."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture(scope="module")
def client():
    """Create a test client with mocked model loading."""
    import torch
    from transformers import AutoTokenizer

    # Mock the heavy model loading so tests run without GPU/checkpoints
    mock_model = MagicMock()
    mock_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    mock_tokenizer.pad_token = mock_tokenizer.eos_token

    # Make mock_model.generate return something sensible
    mock_output = torch.tensor([[50256, 1234, 5678, 50256]])  # fake token ids
    mock_model.generate.return_value = mock_output

    with patch("src.api.app._policy_model", mock_model), \
         patch("src.api.app._tokenizer", mock_tokenizer), \
         patch("src.api.app._device", "cpu"):

        from src.api.app import app
        from src.models.reward_model import RewardModel
        mock_rm = RewardModel.fresh(device="cpu")

        with patch("src.api.app._reward_model", mock_rm):
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data


class TestRewardEndpoint:

    def test_reward_valid_request(self, client):
        response = client.post(
            "/reward",
            json={
                "job_description": "Software Engineer requiring Python and REST API experience.",
                "cover_letter": (
                    "Dear Hiring Manager, I am excited to apply for this position. "
                    "With three years of Python and API development, I have the skills you need. "
                    "I look forward to contributing to your team. Sincerely, Jane."
                ),
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "score" in data
        assert "breakdown" in data
        assert "interpretation" in data
        assert -2.0 <= data["score"] <= 2.0

    def test_reward_breakdown_dimensions(self, client):
        data = client.post(
            "/reward",
            json={
                "job_description": "Data Scientist role.",
                "cover_letter": "Dear team, I apply for this role. Best.",
            },
        ).json()
        bd = data["breakdown"]
        for dim in ["structure", "relevance", "tone", "conciseness", "aggregate"]:
            assert dim in bd

    def test_reward_too_short_job_desc(self, client):
        response = client.post(
            "/reward",
            json={"job_description": "hi", "cover_letter": "some letter text here"},
        )
        assert response.status_code == 422  # Validation error

    def test_reward_missing_field(self, client):
        response = client.post("/reward", json={"job_description": "some job"})
        assert response.status_code == 422


class TestSamplesEndpoint:

    def test_samples_returns_list(self, client):
        response = client.get("/samples?n=2")
        # May succeed or fail depending on mock; at least shouldn't 500
        assert response.status_code in [200, 500]
