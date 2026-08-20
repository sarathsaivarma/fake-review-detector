"""
Basic smoke tests for the FastAPI service. These mock out model loading so
CI doesn't need a real MLflow registry / GPU to validate the API contract.
"""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("src.serve.app.mlflow"), patch("src.serve.app.load_model"):
        from src.serve.app import app, _predict_one  # noqa
        import src.serve.app as app_module

        app_module._model = MagicMock()
        app_module._tokenizer = MagicMock()
        app_module._model_version = "test-1"

        with patch("src.serve.app._predict_one", return_value=(0.87, 12.3)):
            yield TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_single(client):
    resp = client.post("/predict", json={"review_text": "Amazing!!! Best product ever!!!"})
    assert resp.status_code == 200
    body = resp.json()
    assert "fraud_probability" in body
    assert body["is_fake"] is True
    assert 0.0 <= body["fraud_probability"] <= 1.0


def test_predict_batch(client):
    resp = client.post("/predict/batch", json={
        "reviews": [
            {"review_text": "Great quality, works as expected."},
            {"review_text": "BEST THING EVER BUY NOW!!!"},
        ]
    })
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_predict_empty_text_rejected(client):
    resp = client.post("/predict", json={"review_text": ""})
    assert resp.status_code == 422
