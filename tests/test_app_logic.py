"""
test_app_logic.py
-------------------
Tests the thin application-logic layer that app.py exposes as a Flask
API, without needing a running server (Flask's test client is enough).
"""

import os

import pytest

MODEL_EXISTS = os.path.exists("models/phishing_model.joblib")


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet; run "
                     "'python -m src.train_model' first.")
class TestFlaskAPI:

    @pytest.fixture(scope="class")
    def client(self):
        from app import app
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_predict_endpoint_valid_url(self, client):
        resp = client.post("/api/predict", json={"url": "https://www.wikipedia.org"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["prediction"] in ("Phishing", "Legitimate")
        assert 0.0 <= data["phishing_probability"] <= 1.0
        assert data["risk_level"] in ("Low Risk", "Medium Risk", "High Risk")
        assert "explanation" in data

    def test_predict_endpoint_missing_url_field(self, client):
        resp = client.post("/api/predict", json={})
        assert resp.status_code == 400

    def test_predict_endpoint_empty_url(self, client):
        resp = client.post("/api/predict", json={"url": ""})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_predict_endpoint_malformed_url(self, client):
        resp = client.post("/api/predict", json={"url": "not a url at all !!"})
        assert resp.status_code == 400

    def test_metrics_endpoint_returns_metrics(self, client):
        resp = client.get("/api/model-performance")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "accuracy" in data
        assert "confusion_matrix" in data

    def test_health_endpoint(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"
