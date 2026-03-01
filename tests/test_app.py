#!/usr/bin/env python3
"""
End-to-End Test Suite for CardioPredict
========================================
Author: Sandeep Kumar
Tests:
    1. Model loading and prediction
    2. Flask routes (200 OK, form submission, JSON API)
    3. Health-check endpoint
    4. Input validation
    5. Edge cases
    6. Database integration
"""

import os
import sys
import json

import pytest
import numpy as np
import joblib

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app as flask_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Create a Flask test client."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


@pytest.fixture
def model():
    """Load trained model."""
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "heart_disease_model.pkl",
    )
    return joblib.load(model_path)


@pytest.fixture
def sample_healthy():
    """Sample input likely to predict NO heart disease."""
    return {
        "age": "35", "sex": "0", "cp": "0", "trestbps": "120",
        "chol": "180", "fbs": "0", "restecg": "0", "thalach": "185",
        "exang": "0", "oldpeak": "0", "slope": "2", "ca": "0", "thal": "2",
    }


@pytest.fixture
def sample_at_risk():
    """Sample input likely to predict heart disease."""
    return {
        "age": "65", "sex": "1", "cp": "3", "trestbps": "160",
        "chol": "300", "fbs": "1", "restecg": "2", "thalach": "100",
        "exang": "1", "oldpeak": "4.0", "slope": "1", "ca": "3", "thal": "1",
    }


# ---------------------------------------------------------------------------
# 1. Model Tests
# ---------------------------------------------------------------------------

class TestModel:
    """Test the trained ML model directly."""

    def test_model_loads(self, model):
        """Model file should load without errors."""
        assert model is not None

    def test_model_has_predict(self, model):
        """Model should have a predict method (Pipeline)."""
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_model_predicts_binary(self, model):
        """Prediction should be 0 or 1."""
        X = np.array([[55, 1, 2, 130, 250, 0, 1, 150, 0, 1.5, 1, 1, 2]])
        pred = model.predict(X)
        assert pred[0] in (0, 1)

    def test_model_probability_range(self, model):
        """Probabilities should be between 0 and 1."""
        X = np.array([[55, 1, 2, 130, 250, 0, 1, 150, 0, 1.5, 1, 1, 2]])
        prob = model.predict_proba(X)
        assert prob.shape == (1, 2)
        assert 0 <= prob[0][0] <= 1
        assert 0 <= prob[0][1] <= 1
        assert abs(prob[0][0] + prob[0][1] - 1.0) < 1e-6

    def test_model_accepts_13_features(self, model):
        """Model expects exactly 13 features."""
        X = np.array([[55, 1, 2, 130, 250, 0, 1, 150, 0, 1.5, 1, 1, 2]])
        assert X.shape[1] == 13
        pred = model.predict(X)
        assert len(pred) == 1

    def test_model_batch_prediction(self, model):
        """Model should handle batch predictions."""
        X = np.array([
            [55, 1, 2, 130, 250, 0, 1, 150, 0, 1.5, 1, 1, 2],
            [35, 0, 0, 120, 180, 0, 0, 185, 0, 0.0, 2, 0, 2],
        ])
        preds = model.predict(X)
        assert len(preds) == 2


# ---------------------------------------------------------------------------
# 2. Flask Route Tests
# ---------------------------------------------------------------------------

class TestRoutes:
    """Test Flask application routes."""

    def test_index_page(self, client):
        """Home page should return 200."""
        response = client.get("/")
        assert response.status_code == 200
        assert b"CardioPredict" in response.data

    def test_about_on_index(self, client):
        """Home page is now the about page and should show author."""
        response = client.get("/")
        assert response.status_code == 200
        assert b"Sandeep Kumar" in response.data

    def test_model_info_page(self, client):
        """Model info page should return 200."""
        response = client.get("/model-info")
        assert response.status_code == 200

    def test_parameters_page(self, client):
        """Parameters guide page should return 200 and list all features."""
        response = client.get("/parameters")
        assert response.status_code == 200
        html = response.data.decode()
        assert "Age" in html
        assert "Serum Cholesterol" in html

    def test_predict_get(self, client):
        """Predict page GET should show the form."""
        response = client.get("/predict")
        assert response.status_code == 200
        html = response.data.decode()
        for feat in ["age", "sex", "cp", "trestbps", "chol", "fbs",
                      "restecg", "thalach", "exang", "oldpeak",
                      "slope", "ca", "thal"]:
            assert f'name="{feat}"' in html, f"Missing input field: {feat}"

    def test_history_page(self, client):
        """History page should return 200."""
        response = client.get("/history")
        assert response.status_code == 200

    def test_health_endpoint(self, client):
        """Health-check should return JSON with healthy status."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_predict_form_submission(self, client, sample_healthy):
        """Form submission should return 200 with result."""
        response = client.post("/predict", data=sample_healthy)
        assert response.status_code == 200
        assert b"Heart Disease" in response.data or b"No Heart Disease" in response.data

    def test_predict_at_risk(self, client, sample_at_risk):
        """High-risk input should return a prediction."""
        response = client.post("/predict", data=sample_at_risk)
        assert response.status_code == 200

    def test_predict_missing_field(self, client):
        """Incomplete form should return 400."""
        response = client.post("/predict", data={"age": "55"})
        assert response.status_code == 400

    def test_predict_json_api(self, client, sample_healthy):
        """JSON API should return prediction JSON."""
        response = client.post(
            "/predict",
            data=json.dumps(sample_healthy),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "prediction" in data
        assert "probability" in data
        assert "label" in data
        assert data["prediction"] in (0, 1)

    def test_predict_json_missing_field(self, client):
        """JSON API with missing field should return error."""
        response = client.post(
            "/predict",
            data=json.dumps({"age": "55"}),
            content_type="application/json",
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# 3. Content Tests
# ---------------------------------------------------------------------------

class TestContent:
    """Test that page content is correct."""

    def test_author_in_footer(self, client):
        """Footer should mention Sandeep Kumar."""
        response = client.get("/")
        assert b"Sandeep Kumar" in response.data

    def test_disclaimer_on_result(self, client, sample_healthy):
        """Result page should include a medical disclaimer."""
        response = client.post("/predict", data=sample_healthy)
        assert b"disclaimer" in response.data.lower() or b"Disclaimer" in response.data

    def test_form_has_all_fields(self, client):
        """Predict page should have all 13 feature inputs."""
        response = client.get("/predict")
        html = response.data.decode()
        for feat in ["age", "sex", "cp", "trestbps", "chol", "fbs",
                      "restecg", "thalach", "exang", "oldpeak",
                      "slope", "ca", "thal"]:
            assert f'name="{feat}"' in html, f"Missing input field: {feat}"


# ---------------------------------------------------------------------------
# 4. Edge Case Tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_extreme_age(self, client, sample_healthy):
        """Very old age should still produce valid prediction."""
        sample_healthy["age"] = "100"
        response = client.post("/predict", data=sample_healthy)
        assert response.status_code == 200

    def test_zero_values(self, client):
        """All-zero input should produce valid prediction (not crash)."""
        data = {f: "0" for f in [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]}
        response = client.post("/predict", data=data)
        assert response.status_code == 200

    def test_404_page(self, client):
        """Non-existent page should return 404."""
        response = client.get("/nonexistent")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# 5. Metadata Tests
# ---------------------------------------------------------------------------

class TestMetadata:
    """Test model metadata file."""

    def test_metadata_exists(self):
        """Model metadata file should exist."""
        meta_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "model_metadata.json",
        )
        assert os.path.exists(meta_path)

    def test_metadata_structure(self):
        """Metadata should have required keys."""
        meta_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "model_metadata.json",
        )
        with open(meta_path) as f:
            meta = json.load(f)

        required_keys = [
            "project", "author", "best_model", "test_metrics",
            "feature_names", "cross_validation",
        ]
        for key in required_keys:
            assert key in meta, f"Missing metadata key: {key}"

        assert meta["author"] == "Sandeep Kumar"
        assert len(meta["feature_names"]) == 13


# ---------------------------------------------------------------------------
# 6. Database Tests
# ---------------------------------------------------------------------------

class TestDatabase:
    """Test SQLite prediction storage."""

    def test_prediction_saved(self, client, sample_healthy):
        """Submitting a prediction should save it to the database."""
        # Submit a prediction
        client.post("/predict", data=sample_healthy)
        # Check history page shows it
        response = client.get("/history")
        assert response.status_code == 200
        assert b"Healthy" in response.data or b"Disease" in response.data

    def test_history_shows_results(self, client, sample_at_risk):
        """History should display saved predictions."""
        client.post("/predict", data=sample_at_risk)
        response = client.get("/history")
        assert response.status_code == 200
        html = response.data.decode()
        assert "65" in html  # the age we submitted
