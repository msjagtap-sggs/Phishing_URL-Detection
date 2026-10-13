"""
test_prediction.py
--------------------
Tests for src/prediction.py: the prediction pipeline, risk
categorisation, and graceful handling of bad input / missing model
files. These tests require a trained model to exist at models/
(run `python -m src.train_model` first) EXCEPT for the
missing-model-file test, which deliberately points at a path that
does not exist.
"""

import os

import pytest

from src.feature_extraction import InvalidURLError
from src.prediction import ModelNotFoundError, PhishingDetector, risk_category


MODEL_EXISTS = os.path.exists("models/phishing_model.joblib")


def test_risk_category_boundaries():
    assert risk_category(0.0) == "Low Risk"
    assert risk_category(0.29) == "Low Risk"
    assert risk_category(0.30) == "Medium Risk"
    assert risk_category(0.69) == "Medium Risk"
    assert risk_category(0.70) == "High Risk"
    assert risk_category(1.0) == "High Risk"


def test_missing_model_file_raises_clear_error():
    with pytest.raises(ModelNotFoundError):
        PhishingDetector(
            model_path="models/does_not_exist.joblib",
            metadata_path="models/does_not_exist.json",
        )


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not trained yet; run "
                     "'python -m src.train_model' first.")
class TestTrainedModelPredictions:

    @pytest.fixture(scope="class")
    def detector(self):
        return PhishingDetector()

    def test_predict_returns_expected_keys(self, detector):
        result = detector.predict("https://www.google.com")
        expected_keys = {
            "input_url", "normalised_url", "scheme_was_assumed",
            "prediction", "predicted_label", "phishing_probability",
            "legitimate_probability", "risk_level", "features",
            "feature_vector", "model_name",
        }
        assert expected_keys.issubset(result.keys())

    def test_probabilities_sum_to_one(self, detector):
        result = detector.predict("https://example.com")
        total = result["phishing_probability"] + result["legitimate_probability"]
        assert abs(total - 1.0) < 1e-6

    def test_prediction_label_matches_probability(self, detector):
        result = detector.predict("https://example.com")
        if result["phishing_probability"] >= 0.5:
            assert result["prediction"] == "Phishing"
            assert result["predicted_label"] == 1
        else:
            assert result["prediction"] == "Legitimate"
            assert result["predicted_label"] == 0

    def test_risk_level_consistent_with_probability(self, detector):
        result = detector.predict("https://example.com")
        assert result["risk_level"] == risk_category(result["phishing_probability"])

    def test_invalid_url_raises_invalid_url_error(self, detector):
        with pytest.raises(InvalidURLError):
            detector.predict("")

    def test_scheme_assumed_flag_for_bare_domain(self, detector):
        result = detector.predict("example.com")
        assert result["scheme_was_assumed"] is True

    def test_scheme_not_assumed_when_provided(self, detector):
        result = detector.predict("https://example.com")
        assert result["scheme_was_assumed"] is False

    def test_ip_based_url_is_flagged_high_risk_feature(self, detector):
        result = detector.predict("http://192.168.0.1/wp-login.php")
        assert result["features"]["has_ip_address"] == 1
