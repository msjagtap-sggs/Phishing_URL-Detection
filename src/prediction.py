"""
prediction.py
--------------
The single entry point the app (or anything else) should call to get a
prediction for a URL typed by a user. Wraps model loading, feature
extraction, the model itself, and risk categorisation, with clear
error handling for every failure mode a student demo needs to survive:
missing model file, malformed URL, feature/column mismatch.
"""

import json
import os

import joblib
import pandas as pd

from src.config import METADATA_PATH, MODEL_PATH, RISK_THRESHOLDS
from src.feature_extraction import (
    FEATURE_NAMES,
    InvalidURLError,
    extract_features,
    validate_url,
)


class ModelNotFoundError(FileNotFoundError):
    pass


def risk_category(probability: float) -> str:
    """Map a phishing probability to a Low/Medium/High label using the
    single documented threshold table in src/config.py."""
    if probability < RISK_THRESHOLDS["low_max"]:
        return "Low Risk"
    if probability < RISK_THRESHOLDS["medium_max"]:
        return "Medium Risk"
    return "High Risk"


class PhishingDetector:
    """Loads the trained pipeline + metadata once, then serves
    predictions. Designed to be instantiated once per app process
    (e.g. cached as a Streamlit/Flask singleton)."""

    def __init__(self, model_path: str = MODEL_PATH, metadata_path: str = METADATA_PATH):
        if not os.path.exists(model_path):
            raise ModelNotFoundError(
                f"No trained model found at '{model_path}'.\n"
                "Train the model first by running:\n"
                "    python -m src.train_model\n"
                "from the project root. This creates the model, scaler "
                "(if needed) and metadata files under models/."
            )
        if not os.path.exists(metadata_path):
            raise ModelNotFoundError(
                f"Model file exists but metadata file '{metadata_path}' is "
                "missing. Please re-run 'python -m src.train_model' so "
                "both files are regenerated together."
            )

        self.pipeline = joblib.load(model_path)
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        saved_columns = self.metadata.get("feature_columns", [])
        if saved_columns != FEATURE_NAMES:
            raise ValueError(
                "The feature columns used to train the saved model do not "
                "match the current feature_extraction.FEATURE_NAMES. "
                "This usually means feature_extraction.py was changed "
                "after training. Re-run 'python -m src.train_model'."
            )

    def predict(self, raw_url: str) -> dict:
        """
        Full pipeline for one URL: validate -> extract features ->
        predict -> risk category. Returns a dict the UI can render
        directly, or raises InvalidURLError with a user-facing message
        if the input isn't a usable URL.
        """
        normalised_url = validate_url(raw_url)
        scheme_was_assumed = not raw_url.strip().lower().startswith(
            ("http://", "https://")
        )

        feature_dict = extract_features(normalised_url)
        X = pd.DataFrame([feature_dict], columns=FEATURE_NAMES)

        proba_phishing = float(self.pipeline.predict_proba(X)[0, 1])
        predicted_label = int(proba_phishing >= 0.5)

        return {
            "input_url": raw_url.strip(),
            "normalised_url": normalised_url,
            "scheme_was_assumed": scheme_was_assumed,
            "prediction": "Phishing" if predicted_label == 1 else "Legitimate",
            "predicted_label": predicted_label,
            "phishing_probability": proba_phishing,
            "legitimate_probability": 1 - proba_phishing,
            "risk_level": risk_category(proba_phishing),
            "features": feature_dict,
            "feature_vector": X,
            "model_name": self.metadata.get("model_name", "unknown"),
        }


# Module-level cache so Streamlit/Flask can import `get_detector()` and
# only pay the joblib-load cost once per process.
_detector_instance = None


def get_detector() -> PhishingDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = PhishingDetector()
    return _detector_instance
