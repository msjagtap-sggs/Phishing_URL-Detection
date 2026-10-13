"""
app.py
------
Flask application: serves the HTML/CSS/JS frontend (frontend/) and a
small JSON API that the frontend's JavaScript calls with fetch().

This file contains NO machine-learning logic itself -- it only calls
into src/prediction.py and src/explainability.py, which do the real
work. That separation is deliberate: the ML pipeline can be tested and
reused (e.g. from a notebook or a CLI script) independently of Flask.

Run with:
    python app.py
then open http://127.0.0.1:5000 in a browser.
"""

import json
import os

from flask import Flask, jsonify, request, send_from_directory

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

from src.config import (
    CONFUSION_MATRIX_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    REPORTS_DIR,
    RISK_THRESHOLDS,
    SHAP_SUMMARY_PATH,
)
from src.explainability import build_plain_language_summary, explain_prediction
from src.feature_extraction import InvalidURLError
from src.prediction import ModelNotFoundError, get_detector
from src.threat_intel import check_virustotal

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

app = Flask(__name__, static_folder=None)


# ---------------------------------------------------------------------------
# Frontend routes (serve the static HTML/CSS/JS)
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def frontend_assets(filename):
    # Serves css/style.css, js/app.js, etc.
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/reports/<path:filename>")
def report_assets(filename):
    # Serves confusion_matrix.png / shap_summary.png to the frontend.
    return send_from_directory(REPORTS_DIR, filename)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    url = payload.get("url", "")

    if not isinstance(url, str) or not url.strip():
        return jsonify({"error": "Please enter a URL to analyse."}), 400

    try:
        detector = get_detector()
    except ModelNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    try:
        result = detector.predict(url)
    except InvalidURLError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - unexpected failure
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    try:
        explanation = explain_prediction(
            detector.pipeline, result["model_name"], result["feature_vector"]
        )
        plain_language = build_plain_language_summary(explanation, top_k=5)
    except Exception as exc:  # pragma: no cover - explanation is best-effort
        explanation = {"method": "unavailable", "base_value": None, "contributions": []}
        plain_language = [f"Explanation could not be generated: {exc}"]

    # Threat-intel check (VirusTotal) is a SEPARATE, optional signal --
    # never lets a slow/unreachable/unconfigured API break the core
    # prediction feature. See src/threat_intel.py for why this is a
    # useful complement (reactive) rather than a replacement for the
    # ML model (proactive).
    try:
        threat_intel = check_virustotal(result["normalised_url"])
    except Exception as exc:  # pragma: no cover - must never break /api/predict
        threat_intel = {
            "configured": False, "checked": False, "found": False,
            "verdict": "error", "malicious_count": 0, "suspicious_count": 0,
            "harmless_count": 0, "total_engines": 0, "permalink": None,
            "message": f"Threat-intel check failed unexpectedly: {exc}",
        }

    response = {
        "input_url": result["input_url"],
        "normalised_url": result["normalised_url"],
        "scheme_was_assumed": result["scheme_was_assumed"],
        "prediction": result["prediction"],
        "phishing_probability": result["phishing_probability"],
        "legitimate_probability": result["legitimate_probability"],
        "risk_level": result["risk_level"],
        "model_name": result["model_name"],
        "features": result["features"],
        "explanation": {
            "method": explanation["method"],
            "contributions": explanation["contributions"][:10],
            "plain_language": plain_language,
        },
        "threat_intel": threat_intel,
        "risk_thresholds": RISK_THRESHOLDS,
    }
    return jsonify(response)


@app.route("/api/model-performance")
def model_performance():
    if not os.path.exists(METRICS_PATH):
        return jsonify({
            "error": "No metrics found. Train the model first with "
                     "'python -m src.train_model'."
        }), 404

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    comparison = {}
    if os.path.exists(MODEL_COMPARISON_PATH):
        with open(MODEL_COMPARISON_PATH) as f:
            comparison = json.load(f)

    metrics["model_comparison"] = comparison
    metrics["has_confusion_matrix_plot"] = os.path.exists(CONFUSION_MATRIX_PATH)
    metrics["has_shap_summary_plot"] = os.path.exists(SHAP_SUMMARY_PATH)
    return jsonify(metrics)


if __name__ == "__main__":
    # Fail fast with a friendly message if the model hasn't been trained.
    try:
        get_detector()
        print("[app] Model loaded successfully. Starting server...")
    except ModelNotFoundError as exc:
        print("=" * 70)
        print("WARNING: no trained model found.")
        print(str(exc))
        print("The server will still start, but predictions will fail "
              "until you run: python -m src.train_model")
        print("=" * 70)

    app.run(debug=True, host="127.0.0.1", port=5000)
