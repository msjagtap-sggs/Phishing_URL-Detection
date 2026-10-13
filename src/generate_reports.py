"""
generate_reports.py
---------------------
Generates the global feature-importance plot (reports/shap_summary.png)
shown on the Model Performance page, using the background sample of
real test-set rows saved by train_model.py.

Run after training (or after installing shap for the first time, to
upgrade the plot from the feature_importances_ fallback to true SHAP):

    python -m src.generate_reports
"""

import os

import joblib
import pandas as pd

from src.config import MODEL_PATH, METADATA_PATH, REPORTS_DIR
from src.explainability import SHAP_AVAILABLE, global_feature_importance
from src.feature_extraction import FEATURE_NAMES

BACKGROUND_SAMPLE_PATH = os.path.join(REPORTS_DIR, "shap_background_sample.csv")


def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit("No trained model found. Run 'python -m src.train_model' first.")
    if not os.path.exists(BACKGROUND_SAMPLE_PATH):
        raise SystemExit(
            f"Background sample not found at {BACKGROUND_SAMPLE_PATH}. "
            "Run 'python -m src.train_model' first (it writes this file)."
        )

    pipeline = joblib.load(MODEL_PATH)
    background = pd.read_csv(BACKGROUND_SAMPLE_PATH)[FEATURE_NAMES]

    print(f"[generate_reports] SHAP available: {SHAP_AVAILABLE}")
    print(f"[generate_reports] Computing global feature importance on "
          f"{len(background)} background rows...")

    result = global_feature_importance(pipeline, model_name="", background_df=background)
    print(f"[generate_reports] Method used: {result.attrs.get('method')}")
    print(result.head(10).to_string(index=False))
    print(f"[generate_reports] Saved plot to reports/shap_summary.png")


if __name__ == "__main__":
    main()
