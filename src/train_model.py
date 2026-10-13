"""
train_model.py
----------------
End-to-end training pipeline. Run directly:

    python -m src.train_model

What it does, in order:
  1. Load and clean the PhiUSIIL dataset (src/data_loader.py).
  2. Build the feature matrix from raw URLs (src/feature_extraction.py).
  3. Split into train/test, grouped by registrable domain, so the same
     domain never appears in both sets (a stronger guard against
     leakage than a plain random split).
  4. MODEL SELECTION: compare Logistic Regression, Random Forest, and
     Gradient Boosting with stratified cross-validation, using F1-score
     on the phishing class as the selection criterion. To keep this
     fast on a normal laptop (this project intentionally avoids GPU/
     cluster dependencies), the comparison runs on a stratified
     subsample of the training set. This is a standard, documented
     speed/thoroughness tradeoff -- it is used ONLY to pick which
     algorithm to commit to, never to produce the final reported
     numbers.
  5. FINAL FIT: retrain the selected algorithm on the FULL training set.
  6. Evaluate ONCE on the untouched test set -> these are the numbers
     reported everywhere in the app and README.
  7. Save the fitted pipeline (joblib), metadata (feature list, chosen
     model, thresholds), and metrics/plots (reports/).

No numbers in this file are hard-coded; every metric written to
reports/metrics.json comes from an actual scikit-learn evaluation run
on this machine.
"""

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import joblib

from src.config import (
    CONFUSION_MATRIX_PATH,
    METADATA_PATH,
    METRICS_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TEST_SIZE,
)
from src.data_loader import get_registrable_domain, load_clean_dataset
from src.evaluation import compute_metrics, plot_confusion_matrix
from src.feature_extraction import FEATURE_NAMES
from src.preprocessing import build_feature_matrix

# Cap on how many training rows are used for the *model comparison*
# step only (see module docstring). The final model is always trained
# on the full training set.
COMPARISON_SAMPLE_SIZE = 50_000


def _candidate_models():
    """Return {name: sklearn Pipeline} for every model being compared.
    Logistic Regression is wrapped with a scaler because it is
    scale-sensitive; the tree models are not, so they skip scaling."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"
            )),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=12, random_state=RANDOM_SEED,
                n_jobs=-1, class_weight="balanced",
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("clf", GradientBoostingClassifier(
                n_estimators=100, max_depth=3, random_state=RANDOM_SEED,
            )),
        ]),
    }


def compare_models(X_train, y_train) -> dict:
    """3-fold stratified CV F1-score for each candidate, on a stratified
    subsample of the training set for speed. Returns a dict suitable for
    saving to JSON, used both to print a comparison table and as
    documented evidence for the final model choice."""
    rng = np.random.RandomState(RANDOM_SEED)
    if len(X_train) > COMPARISON_SAMPLE_SIZE:
        # Stratified subsample: sample proportionally within each class
        # so the class balance of the subsample matches the full set.
        idx_pos = y_train[y_train == 1].index
        idx_neg = y_train[y_train == 0].index
        frac = COMPARISON_SAMPLE_SIZE / len(X_train)
        n_pos = int(len(idx_pos) * frac)
        n_neg = int(len(idx_neg) * frac)
        sample_idx = np.concatenate([
            rng.choice(idx_pos, size=n_pos, replace=False),
            rng.choice(idx_neg, size=n_neg, replace=False),
        ])
        X_cmp, y_cmp = X_train.loc[sample_idx], y_train.loc[sample_idx]
    else:
        X_cmp, y_cmp = X_train, y_train

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    results = {}
    for name, pipeline in _candidate_models().items():
        t0 = time.time()
        scores = cross_val_score(pipeline, X_cmp, y_cmp, cv=cv, scoring="f1", n_jobs=1)
        elapsed = time.time() - t0
        results[name] = {
            "cv_f1_mean": float(scores.mean()),
            "cv_f1_std": float(scores.std()),
            "cv_scores": [float(s) for s in scores],
            "comparison_sample_size": int(len(X_cmp)),
            "seconds": round(elapsed, 1),
        }
        print(f"[train_model] {name}: CV F1 = {scores.mean():.4f} "
              f"(+/- {scores.std():.4f}) on {len(X_cmp)} rows, "
              f"{elapsed:.1f}s")
    return results


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("[train_model] Loading and cleaning dataset...")
    data = load_clean_dataset()

    print("[train_model] Extracting features from raw URLs "
          f"(this recomputes {len(FEATURE_NAMES)} lexical features per "
          "URL; it can take a few minutes for the full dataset)...")
    t0 = time.time()
    X_full = build_feature_matrix(data["url"])
    y_full = data.loc[X_full.index, "label"]
    print(f"[train_model] Feature extraction done in {time.time() - t0:.1f}s "
          f"-> {X_full.shape[0]} rows x {X_full.shape[1]} features.")

    # Group-aware split: same registrable domain never in both sets.
    groups = data.loc[X_full.index, "url"].apply(get_registrable_domain)
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(X_full, y_full, groups=groups))

    X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
    y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]
    print(f"[train_model] Train: {len(X_train)} rows | Test: {len(X_test)} rows "
          f"(split grouped by domain to avoid leakage)")

    # ---------------- Model selection ----------------
    print("\n[train_model] Comparing candidate models with cross-validation...")
    comparison = compare_models(X_train, y_train)
    with open(MODEL_COMPARISON_PATH, "w") as f:
        json.dump(comparison, f, indent=2)

    best_name = max(comparison, key=lambda k: comparison[k]["cv_f1_mean"])
    print(f"\n[train_model] Selected model: {best_name} "
          f"(highest CV F1 = {comparison[best_name]['cv_f1_mean']:.4f})")

    # ---------------- Final fit on FULL training set ----------------
    print(f"[train_model] Fitting {best_name} on the full training set...")
    best_pipeline = _candidate_models()[best_name]
    t0 = time.time()
    best_pipeline.fit(X_train, y_train)
    print(f"[train_model] Final fit done in {time.time() - t0:.1f}s")

    # ---------------- Evaluate ONCE on the held-out test set ----------------
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics["model_name"] = best_name
    metrics["train_size"] = int(len(X_train))
    metrics["test_size"] = int(len(X_test))
    metrics["feature_count"] = int(X_full.shape[1])

    print("\n[train_model] === Held-out test set performance ===")
    for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        print(f"  {k:>10}: {metrics[k]:.4f}")
    print(f"  confusion_matrix [[TN,FP],[FN,TP]]: {metrics['confusion_matrix']}")

    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        CONFUSION_MATRIX_PATH,
        title=f"Confusion Matrix - {best_name} (test set)",
    )

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # ---------------- Save model + metadata ----------------
    # compress=3: meaningfully shrinks tree-ensemble models (a deep/wide
    # Random Forest or Gradient Boosting can otherwise be tens of MB) at
    # negligible extra save/load time, with zero effect on predictions.
    joblib.dump(best_pipeline, MODEL_PATH, compress=3)

    metadata = {
        "model_name": best_name,
        "feature_columns": FEATURE_NAMES,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "sklearn_pipeline_steps": [s[0] for s in best_pipeline.steps],
        "trained_on_rows": int(len(X_train)),
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[train_model] Saved model to {MODEL_PATH}")
    print(f"[train_model] Saved metadata to {METADATA_PATH}")
    print(f"[train_model] Saved metrics to {METRICS_PATH}")
    print(f"[train_model] Saved confusion matrix plot to {CONFUSION_MATRIX_PATH}")

    # Also persist a small held-out sample of test features/labels so
    # explainability.py can build a real SHAP global summary plot
    # without re-running feature extraction on the whole dataset.
    sample_n = min(500, len(X_test))
    sample_idx = X_test.sample(n=sample_n, random_state=RANDOM_SEED).index
    X_test.loc[sample_idx].to_csv(
        os.path.join(REPORTS_DIR, "shap_background_sample.csv"), index=False
    )
    print(f"[train_model] Saved a {sample_n}-row test-set sample for SHAP "
          f"global explanations to reports/shap_background_sample.csv")


if __name__ == "__main__":
    main()
