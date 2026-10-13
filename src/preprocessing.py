"""
preprocessing.py
------------------
Turns a list/Series of raw URLs into a numeric feature matrix (pandas
DataFrame) using feature_extraction.extract_features, and provides a
scaler helper for the models (Logistic Regression) that need scaled
input. Tree-based models do not need scaling, but applying the same
build_feature_matrix() to both keeps the code identical for every model.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.feature_extraction import FEATURE_NAMES, extract_features, validate_url


def build_feature_matrix(urls) -> pd.DataFrame:
    """
    Convert an iterable of raw URL strings into a DataFrame with one row
    per URL and columns in the exact order of FEATURE_NAMES.

    URLs that fail validation are skipped with a warning printed (this
    matters during training, where a handful of malformed rows in a
    235k-row public dataset is expected); the returned DataFrame's index
    tells the caller which original rows survived, via `.index`.
    """
    rows = []
    kept_index = []
    skipped = 0
    for i, raw_url in enumerate(urls):
        try:
            url = validate_url(str(raw_url))
            rows.append(extract_features(url))
            kept_index.append(i)
        except Exception:
            skipped += 1
            continue

    if skipped:
        print(f"[preprocessing] Skipped {skipped} URL(s) that could not "
              f"be parsed into features.")

    matrix = pd.DataFrame(rows, columns=FEATURE_NAMES)
    matrix.index = kept_index
    return matrix


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_with_scaler(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    scaled = scaler.transform(X)
    return pd.DataFrame(scaled, columns=X.columns, index=X.index)
