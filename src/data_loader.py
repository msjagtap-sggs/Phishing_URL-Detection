"""
data_loader.py
---------------
Loads the PhiUSIIL Phishing URL Dataset and turns it into exactly two
columns we trust: the raw URL string and a clean 0/1 label, where
1 = phishing and 0 = legitimate.

WHY we throw away the other ~50 columns in the CSV
----------------------------------------------------
The original PhiUSIIL CSV includes many columns (e.g. URLSimilarityIndex,
URLTitleMatchScore, LineOfCode, NoOfImage, HasFavicon...) that were
computed by the dataset authors from the *webpage's source code*, not
from the URL text alone. A real prediction pipeline that only receives a
URL string from a user can never compute those columns, so training on
them would be a severe case of data leakage: the model would learn to
rely on information it will never have at prediction time, and its
real-world accuracy would collapse compared to its test-set accuracy.

So this loader keeps only:
    - URL      (raw text, used to recompute our own features)
    - label    (recoded so 1 = phishing, matching the rest of the project)

and src/feature_extraction.py recomputes every feature from scratch,
identically for training and for prediction.
"""

import os

import pandas as pd

from src.config import RAW_DATASET_PATH


class DatasetNotFoundError(FileNotFoundError):
    pass


def load_raw_dataset(path: str = RAW_DATASET_PATH) -> pd.DataFrame:
    """Load the CSV and fail with a clear, actionable message if it's
    missing, rather than a raw pandas traceback."""
    if not os.path.exists(path):
        raise DatasetNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download the PhiUSIIL Phishing URL Dataset from the UCI "
            "Machine Learning Repository:\n"
            "  https://archive.ics.uci.edu/dataset/967\n"
            "and save it as 'data/dataset.csv' (or use "
            "'pip install ucimlrepo' and fetch id=967)."
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce the raw PhiUSIIL dataframe to a clean (url, label) table.

    Steps:
        1. Keep only the URL and label columns.
        2. In the original dataset, label == 1 means LEGITIMATE and
           label == 0 means PHISHING. We flip this so label == 1 means
           PHISHING everywhere else in this project, because precision/
           recall are most naturally discussed with the harmful class
           (phishing) as the positive class.
        3. Drop rows with missing/empty URLs.
        4. Drop duplicate URLs (exact string duplicates).
        5. Reset the index.
    """
    required = {"URL", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing expected column(s): {missing}. "
            "Expected the PhiUSIIL dataset format with 'URL' and 'label' "
            "columns."
        )

    clean = df[["URL", "label"]].copy()
    clean.columns = ["url", "label_legit_is_1"]

    clean["url"] = clean["url"].astype(str).str.strip()
    clean = clean[clean["url"].ne("") & clean["url"].ne("nan")]

    before = len(clean)
    clean = clean.drop_duplicates(subset="url")
    removed = before - len(clean)

    # Flip label so 1 = phishing (matches the rest of this project).
    clean["label"] = 1 - clean["label_legit_is_1"].astype(int)
    clean = clean.drop(columns=["label_legit_is_1"])

    clean = clean.dropna(subset=["url", "label"])
    clean = clean.reset_index(drop=True)

    print(f"[data_loader] Loaded {before} URLs, removed {removed} exact "
          f"duplicate URLs, {len(clean)} remain.")
    print(f"[data_loader] Class balance -> phishing (1): "
          f"{int(clean['label'].sum())}, legitimate (0): "
          f"{int((clean['label'] == 0).sum())}")

    return clean


def get_registrable_domain(url: str) -> str:
    """Cheap 'group key' used only for the train/test split, so the same
    domain can never appear in both the training set and the test set
    (a second, independent leakage guard on top of exact-URL dedup)."""
    from urllib.parse import urlparse
    try:
        host = urlparse(url if "://" in url else "http://" + url).hostname or ""
    except ValueError:
        host = ""
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def load_clean_dataset(path: str = RAW_DATASET_PATH) -> pd.DataFrame:
    """Convenience wrapper: load + clean in one call."""
    raw = load_raw_dataset(path)
    return clean_dataset(raw)
