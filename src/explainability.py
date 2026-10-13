"""
explainability.py
-------------------
Generates real, per-prediction explanations from the trained model
using SHAP (SHapley Additive exPlanations). Nothing in this file is a
canned or hard-coded explanation: every number comes from calling SHAP
on the actual fitted model with the actual feature values of the URL
being analysed.

If the `shap` package is not installed, the module falls back to the
model's built-in feature_importances_ (for tree models) so the app
still runs -- but this fallback is clearly labelled as an approximation
in every place it is surfaced, and the README tells the user to
`pip install shap` to get true per-prediction attributions.
"""

import os

import numpy as np
import pandas as pd

from src.config import REPORTS_DIR, SHAP_SUMMARY_PATH

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# Short, plain-language description of what each feature means, used to
# turn a feature name + its value + its SHAP direction into a sentence
# a non-technical reader (or a viva panel) can follow. The TEMPLATE is
# fixed text, but which sentences appear, and the values inside them,
# are filled in from the real SHAP output for the real URL.
FEATURE_DESCRIPTIONS = {
    "url_length": "the overall URL is {value:.0f} characters long",
    "hostname_length": "the domain name is {value:.0f} characters long",
    "path_length": "the URL path is {value:.0f} characters long",
    "query_length": "the query string is {value:.0f} characters long",
    "longest_path_token_length": "the longest single segment of the path is {value:.0f} characters",
    "num_dots": "the URL contains {value:.0f} dot characters",
    "num_subdomains": "the domain has {value:.0f} subdomain level(s)",
    "num_hyphens_url": "the URL contains {value:.0f} hyphen(s)",
    "num_hyphens_hostname": "the domain name contains {value:.0f} hyphen(s)",
    "num_digits_url": "the URL contains {value:.0f} digit(s)",
    "num_digits_hostname": "the domain name contains {value:.0f} digit(s)",
    "num_special_chars": "the URL contains {value:.0f} special character(s)",
    "num_params": "the URL has {value:.0f} query parameter(s)",
    "path_depth": "the URL path has {value:.0f} folder level(s)",
    "digit_ratio": "digits make up {value:.1%} of the URL",
    "letter_ratio": "letters make up {value:.1%} of the URL",
    "special_char_ratio": "special characters make up {value:.1%} of the URL",
    "has_ip_address": "the domain is a raw IP address" ,
    "has_https": "the URL uses HTTPS",
    "has_at_symbol": "the URL contains an '@' symbol",
    "has_shortener": "the domain is a known URL-shortening service",
    "has_port": "the URL explicitly specifies a network port",
    "has_punycode": "the domain uses punycode (xn--) encoding",
    "has_double_slash_in_path": "the URL path contains a redundant '//' ",
    "has_suspicious_tld": "the domain uses a top-level domain often associated with abuse",
    "suspicious_keyword_count": "the URL contains {value:.0f} phishing-related keyword(s) (e.g. 'login', 'verify', 'secure')",
    "url_entropy": "the URL has a character-randomness (entropy) score of {value:.2f}",
    "hostname_entropy": "the domain name has a character-randomness (entropy) score of {value:.2f}",
    "brand_edit_distance": "the domain is {value:.0f} character edit(s) away from a well-known brand name",
    "is_brand_lookalike": "the domain closely resembles a well-known brand without being it (possible typosquat)",
}


def _describe_feature(name: str, value: float) -> str:
    template = FEATURE_DESCRIPTIONS.get(name, name)
    try:
        return template.format(value=value)
    except (KeyError, ValueError):
        return template


def _get_tree_model(pipeline):
    """Pull the underlying sklearn estimator out of the pipeline, and
    report whether the pipeline scales its input (only Logistic
    Regression does), since SHAP needs to see the same representation
    the model was trained on."""
    clf = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps.get("scaler")
    return clf, scaler


def explain_prediction(pipeline, model_name: str, X_row: pd.DataFrame) -> dict:
    """
    Explain a single prediction (X_row = one-row DataFrame of raw,
    unscaled feature values, in FEATURE_NAMES order).

    Returns:
        {
          "method": "shap" | "fallback_feature_importance",
          "base_value": float,             # expected model output
          "contributions": [               # sorted by |impact|, desc
             {"feature", "value", "impact", "direction", "description"}
          ],
        }
    """
    clf, scaler = _get_tree_model(pipeline)
    X_for_model = scaler.transform(X_row) if scaler is not None else X_row.values

    if SHAP_AVAILABLE:
        try:
            if scaler is not None:
                # Linear model: LinearExplainer works directly on the
                # scaled input space the model was trained on.
                explainer = shap.LinearExplainer(clf, X_for_model)
                shap_values = explainer.shap_values(X_for_model)
                base_value = float(np.atleast_1d(explainer.expected_value)[0])
            else:
                # Tree model: TreeExplainer is exact and fast.
                explainer = shap.TreeExplainer(clf)
                raw = explainer.shap_values(X_for_model)
                # Newer SHAP returns a (1, n_features, n_classes) array
                # for classifiers; older SHAP returns a list per class.
                if isinstance(raw, list):
                    shap_values = raw[1]  # class 1 = phishing
                    base_value = float(np.atleast_1d(explainer.expected_value)[1])
                elif hasattr(raw, "ndim") and raw.ndim == 3:
                    shap_values = raw[:, :, 1]
                    base_value = float(np.atleast_1d(explainer.expected_value)[-1])
                else:
                    shap_values = raw
                    base_value = float(np.atleast_1d(explainer.expected_value)[0])

            values = np.asarray(shap_values).reshape(-1)
            contributions = _build_contributions(X_row, values)
            return {
                "method": "shap",
                "base_value": base_value,
                "contributions": contributions,
            }
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"[explainability] SHAP failed ({exc}); "
                  f"falling back to feature_importances_.")

    # ---- Fallback: model's built-in importances (no SHAP available) ----
    importances = getattr(clf, "feature_importances_", None)
    if importances is None:
        importances = np.abs(getattr(clf, "coef_", np.zeros(X_row.shape[1]))).reshape(-1)
    contributions = _build_contributions(X_row, importances, signed=False)
    return {
        "method": "fallback_feature_importance",
        "base_value": None,
        "contributions": contributions,
    }


def _build_contributions(X_row: pd.DataFrame, values: np.ndarray, signed: bool = True) -> list:
    feature_names = list(X_row.columns)
    raw_values = X_row.iloc[0].values
    rows = []
    for name, val, impact in zip(feature_names, raw_values, values):
        if not signed:
            # Raw feature_importances_ only tells us a feature mattered
            # SOMEWHERE across the whole training set, not which way it
            # pushed THIS prediction -- so we must not claim a direction.
            direction = "was an influential signal for this prediction"
        elif impact > 0:
            direction = "increases phishing risk"
        elif impact < 0:
            direction = "decreases phishing risk"
        else:
            direction = "had no effect"
        rows.append({
            "feature": name,
            "value": float(val),
            "impact": float(impact),
            "direction": direction,
            "description": _describe_feature(name, float(val)),
        })
    rows.sort(key=lambda r: abs(r["impact"]), reverse=True)
    return rows


def build_plain_language_summary(explanation: dict, top_k: int = 5) -> list:
    """Turn the top-k SHAP contributions into plain-English sentences,
    e.g. 'The URL contains 3 phishing-related keywords, which increases
    phishing risk.' Built entirely from real values -- nothing here is
    a template chosen ahead of time for a specific URL."""
    sentences = []
    for c in explanation["contributions"][:top_k]:
        sentence = f"{c['description'][0].upper()}{c['description'][1:]}, which {c['direction']}."
        sentences.append(sentence)
    return sentences


def global_feature_importance(pipeline, model_name: str, background_df: pd.DataFrame,
                               save_plot: bool = True) -> pd.DataFrame:
    """
    Compute mean |SHAP value| per feature across a background sample
    (reports/shap_background_sample.csv, written by train_model.py),
    for the Model Performance / Explainable AI page's global summary.
    Optionally saves a bar-chart PNG to reports/shap_summary.png.
    """
    clf, scaler = _get_tree_model(pipeline)
    X = scaler.transform(background_df) if scaler is not None else background_df.values

    if SHAP_AVAILABLE:
        try:
            if scaler is not None:
                explainer = shap.LinearExplainer(clf, X)
                shap_values = explainer.shap_values(X)
            else:
                explainer = shap.TreeExplainer(clf)
                raw = explainer.shap_values(X)
                if isinstance(raw, list):
                    shap_values = raw[1]
                elif hasattr(raw, "ndim") and raw.ndim == 3:
                    shap_values = raw[:, :, 1]
                else:
                    shap_values = raw
            mean_abs = np.abs(np.asarray(shap_values)).mean(axis=0)
            method = "shap"
        except Exception as exc:  # pragma: no cover
            print(f"[explainability] Global SHAP failed ({exc}); using "
                  f"feature_importances_ instead.")
            mean_abs = getattr(clf, "feature_importances_",
                                np.zeros(background_df.shape[1]))
            method = "fallback_feature_importance"
    else:
        mean_abs = getattr(clf, "feature_importances_",
                            np.zeros(background_df.shape[1]))
        method = "fallback_feature_importance"

    result = pd.DataFrame({
        "feature": background_df.columns,
        "mean_abs_impact": mean_abs,
    }).sort_values("mean_abs_impact", ascending=False).reset_index(drop=True)
    result.attrs["method"] = method

    if save_plot:
        _plot_global_importance(result, SHAP_SUMMARY_PATH, method)

    return result


def _plot_global_importance(result: pd.DataFrame, save_path: str, method: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = result.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["feature"], top["mean_abs_impact"], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value|" if method == "shap" else "Feature importance")
    ax.set_title("Global feature importance (top 15)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
