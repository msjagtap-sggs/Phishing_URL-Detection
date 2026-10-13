"""
config.py
---------
Single source of truth for paths, random seeds, and the risk-threshold
rules used across the whole project. Keeping these in one place means
training and prediction can never quietly drift apart from each other.
"""

import os

# ---------------------------------------------------------------------------
# Paths (all relative to the project root, so the project is portable)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "phishing_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
METRICS_PATH = os.path.join(REPORTS_DIR, "metrics.json")
MODEL_COMPARISON_PATH = os.path.join(REPORTS_DIR, "model_comparison.json")
CONFUSION_MATRIX_PATH = os.path.join(REPORTS_DIR, "confusion_matrix.png")
SHAP_SUMMARY_PATH = os.path.join(REPORTS_DIR, "shap_summary.png")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# Risk thresholds
# Applied to the model's predicted probability of the PHISHING class (1).
# Documented here so the rule is visible in exactly one place, and shown
# to the user in the "About" section of the app.
# ---------------------------------------------------------------------------
RISK_THRESHOLDS = {
    "low_max": 0.30,     # probability < 0.30  -> Low Risk
    "medium_max": 0.70,  # 0.30 <= probability < 0.70 -> Medium Risk
                          # probability >= 0.70 -> High Risk
}

# ---------------------------------------------------------------------------
# Suspicious keyword list used by the keyword-count feature.
# These are common words phishing pages use to impersonate legitimate
# login / payment / security flows.
# ---------------------------------------------------------------------------
SUSPICIOUS_KEYWORDS = [
    "login", "signin", "sign-in", "verify", "verification", "secure",
    "account", "update", "confirm", "bank", "password", "credential",
    "wallet", "pay", "billing", "invoice", "suspend", "suspended",
    "unlock", "recover", "reset", "support", "alert", "webscr",
    "ebayisapi", "authenticate", "limited", "security",
]

# Known URL-shortening services (indicative, not exhaustive).
SHORTENER_DOMAINS = [
    "bit.ly", "goo.gl", "shorte.st", "go2l.ink", "x.co", "ow.ly",
    "tinyurl.com", "tr.im", "is.gd", "cli.gs", "yfrog.com", "migre.me",
    "ff.im", "tiny.cc", "url4.eu", "twit.ac", "su.pr", "twurl.nl",
    "snipurl.com", "short.to", "budurl.com", "ping.fm", "post.ly",
    "just.as", "bkite.com", "snipr.com", "fic.kr", "loopt.us",
    "doiop.com", "short.ie", "kl.am", "wp.me", "rubyurl.com",
    "om.ly", "to.ly", "t.co", "lnkd.in", "db.tt", "qr.ae", "adf.ly",
    "bitly.com", "cur.lv", "tinyarrows.com", "v.gd", "buff.ly",
]

# TLDs that are disproportionately associated with abuse in public
# phishing reports. Used only as one weak signal among many features,
# never as a standalone rule.
SUSPICIOUS_TLDS = [
    "zip", "review", "country", "kim", "cricket", "science", "work",
    "party", "gq", "link", "xyz", "top", "club", "click", "tk", "ml",
    "ga", "cf", "men", "loan", "download", "racing", "win",
]

# Well-known brand names that phishing pages frequently impersonate via
# a lookalike domain (e.g. "paypa1.com", "arnazon-support.com"). Used by
# the brand_edit_distance / is_brand_lookalike features in
# feature_extraction.py. This list is inevitably incomplete -- expanding
# it (or replacing it with a larger public brand-name corpus) is a
# natural improvement for this project.
KNOWN_BRANDS = [
    "google", "youtube", "facebook", "instagram", "whatsapp", "amazon",
    "apple", "microsoft", "netflix", "paypal", "ebay", "linkedin",
    "twitter", "outlook", "office365", "dropbox", "adobe", "yahoo",
    "gmail", "icloud", "chase", "wellsfargo", "bankofamerica", "citibank",
    "hsbc", "coinbase", "binance", "fedex", "dhl", "usps", "irs",
    "spotify", "steam", "discord", "telegram", "tiktok", "snapchat",
    "walmart", "target", "costco", "americanexpress", "visa",
    "mastercard", "booking", "airbnb", "uber", "samsung",
]

# Number of feature columns expected by the model. Filled in / verified
# by train_model.py and cross-checked at prediction time so a mismatch
# between training-time and prediction-time features fails loudly
# instead of silently producing garbage predictions.
FEATURE_COLUMNS_KEY = "feature_columns"

# ---------------------------------------------------------------------------
# VirusTotal integration (optional threat-intel check, separate from and
# in addition to the ML model). Read from an environment variable so no
# API key is ever committed to the project files. Get a free key at:
# https://www.virustotal.com/gui/join-us  (free tier: ~4 requests/minute)
# ---------------------------------------------------------------------------
VIRUSTOTAL_API_KEY = os.environ.get("VIRUSTOTAL_API_KEY", "")
VIRUSTOTAL_URL_ENDPOINT = "https://www.virustotal.com/api/v3/urls"
VIRUSTOTAL_REQUEST_TIMEOUT = 8  # seconds
