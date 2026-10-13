<<<<<<< HEAD
# URLGuard — An Explainable Machine Learning Framework for Phishing URL Detection and Risk Assessment

A college final-year project that predicts whether a URL is **Phishing** or
**Legitimate**, assigns a **risk level**, and explains the prediction using
**SHAP** — all from a Flask backend with a custom HTML/CSS/JS frontend.

Every number in this README (accuracy, precision, recall, confusion matrix)
was produced by actually running `src/train_model.py` on the real
PhiUSIIL dataset on this machine. Nothing here is invented.

---

## 1. Project objective

Given a URL typed by a user, decide whether it is likely to lead to a
phishing page, and — critically — **explain why**, so the tool is useful for
learning, not just a black-box yes/no answer.

## 2. Problem statement

Phishing URLs try to imitate legitimate ones (fake login pages, spoofed
banking domains, etc.). Blocklists catch known phishing URLs but miss new
ones. A model trained on the *structure* of URLs can generalise to URLs it
has never seen, using only information a user has before clicking: the URL
text itself.

## 3. Features (of the application)

- Enter any URL and get an instant prediction — no page is ever visited or
  fetched; only the URL string is analysed.
- **Brand-lookalike / typosquat detection** via Levenshtein edit distance
  against ~46 well-known brand names (e.g. catches `paypa1.com`).
- A risk category (Low / Medium / High) with a documented, transparent rule.
- A real, per-prediction SHAP explanation: which features pushed the
  prediction toward phishing or legitimate, and by how much.
- **An optional VirusTotal threat-intel check**, shown as a separate,
  clearly-labelled signal alongside the model's own verdict.
- A Model Performance page with real test-set metrics, a confusion matrix,
  and a model-comparison table.
- A custom HTML/CSS/JS frontend (no template UI framework), served by Flask.

## 4. Architecture / workflow

```
User enters URL (frontend)
        │  fetch() POST /api/predict
        ▼
Flask app.py
        │
        ▼
validate_url()            src/feature_extraction.py
        │
        ▼
extract_features()        src/feature_extraction.py   (30 lexical features,
        │                                               incl. brand-lookalike)
        ▼
trained sklearn Pipeline  models/phishing_model.joblib  (src/prediction.py)
        │
        ▼
phishing probability → risk_category()                 src/prediction.py
        │
        ├──────────────────────────────┐
        ▼                              ▼
explain_prediction()          check_virustotal()
src/explainability.py (SHAP)   src/threat_intel.py (optional, separate)
        │                              │
        └──────────────┬───────────────┘
                        ▼
        JSON response → rendered by frontend/js/app.js
```

The same `extract_features()` function is used during training
(`src/train_model.py`) and at prediction time (`src/prediction.py`), so the
model always sees features computed the same way. `models/metadata.json`
stores the exact feature-column list used for training, and
`src/prediction.py` checks it on startup — if `feature_extraction.py` is ever
changed without retraining, the app fails loudly instead of silently
misaligning columns.

## 5. Dataset

**PhiUSIIL Phishing URL Dataset** (Prasad & Chandra, 2024, *Computers &
Security*), from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/967

- 235,795 rows: 134,850 legitimate URLs and 100,945 phishing URLs.
- `label == 1` means legitimate in the *original* file. `src/data_loader.py`
  **flips this** so `label == 1` means **phishing** everywhere else in this
  project, since phishing is the class we care most about detecting.

**Only two columns are used: `URL` and `label`.** The other ~50 columns in
the raw CSV (`URLTitleMatchScore`, `HasFavicon`, `NoOfImage`, `LineOfCode`,
etc.) are computed from the **webpage's source code**, not from the URL
text. A real prediction pipeline only ever receives a URL string from the
user, so training on those columns would be data leakage — the model would
learn to rely on information it can never have at prediction time. All 28
features actually used are recomputed from the raw URL in
`src/feature_extraction.py`.

To obtain the dataset yourself: download `PhiUSIIL_Phishing_URL_Dataset.csv`
from the UCI link above (or `pip install ucimlrepo` and fetch id=967) and
place it at `data/dataset.csv`.

## 6. Feature engineering

30 features, computed **only** from the URL string (see
`src/feature_extraction.py` for exact definitions):

| Group | Features |
|---|---|
| Length | `url_length`, `hostname_length`, `path_length`, `query_length`, `longest_path_token_length` |
| Counts | `num_dots`, `num_subdomains`, `num_hyphens_url`, `num_hyphens_hostname`, `num_digits_url`, `num_digits_hostname`, `num_special_chars`, `num_params`, `path_depth` |
| Ratios | `digit_ratio`, `letter_ratio`, `special_char_ratio` |
| Binary flags | `has_ip_address`, `has_https`, `has_at_symbol`, `has_shortener`, `has_port`, `has_punycode`, `has_double_slash_in_path`, `has_suspicious_tld` |
| Keyword | `suspicious_keyword_count` |
| Randomness | `url_entropy`, `hostname_entropy` (Shannon entropy) |
| **Brand impersonation** | `brand_edit_distance`, `is_brand_lookalike` |

Features that require visiting the page, checking WHOIS/domain age, or an
SSL certificate were **deliberately excluded** — they are slow, can fail,
and are not "reliably extractable from a new URL" as required.

### 6a. Brand-lookalike detection (Levenshtein edit distance)

`brand_edit_distance` and `is_brand_lookalike` compare each label of the
hostname against a list of ~46 well-known brand names (`KNOWN_BRANDS` in
`src/config.py`) using Levenshtein edit distance — the minimum number of
single-character insertions/deletions/substitutions needed to turn one
string into another (implemented from scratch in
`src/feature_extraction.py`, no extra dependency). `paypa1.com` is
distance 1 from `paypal`; `arnazon.com` is distance 2 from `amazon`.
`is_brand_lookalike` fires when a label is *close* to a brand (distance 1
or 2) **without being an exact match** — an exact match (the domain
genuinely is `paypal.com`) is normal legitimate use and scores 0.

This directly targets typosquatting, which none of the other 28 features
were designed to catch. It became the **2nd most important feature
globally** after retraining (see `reports/shap_summary.png`).

**A bug worth mentioning in a viva:** the first version of this feature
compared *every* hostname label against the brand list — including the
TLD itself. Since short TLDs like `.org` happened to be edit-distance 2
from short brand abbreviations like `irs`, `wikipedia.org` was flagged as
a brand lookalike purely by coincidence. The fix excludes the last label
(the TLD) from the comparison. This is caught by
`test_brand_lookalike_does_not_flag_unrelated_domain` in
`tests/test_features.py` as a regression test.

**Known limitation:** because the comparison is per-label, a brand name
*embedded inside* a longer hyphenated label (e.g.
`paypal-secure-login.tk`, where the whole label is far from `paypal` in
edit distance) is not reliably caught. A substring/n-gram-based check
would catch more of these, at the cost of more false positives — a good
candidate for a future iteration.

## 7. Data cleaning and leakage prevention

`src/data_loader.py` and `src/train_model.py`:

1. Keep only `URL` and `label`; drop 425 exact-duplicate URLs
   (235,795 → 235,370 rows).
2. Drop 8 rows whose URL could not be parsed at all (235,362 rows used).
3. **Split by registrable domain** (`GroupShuffleSplit`), not by row, so the
   same domain can never appear in both the training set and the test set —
   a stronger leakage guard than a plain random split, since near-identical
   URLs from the same domain would otherwise make the test set easier than
   real-world unseen traffic.

## 8. ML algorithms and model selection

Three scikit-learn models were compared with 3-fold stratified
cross-validation (F1-score on the phishing class), on a 49,999-row
stratified subsample of the training set for speed — this subsample is
used **only** to choose which algorithm to commit to; the final numbers
below come from a full refit and a single evaluation on the untouched test
set.

| Model | CV F1 (mean ± std) | Time |
|---|---|---|
| Logistic Regression | 0.9719 ± 0.0022 | 0.3s |
| Random Forest | 0.9764 ± 0.0015 | 8.6s |
| **Gradient Boosting (selected)** | **0.9765 ± 0.0016** | 11.5s |

**Gradient Boosting** was selected (highest CV F1, essentially tied with
Random Forest) and refit on the full 189,609-row training set. (Random
Forest's `max_depth` is intentionally capped at 12 in
`_candidate_models()` — an earlier, uncapped run picked Random Forest and
produced a 55 MB model file for a ~0.2-point F1 gain, which isn't a
sensible trade for a project meant to be emailed/uploaded/cloned easily.
Gradient Boosting's default depth-3 trees stay small regardless, which is
part of why it's the more practical choice here even when a deeper
Random Forest narrowly wins on paper.)

## 9. Model evaluation (held-out test set, 45,753 URLs, never used for
training or model selection)

| Metric | Value |
|---|---|
| Accuracy | **96.51%** |
| Precision (phishing) | **99.35%** |
| Recall (phishing) | **92.85%** |
| F1-score (phishing) | **95.99%** |
| ROC-AUC | **0.9889** |

Confusion matrix (rows = true label, columns = predicted label):

|  | Predicted Legitimate | Predicted Phishing |
|---|---|---|
| **Actual Legitimate** | 25,058 | 125 |
| **Actual Phishing** | 1,470 | 19,100 |

**Why precision and recall, not just accuracy?** The two error types have
different real-world costs:
- A **false negative** (1,470 cases here) is a phishing URL the model calls
  legitimate — the dangerous miss, since a user could be attacked.
- A **false positive** (125 cases here) is a legitimate URL flagged as
  phishing — annoying, and erodes trust in the tool if frequent.

This model's precision (99.35%) is higher than its recall (92.85%), meaning
it is very rarely wrong when it says "phishing", but still misses about 7%
of actual phishing URLs. That trade-off is exactly the kind of thing
accuracy alone would hide.

**These numbers moved twice during development — both times for
documented, honest reasons, not tuning for a better-looking number: a drop
after fixing the trailing-slash bug (§9a), then a rise after adding and
then bugfixing the brand-lookalike features (§6a).**

### 9a. A debugging story worth including in a viva: the trailing-slash bug

While testing the deployed app, `https://www.google.com` was correctly
scored as 0.3% phishing risk, but `https://www.google.com/` — the exact
same site, differing only by one trailing slash a browser adds
automatically — was scored at **99.4% phishing risk**.

Investigating why revealed a genuine, serious dataset bias: in the raw
PhiUSIIL dataset, **100% of legitimate URLs have `path_length == 0`**
(every legitimate example is a bare homepage; none has so much as a
trailing slash), while real phishing URLs mostly do have a path. A single
trailing character was therefore an almost perfect — but meaningless —
signal for the model to exploit, and it did.

**The fix** (`src/feature_extraction.py`): a URL whose path is exactly `"/"`
with no query or fragment is now canonicalised — for feature-computation
purposes only — to have no path at all, before *any* feature (not just
`path_length`, but also `url_length`, `num_special_chars`,
`special_char_ratio`, and `url_entropy`, which had silently re-encoded the
same one-character difference) is computed. The displayed/normalised URL
shown to the user is untouched; only the numbers fed to the model change.

**The consequence of the fix**: retraining afterwards, accuracy dropped
from 99.08% to 95.49%. That drop is not a regression — it means a real
chunk of the original "accuracy" was the model learning to detect *"is
this URL byte-for-byte identical in length/structure to a bare training
domain"* rather than anything about phishing. The lower, post-fix number is
the more honest one.

**Takeaway for the viva:** a model can pass every automated metric and
still be trivially broken by something as small as a trailing slash, if the
training data has a structural quirk the metric can't see. Testing a model
against inputs a user would actually type — not just the held-out test
split — caught this; the held-out split couldn't, because it shared the
exact same bias. This is documented further in §17 (Limitations).

*(Related, narrower note: this dataset's URLs were also collected and
curated by its authors at one point in time; even after this fix, a strong
score here is not a guarantee of the same performance on completely
different, unfiltered real-world traffic.)*

### 9b. The second bugfix: brand comparison including the TLD

After adding `brand_edit_distance` / `is_brand_lookalike` (§6a), the first
version had a real bug: it compared every hostname label — including the
TLD itself — against the brand list, which meant `wikipedia.org` scored as
a brand lookalike purely because `org` happens to be edit-distance 2 from
`irs`. Excluding the TLD from the comparison (keeping only the
registrable-domain and subdomain labels) fixed the false positive, and the
feature became genuinely useful: `brand_edit_distance` is now the **2nd
most globally important feature** (see `reports/shap_summary.png`), behind
only `has_https`.

**A separate, practical finding from this round of retraining:** an
uncapped Random Forest (`max_depth=20`) narrowly won model selection but
produced a **55 MB** model file — a bad trade for a ~0.2-point F1 gain, in
a project meant to be zipped, emailed, and cloned. Capping
`max_depth=12` for Random Forest (`src/train_model.py`) brought Gradient
Boosting back to the top spot (its depth-3 trees stay small regardless of
how many estimators are used) at a **44 KB** file size. This is a real
engineering trade-off worth naming explicitly in a viva: the
highest-scoring model on one metric isn't automatically the right choice
once file size, load time, or deployment constraints matter too.

## 10. Explainable AI

`src/explainability.py` uses **SHAP's `TreeExplainer`** for whichever
tree-based model was selected (Random Forest or Gradient Boosting — both
supported; `LinearExplainer` is used instead if Logistic Regression were
selected). For a given URL:

- Real SHAP values are computed from the actual fitted model and the
  actual feature vector of that URL — nothing is templated per-URL.
- The top contributing features and their direction (toward phishing /
  toward legitimate) are turned into plain-English sentences using a fixed
  sentence *template*, but the feature, value, and direction filled into it
  are always the real SHAP output for that specific prediction.
- A global feature-importance plot (`reports/shap_summary.png`) is computed
  once from a 500-row sample of the test set, for the Model Performance page.

**If `shap` is not installed**, the app automatically falls back to the
model's built-in `feature_importances_` and labels every explanation
"fallback_feature_importance" in the UI — this fallback can only say a
feature was influential, not which direction it pushed the prediction,
and the app is honest about that distinction rather than guessing.
Run `pip install shap` (in `requirements.txt`) for full per-prediction
attributions, then re-run `python -m src.generate_reports` to regenerate
the global summary plot with true SHAP values.

## 11. Risk assessment

The phishing probability *p* from the model maps to a risk level via a
single documented rule in `src/config.py`:

| Phishing probability | Risk level |
|---|---|
| p < 0.30 | Low Risk |
| 0.30 ≤ p < 0.70 | Medium Risk |
| p ≥ 0.70 | High Risk |

The predicted class itself uses the standard p ≥ 0.5 cutoff. These
thresholds are a simple, transparent heuristic on top of the model's
output, not a separately trained rule — and, as the calibration-curve cell
in `notebooks/model_analysis.ipynb` shows, tree-ensemble probabilities are
not perfectly calibrated, so "70% probability" is a strong signal, not an
exact frequency.

## 11a. Threat-intel check (VirusTotal) — a second, separate signal

Alongside the ML model's own verdict, the app can optionally check a URL
against **VirusTotal**, a service that aggregates verdicts from ~70
antivirus/security vendors for URLs that have already been scanned and
reported (`src/threat_intel.py`).

**Why this is a useful complement, not a replacement, for the ML model:**

| | VirusTotal (blocklist) | The ML model |
|---|---|---|
| Catches already-reported phishing | ✅ instantly, very reliably | ❌ only if it resembles training patterns |
| Catches brand-new phishing (minutes/hours old) | ❌ not yet reported anywhere | ✅ this is its actual job |
| Needs internet access | ✅ required | ❌ works fully offline |
| Explains *why* something is risky | ❌ just a yes/no vendor tally | ✅ SHAP shows exactly why |

A blocklist is **reactive** (needs someone to get phished and report it
first); the model is **proactive** (judges structure directly, so it can
flag a URL nobody has ever seen before). The app shows both, clearly
labelled as separate signals, rather than merging them into one score.

**Setup (optional — the app works fine without this):**
1. Create a free account at https://www.virustotal.com/gui/join-us
2. Copy your API key from your VirusTotal profile page
3. Set it as an environment variable before running the app:
   ```powershell
   $env:VIRUSTOTAL_API_KEY = "your_key_here"     # PowerShell, current session
   ```
   or add it permanently via Windows' "Edit environment variables for your
   account" settings, so you don't have to re-set it every session.

**What it actually does:** a single fast lookup (not a new scan) of
VirusTotal's *existing* report for the exact URL. It does **not** submit
the URL for a brand-new scan, because that's asynchronous (scans take
30–60+ seconds) and would make every prediction slow and quickly exhaust
the free tier's ~4-requests/minute limit. A URL VirusTotal has never seen
comes back "Not Previously Scanned" — which is itself informative
alongside the model's own verdict.

**Failure handling:** no API key, an invalid key, a network timeout, or a
VirusTotal outage all degrade gracefully to a "Check Unavailable" /
"Not Configured" message in the UI — they never break the core prediction
feature. This was verified during development by deliberately testing with
no key, a fake key, and a blocked network connection.

## 12. Installation (Windows + VS Code)

```powershell
# 1. Clone / open the project folder in VS Code

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset
#    Download PhiUSIIL_Phishing_URL_Dataset.csv from
#    https://archive.ics.uci.edu/dataset/967 and save it as:
#    data/dataset.csv

# 5. (Optional) Enable the VirusTotal check -- see §11a
$env:VIRUSTOTAL_API_KEY = "your_key_here"
```

## 13. How to train the model

```powershell
python -m src.train_model
```

This prints progress, saves:
- `models/phishing_model.joblib` — the fitted scikit-learn pipeline
- `models/metadata.json` — feature list, model name, training config
- `reports/metrics.json`, `reports/model_comparison.json` — real metrics
- `reports/confusion_matrix.png`
- `reports/shap_background_sample.csv` — used by the next step

Then generate the global explainability plot:

```powershell
python -m src.generate_reports
```

## 14. How to run the application

```powershell
python app.py
```

Open **http://127.0.0.1:5000** in a browser. The Flask app serves the
`frontend/` HTML/CSS/JS and the `/api/predict` and `/api/model-performance`
endpoints. If `VIRUSTOTAL_API_KEY` isn't set, the app still runs fully —
the threat-intel panel just shows "Not Configured" (see §11a).

## 15. Example usage

1. Go to the **Analyze** tab.
2. Enter a URL, e.g. `http://secure-paypal-login-verify.tk/account/update.php?user=12345`
3. Click **Analyze URL**.
4. See: prediction (Phishing), phishing probability, risk level (High
   Risk), the top SHAP-driven reasons (e.g. suspicious keywords, unusual
   TLD), the VirusTotal threat-intel panel, and the full extracted
   feature table.

**A second example worth trying — brand impersonation:** enter
`https://paypa1.com/login`. This isn't caught by any keyword list (there's
no "verify" or "secure" in it) — it's caught by `brand_edit_distance`,
since `paypa1` is a single-character edit away from `paypal`. Compare it
against `https://www.paypal.com` (the real domain) to see
`is_brand_lookalike` correctly stay at 0 for the genuine site.

Try a real, well-known URL (e.g. `https://www.wikipedia.org`) to see the
contrast.

## 16. Running the tests

```powershell
pytest tests/ -v
```

Covers: URL feature extraction (validation + all 30 features, including
brand-lookalike detection and its TLD-comparison regression test),
the prediction pipeline (risk categories, probability consistency,
missing-model handling, invalid URLs), and the Flask API layer
(valid/invalid requests, missing model handling).

## 17. Limitations

- **Every legitimate URL in the training dataset is a bare homepage**
  (`path_length == 0` for 100% of them) — see §9a for how this was
  discovered (a trailing slash flipping `google.com`'s verdict) and fixed
  at the feature level. The fix removes the *specific* brittleness (a
  harmless trailing slash), but the *underlying* bias remains: the model
  has still never seen a legitimate URL with a real path or query string
  during training, so it may still be somewhat more suspicious of deep
  links (e.g. `wikipedia.org/wiki/...`, `amazon.com/dp/...`) than it
  should be, purely because that pattern was absent from the "legitimate"
  half of its training data. A meaningful future improvement is
  augmenting the legitimate class with real, non-homepage URLs.
- **Brand-lookalike detection only catches whole-label similarity**
  (§6a) — a brand name embedded inside a longer hyphenated label (e.g.
  `paypal-secure-login.tk`) is not reliably caught, since the whole label
  is far from the brand name in edit distance even though the brand name
  appears as a substring. A substring/n-gram-based check would catch more
  of these.
- **The known-brands list (`KNOWN_BRANDS` in `src/config.py`) is a fixed
  list of ~46 names I chose** — any brand not on it gets no lookalike
  protection. Expanding this from a larger public corpus is a natural
  improvement.
- **VirusTotal is a lookup, not a scan** (§11a) — it only reports what's
  already been scanned/reported by someone else. A brand-new phishing URL
  will show "Not Previously Scanned" regardless of how dangerous it is;
  that's expected, and exactly why it's shown as a secondary signal
  alongside the ML model's own independent judgement, not a replacement
  for it.
- **Lexical features only.** No page content, WHOIS/domain-age, or SSL
  certificate checks are used, so a phishing page hosted at a
  clean-looking URL (e.g. via a compromised legitimate domain) could be
  missed by the ML model (though VirusTotal may still catch it if
  reported).
- **Subdomain counting is a heuristic** (splits on `.`), not a full
  public-suffix-list lookup, so multi-part TLDs like `co.uk` are slightly
  over-counted as subdomains.
- **Single dataset.** PhiUSIIL's URLs were collected and curated by its
  authors at one point in time; real-world phishing tactics evolve, and a
  model trained here would need periodic retraining in a real deployment.
- **Probabilities are not perfectly calibrated** (see the calibration
  curve in the notebook) — risk levels should be read as ordinal signals,
  not exact percentages.
- **SHAP is only as fast as the model**; for very high request volumes a
  production system would want to cache the explainer instance (this app
  already does, via `get_detector()`), and possibly precompute background
  distributions.

## 18. Future improvements

- Add a second, independent phishing-URL dataset for cross-dataset
  evaluation, to test generalisation beyond PhiUSIIL.
- Replace the simple subdomain heuristic with a public-suffix-list-based
  implementation (e.g. the `tldextract` package).
- Replace whole-label brand comparison with a substring/n-gram-based
  check to catch brand names embedded in longer hyphenated labels.
- Expand `KNOWN_BRANDS` from a larger public brand-name corpus.
- Add periodic retraining and a feedback loop (user-reported
  false positives/negatives).
- Add rate limiting and basic auth if ever deployed beyond localhost.
- Optionally submit not-yet-scanned URLs to VirusTotal for a fresh scan
  (currently only an existing-report lookup is done — see §11a for why),
  with a clear "scanning..." state in the UI while polling.
- Optionally add a lightweight content-based feature (e.g. via a safe,
  read-only HTTP HEAD request) as an opt-in "deep scan" mode, clearly
  separated from the pure-lexical mode this project focuses on.

## 19. Project structure

```
phishing-url-detection/
├── app.py                        # Flask app: serves frontend + /api routes
├── requirements.txt
├── README.md
├── pytest.ini
├── data/
│   └── dataset.csv               # PhiUSIIL dataset (place here yourself)
├── models/
│   ├── phishing_model.joblib     # trained sklearn Pipeline
│   └── metadata.json             # feature list, model name, config
├── reports/
│   ├── metrics.json
│   ├── model_comparison.json
│   ├── confusion_matrix.png
│   ├── shap_summary.png
│   └── shap_background_sample.csv
├── src/
│   ├── config.py                 # paths, seeds, risk thresholds, keyword/brand lists, VT config
│   ├── data_loader.py            # load/clean/dedup dataset
│   ├── feature_extraction.py     # validate_url + extract_features (30 features)
│   ├── preprocessing.py          # URLs -> feature matrix, scaler helpers
│   ├── train_model.py            # split, compare models, select, save
│   ├── evaluation.py             # metrics + confusion matrix plot
│   ├── generate_reports.py       # global SHAP summary plot
│   ├── prediction.py             # load model, predict, risk category
│   ├── explainability.py         # SHAP per-URL + global explanations
│   └── threat_intel.py           # VirusTotal lookup (separate, optional signal)
├── frontend/
│   ├── index.html                # single-page app (Home/Analyze/Performance/About)
│   ├── css/style.css
│   └── js/app.js                 # calls the Flask API, renders results
├── notebooks/
│   └── model_analysis.ipynb      # EDA, model comparison, calibration check
└── tests/
    ├── test_features.py
    ├── test_prediction.py
    └── test_app_logic.py
```

## 20. Citation

Prasad, A., & Chandra, S. (2024). PhiUSIIL: A diverse security profile
empowered phishing URL detection framework based on similarity index and
incremental learning. *Computers & Security*, 136, 103545.
https://doi.org/10.1016/j.cose.2023.103545
=======
# Phishing_URL-Detection
>>>>>>> 568bf948725ea2c3db8b02549b6353ac69f437ec
