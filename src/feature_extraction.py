"""
feature_extraction.py
----------------------
This is the single most important file in the project.

Every feature the model is trained on, and every feature used at
prediction time, comes from the two functions below:

    validate_url(raw_url)   -> normalised URL string, or raises ValueError
    extract_features(url)   -> dict of feature_name -> value

Because train_model.py and prediction.py both import THIS module and
call THIS function, the training-time and prediction-time features are
guaranteed to be computed the same way. This is what prevents the
"works in training, breaks in production" bug that is extremely common
in URL-classification projects.

All features are derived ONLY from the URL string itself. Nothing here
makes a network request, so predicting on a URL never visits it.
"""

import math
import re
from functools import lru_cache
from urllib.parse import urlparse

from src.config import KNOWN_BRANDS, SHORTENER_DOMAINS, SUSPICIOUS_KEYWORDS, SUSPICIOUS_TLDS

# The ordered list of feature names this module produces. train_model.py
# saves this exact list into models/metadata.json, and prediction.py
# checks the incoming feature dict against it before ever calling the
# model, so a silent column-order mismatch is impossible.
FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "longest_path_token_length",
    "num_dots",
    "num_subdomains",
    "num_hyphens_url",
    "num_hyphens_hostname",
    "num_digits_url",
    "num_digits_hostname",
    "num_special_chars",
    "num_params",
    "path_depth",
    "digit_ratio",
    "letter_ratio",
    "special_char_ratio",
    "has_ip_address",
    "has_https",
    "has_at_symbol",
    "has_shortener",
    "has_port",
    "has_punycode",
    "has_double_slash_in_path",
    "has_suspicious_tld",
    "suspicious_keyword_count",
    "url_entropy",
    "hostname_entropy",
    "brand_edit_distance",
    "is_brand_lookalike",
]

_IP_PATTERN = re.compile(
    r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$"
)


class InvalidURLError(ValueError):
    """Raised when a string cannot be treated as a URL to analyse."""


def validate_url(raw_url: str) -> str:
    """
    Normalise user input into something urlparse can handle consistently,
    and reject input that clearly is not a URL at all.

    - Strips whitespace.
    - Adds an "http://" scheme if none is present (common when users paste
      a bare domain like "example.com"), and remembers that the scheme was
      *assumed*, since that matters for the HTTPS feature.
    - Rejects empty strings and strings with no plausible hostname.
    """
    if raw_url is None:
        raise InvalidURLError("No URL was provided.")

    url = raw_url.strip()
    if not url:
        raise InvalidURLError("The URL is empty.")

    if len(url) > 2048:
        raise InvalidURLError("The URL is too long to be a valid web address.")

    # Reject obvious non-URLs early with a clear message.
    if " " in url and "://" not in url:
        # Still allow spaces that show up after a scheme is added below;
        # this only catches things like "hello world" typed by mistake.
        pass

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", url):
        # No scheme present -> assume http and let the caller know via
        # the returned string (prediction.py surfaces this to the UI).
        url = "http://" + url

    parsed = urlparse(url)
    if not parsed.hostname:
        raise InvalidURLError(
            "Could not find a valid domain name in that URL. "
            "Please enter something like 'https://example.com'."
        )

    # A hostname needs at least one dot OR be an IP address to be a
    # realistic web address (localhost-style names are rejected).
    if not _IP_PATTERN.match(parsed.hostname) and "." not in parsed.hostname:
        raise InvalidURLError(
            "That does not look like a complete domain name "
            "(expected something like 'example.com')."
        )

    return url


def _levenshtein(a: str, b: str, max_dist: int) -> int:
    """Edit-distance between a and b, with an early-exit once the
    minimum possible distance already exceeds max_dist (branch-and-bound
    pruning). Returns max_dist + 1 (a sentinel "too far to matter") once
    pruned, since this project only ever cares about small distances
    (0-2) -- computing the exact large distance would be wasted work
    across ~48 brands x 235k URLs during training."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i] + [0] * len(b)
        row_min = curr_row[0]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr_row[j] = min(
                prev_row[j] + 1,          # deletion
                curr_row[j - 1] + 1,      # insertion
                prev_row[j - 1] + cost,   # substitution
            )
            row_min = min(row_min, curr_row[j])
        if row_min > max_dist:
            return max_dist + 1  # can't possibly recover to <= max_dist
        prev_row = curr_row
    return prev_row[-1]


@lru_cache(maxsize=300_000)
def _brand_lookalike_signal(hostname: str) -> tuple:
    """
    Compare each label of the hostname against a list of well-known
    brand names (src/config.py -> KNOWN_BRANDS). Cached per hostname,
    since the same hostname often recurs across many rows/predictions.

    Returns (brand_edit_distance, is_brand_lookalike):
      - brand_edit_distance: smallest edit distance from any hostname
        label to any known brand, capped at 5 (beyond that, "how
        similar" stops being meaningful and just adds noise).
      - is_brand_lookalike: 1 if a label is CLOSE to a brand (edit
        distance 1 or 2) WITHOUT being an exact match -- i.e. it looks
        like an impersonation attempt (e.g. "paypa1", "arnazon"), not
        the real site. An exact match (the domain genuinely IS the
        brand name, e.g. "paypal.com") scores 0, since that is normal,
        legitimate use, not lookalike behaviour.
    """
    if not hostname:
        return 5, 0

    parts = hostname.split(".")
    # Exclude the TLD itself (last label) -- otherwise short TLDs like
    # "org" or "com" get compared against short brand abbreviations
    # ("irs", "dhl") and produce false positives on completely innocent
    # domains (caught via testing: "wikipedia.org" was flagging as a
    # lookalike of "irs" purely because "org" and "irs" are both 3
    # characters and happen to be edit-distance 2 apart).
    labels = [p for p in parts[:-1] if p and not p.isdigit()]
    if not labels:
        return 5, 0

    best_distance = 5
    exact_hit = False
    for label in labels:
        for brand in KNOWN_BRANDS:
            if label == brand:
                exact_hit = True
                continue
            d = _levenshtein(label, brand, best_distance)
            if d < best_distance:
                best_distance = d

    if exact_hit:
        return 0, 0

    is_lookalike = 1 if 1 <= best_distance <= 2 else 0
    return best_distance, is_lookalike


def _shannon_entropy(s: str) -> float:
    """Shannon entropy of a string, in bits. Higher = more 'random-looking',
    which phishing URLs (auto-generated subdomains, encoded tokens) tend
    to have more of than typical human-chosen legitimate URLs."""
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(s)
    entropy = 0.0
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


def _count_subdomains(hostname: str) -> int:
    """Number of subdomain labels, excluding the registrable domain + TLD.
    This is a simple heuristic (splits on '.') rather than a full public-
    suffix-list lookup, which keeps the feature dependency-free and fast,
    at the cost of slightly over-counting for multi-part TLDs like
    'co.uk'. This tradeoff is called out explicitly in the README."""
    if _IP_PATTERN.match(hostname):
        return 0
    parts = hostname.split(".")
    # e.g. "www.mail.example.com" -> 4 parts -> 2 subdomains (www, mail)
    return max(0, len(parts) - 2)


def extract_features(url: str) -> dict:
    """
    Compute the full feature vector for a single (already-validated) URL.

    Returns a plain dict of {feature_name: numeric_value} using exactly
    the names in FEATURE_NAMES, so it can be turned into a single-row
    DataFrame with a fixed, known column order.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    # Normalise a bare trailing slash and nothing else ("/") to "" before
    # computing path-based features. "https://example.com" and
    # "https://example.com/" point at the same resource, but a browser
    # (or a user pasting a link) very commonly adds the trailing slash.
    # Without this, path_length/path_depth flip from 0 to 1 on that
    # single character alone -- and because every legitimate URL in the
    # PhiUSIIL training data happens to have path_length == 0 (see the
    # README's Limitations section), a trained model can treat that one
    # character as an almost perfect phishing signal. This normalisation
    # fixes that specific brittleness without hiding the deeper, real
    # dataset-bias limitation, which remains documented.
    if path == "/":
        path = ""

    # Also canonicalise the URL text itself for every full-URL-derived
    # feature (length, special-char counts, ratios, entropy): a bare
    # trailing slash with no query/fragment carries no real information,
    # but because it was the only structural difference between
    # otherwise-identical training URLs, leaving it in let features like
    # url_length and url_entropy silently re-encode the same brittle
    # signal the path-only fix above was meant to remove.
    full_url = url
    if parsed.path == "/" and not parsed.query and not parsed.fragment and url.endswith("/"):
        full_url = url[:-1]

    letters = sum(c.isalpha() for c in full_url)
    digits_url = sum(c.isdigit() for c in full_url)
    digits_host = sum(c.isdigit() for c in hostname)
    specials = sum(not c.isalnum() for c in full_url)

    path_tokens = [t for t in re.split(r"[/\-_]", path) if t]
    longest_token = max((len(t) for t in path_tokens), default=0)

    keyword_hits = sum(
        1 for kw in SUSPICIOUS_KEYWORDS if kw in full_url.lower()
    )

    registered_domain = ".".join(hostname.split(".")[-2:]) if hostname else ""
    has_shortener = 1 if registered_domain in SHORTENER_DOMAINS else 0

    tld = hostname.split(".")[-1].lower() if "." in hostname else ""
    has_suspicious_tld = 1 if tld in SUSPICIOUS_TLDS else 0

    brand_distance, is_lookalike = _brand_lookalike_signal(hostname)

    features = {
        "url_length": len(full_url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "longest_path_token_length": longest_token,
        "num_dots": full_url.count("."),
        "num_subdomains": _count_subdomains(hostname),
        "num_hyphens_url": full_url.count("-"),
        "num_hyphens_hostname": hostname.count("-"),
        "num_digits_url": digits_url,
        "num_digits_hostname": digits_host,
        "num_special_chars": specials,
        "num_params": len(query.split("&")) if query else 0,
        "path_depth": len([p for p in path.split("/") if p]),
        "digit_ratio": digits_url / len(full_url) if full_url else 0.0,
        "letter_ratio": letters / len(full_url) if full_url else 0.0,
        "special_char_ratio": specials / len(full_url) if full_url else 0.0,
        "has_ip_address": 1 if _IP_PATTERN.match(hostname) else 0,
        "has_https": 1 if parsed.scheme == "https" else 0,
        "has_at_symbol": 1 if "@" in full_url else 0,
        "has_shortener": has_shortener,
        "has_port": 1 if parsed.port else 0,
        "has_punycode": 1 if "xn--" in hostname else 0,
        "has_double_slash_in_path": 1 if "//" in path else 0,
        "has_suspicious_tld": has_suspicious_tld,
        "suspicious_keyword_count": keyword_hits,
        "url_entropy": _shannon_entropy(full_url),
        "hostname_entropy": _shannon_entropy(hostname),
        "brand_edit_distance": brand_distance,
        "is_brand_lookalike": is_lookalike,
    }

    # Defensive check: this dict must always match FEATURE_NAMES exactly.
    # If a feature is ever added/removed above without updating the list,
    # fail loudly here rather than silently misaligning columns later.
    assert set(features.keys()) == set(FEATURE_NAMES), (
        "extract_features() output does not match FEATURE_NAMES. "
        "Update FEATURE_NAMES if you changed the feature set."
    )
    return features
