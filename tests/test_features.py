"""
test_features.py
------------------
Tests for src/feature_extraction.py. Run with:
    python -m pytest tests/ -v
"""

import pytest

from src.feature_extraction import (
    FEATURE_NAMES,
    InvalidURLError,
    extract_features,
    validate_url,
)


def test_validate_url_adds_missing_scheme():
    result = validate_url("example.com")
    assert result.startswith("http://")


def test_validate_url_keeps_https_scheme():
    result = validate_url("https://example.com")
    assert result == "https://example.com"


def test_validate_url_strips_whitespace():
    result = validate_url("   https://example.com   ")
    assert result == "https://example.com"


@pytest.mark.parametrize("bad_input", ["", "   ", None])
def test_validate_url_rejects_empty(bad_input):
    with pytest.raises(InvalidURLError):
        validate_url(bad_input)


def test_validate_url_rejects_no_domain():
    with pytest.raises(InvalidURLError):
        validate_url("http://")


def test_validate_url_rejects_incomplete_hostname():
    with pytest.raises(InvalidURLError):
        validate_url("http://localhost")


def test_validate_url_rejects_overlong_url():
    with pytest.raises(InvalidURLError):
        validate_url("http://example.com/" + "a" * 3000)


def test_extract_features_returns_all_expected_columns():
    features = extract_features("https://www.example.com/path?query=1")
    assert set(features.keys()) == set(FEATURE_NAMES)
    assert all(isinstance(v, (int, float)) for v in features.values())


def test_extract_features_detects_https():
    https_features = extract_features("https://example.com")
    http_features = extract_features("http://example.com")
    assert https_features["has_https"] == 1
    assert http_features["has_https"] == 0


def test_extract_features_detects_ip_address():
    features = extract_features("http://192.168.1.1/login")
    assert features["has_ip_address"] == 1


def test_extract_features_detects_at_symbol():
    features = extract_features("http://example.com@evil.com")
    assert features["has_at_symbol"] == 1


def test_extract_features_counts_subdomains():
    features = extract_features("https://mail.accounts.example.com")
    assert features["num_subdomains"] == 2


def test_extract_features_detects_suspicious_keywords():
    features = extract_features("http://secure-login-verify.example.com")
    assert features["suspicious_keyword_count"] >= 2


def test_extract_features_url_length_matches_actual_length():
    url = "https://example.com/a/b/c"
    features = extract_features(url)
    assert features["url_length"] == len(url)


def test_extract_features_entropy_is_nonnegative():
    features = extract_features("https://example.com")
    assert features["url_entropy"] >= 0
    assert features["hostname_entropy"] >= 0


def test_brand_lookalike_flags_typosquat():
    features = extract_features("https://paypa1.com/login")
    assert features["is_brand_lookalike"] == 1
    assert 1 <= features["brand_edit_distance"] <= 2


def test_brand_lookalike_does_not_flag_real_brand():
    features = extract_features("https://www.paypal.com")
    assert features["is_brand_lookalike"] == 0
    assert features["brand_edit_distance"] == 0


def test_brand_lookalike_does_not_flag_unrelated_domain():
    # Regression test for a real bug found during development: the TLD
    # itself ("org") was being compared against short brand abbreviations
    # ("irs"), flagging wikipedia.org as a lookalike. The TLD must be
    # excluded from the comparison.
    features = extract_features("https://www.wikipedia.org")
    assert features["is_brand_lookalike"] == 0


def test_trailing_slash_does_not_change_features():
    """Regression test: a bare trailing slash ('https://x.com' vs
    'https://x.com/') must not change ANY feature. Earlier versions of
    this project let it flip path_length (and, via url_length/entropy/
    special-char-ratio, several other features too), which combined with
    a real dataset bias -- every legitimate URL in PhiUSIIL happens to
    have path_length == 0 -- made a trained model flip verdicts on
    google.com vs google.com/. See feature_extraction.py for the fix."""
    f1 = extract_features("https://www.google.com")
    f2 = extract_features("https://www.google.com/")
    assert f1 == f2


def test_real_path_is_still_counted_after_slash_fix():
    """Make sure fixing the bare-trailing-slash case didn't accidentally
    zero out real, meaningful paths."""
    features = extract_features("https://example.com/login")
    assert features["path_length"] == len("/login")
    assert features["path_depth"] == 1

