"""
threat_intel.py
-----------------
An OPTIONAL, SEPARATE check against VirusTotal's URL database, shown
alongside (never instead of) the ML model's own prediction.

Why this is a useful complement, not a replacement, for the model:
  - VirusTotal aggregates ~70 antivirus/security vendors' verdicts on
    URLs that have already been submitted and scanned by someone. It is
    REACTIVE: a brand-new phishing URL that nobody has scanned yet will
    come back "unknown", even if it's obviously malicious.
  - The ML model in src/prediction.py is PROACTIVE: it judges the URL's
    text/structure directly, so it can flag brand-new phishing URLs
    that no one has reported anywhere yet -- but it can also be wrong
    in ways a confirmed vendor verdict never is.
  - Showing both gives the user the best of each: an instant "this is
    already confirmed bad" signal when available, plus the model's own
    independent judgement either way.

This module NEVER raises on network/API failure -- a slow or unreachable
VirusTotal should never break the app's core prediction feature. Every
failure path returns a dict with "checked": False and a human-readable
reason, and the caller (app.py) treats that as "no threat-intel data
available", not an error.
"""

import base64

import requests

from src.config import (
    VIRUSTOTAL_API_KEY,
    VIRUSTOTAL_REQUEST_TIMEOUT,
    VIRUSTOTAL_URL_ENDPOINT,
)


def _url_to_vt_id(url: str) -> str:
    """VirusTotal identifies a URL by the URL-safe base64 encoding of the
    URL string, with the '=' padding stripped, per their API docs."""
    return base64.urlsafe_b64encode(url.encode("utf-8")).decode("utf-8").rstrip("=")


def check_virustotal(url: str) -> dict:
    """
    Look up an existing VirusTotal report for this exact URL.

    Returns a dict that is always safe to render, even on failure:
        {
          "configured": bool,   # was an API key found at all?
          "checked": bool,      # did we get a real answer from VT?
          "found": bool,        # has VT seen/scanned this URL before?
          "verdict": str,       # "malicious" | "suspicious" | "clean" |
                                 # "unknown" | "not_configured" | "error"
          "malicious_count": int,
          "suspicious_count": int,
          "harmless_count": int,
          "total_engines": int,
          "permalink": str | None,
          "message": str,       # human-readable summary for the UI
        }

    This only performs a LOOKUP of an existing report (a single fast GET
    request). It does not submit the URL for a brand-new scan, because
    that requires polling for results (VirusTotal scans are asynchronous
    and can take 30-60+ seconds), which would make every prediction slow
    and would quickly exhaust the free tier's ~4-requests/minute limit.
    A URL VirusTotal has never seen before comes back "unknown" -- which
    is itself informative alongside the ML model's own verdict.
    """
    base = {
        "configured": bool(VIRUSTOTAL_API_KEY),
        "checked": False,
        "found": False,
        "verdict": "not_configured",
        "malicious_count": 0,
        "suspicious_count": 0,
        "harmless_count": 0,
        "total_engines": 0,
        "permalink": None,
        "message": "",
    }

    if not VIRUSTOTAL_API_KEY:
        base["message"] = (
            "VirusTotal is not configured. Set the VIRUSTOTAL_API_KEY "
            "environment variable to enable this check (free key at "
            "virustotal.com/gui/join-us)."
        )
        return base

    url_id = _url_to_vt_id(url)
    endpoint = f"{VIRUSTOTAL_URL_ENDPOINT}/{url_id}"
    headers = {"x-apikey": VIRUSTOTAL_API_KEY}

    try:
        resp = requests.get(endpoint, headers=headers, timeout=VIRUSTOTAL_REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        base["verdict"] = "error"
        base["message"] = f"Could not reach VirusTotal: {exc}"
        return base

    base["checked"] = True

    if resp.status_code == 404:
        base["found"] = False
        base["verdict"] = "unknown"
        base["message"] = (
            "VirusTotal has no prior report for this exact URL. This does "
            "NOT mean it's safe -- it just hasn't been scanned/reported "
            "yet, which is common for brand-new phishing pages."
        )
        return base

    if resp.status_code == 401:
        base["verdict"] = "error"
        base["message"] = "VirusTotal rejected the API key (401 Unauthorized)."
        return base

    if resp.status_code == 429:
        base["verdict"] = "error"
        base["message"] = "VirusTotal rate limit hit (free tier: ~4 requests/minute). Try again shortly."
        return base

    if resp.status_code != 200:
        base["verdict"] = "error"
        base["message"] = f"VirusTotal returned an unexpected status: {resp.status_code}."
        return base

    try:
        data = resp.json()
        stats = data["data"]["attributes"]["last_analysis_stats"]
        malicious = int(stats.get("malicious", 0))
        suspicious = int(stats.get("suspicious", 0))
        harmless = int(stats.get("harmless", 0))
        undetected = int(stats.get("undetected", 0))
        total = malicious + suspicious + harmless + undetected
    except (KeyError, ValueError, TypeError) as exc:
        base["verdict"] = "error"
        base["message"] = f"Could not parse VirusTotal's response: {exc}"
        return base

    base["found"] = True
    base["malicious_count"] = malicious
    base["suspicious_count"] = suspicious
    base["harmless_count"] = harmless
    base["total_engines"] = total
    base["permalink"] = f"https://www.virustotal.com/gui/url/{url_id}"

    if malicious > 0:
        base["verdict"] = "malicious"
        base["message"] = f"{malicious} of {total} security vendors flag this URL as malicious."
    elif suspicious > 0:
        base["verdict"] = "suspicious"
        base["message"] = f"{suspicious} of {total} security vendors flag this URL as suspicious."
    else:
        base["verdict"] = "clean"
        base["message"] = f"No vendors out of {total} flag this URL as malicious (previously scanned)."

    return base
