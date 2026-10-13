/* app.js
 * ------
 * All data on this page comes from the Flask API (/api/predict and
 * /api/model-performance), which in turn comes from the real trained
 * model. Nothing here hard-codes a prediction, a metric, or an
 * explanation -- this file only fetches and renders.
 */

// ---------------------------------------------------------------------
// Simple client-side "router": show one .view section at a time.
// ---------------------------------------------------------------------
function navigateTo(viewId) {
  document.querySelectorAll(".view").forEach(el => {
    el.classList.toggle("is-active", el.id === viewId);
  });
  document.querySelectorAll(".nav__link").forEach(el => {
    el.classList.toggle("is-active", el.dataset.nav === viewId);
  });
  window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });

  if (viewId === "performance") {
    loadModelPerformance();
  }
}

document.querySelectorAll("[data-nav]").forEach(el => {
  el.addEventListener("click", () => navigateTo(el.dataset.nav));
});

// ---------------------------------------------------------------------
// Home page hero stats, pulled from the real metrics endpoint.
// ---------------------------------------------------------------------
async function loadHeroStats() {
  try {
    const res = await fetch("/api/model-performance");
    if (!res.ok) return;
    const data = await res.json();
    const stats = document.querySelectorAll("#hero-stats .stat__value");
    stats[0].textContent = formatPct(data.accuracy);
    stats[1].textContent = formatPct(data.precision);
    stats[2].textContent = formatPct(data.recall);
  } catch (err) {
    // Home page still works without stats; performance page will
    // surface the real error if the model truly isn't trained yet.
    console.warn("Could not load hero stats:", err);
  }
}

// ---------------------------------------------------------------------
// Analyze URL
// ---------------------------------------------------------------------
const form = document.getElementById("analyze-form");
const urlInput = document.getElementById("url-input");
const analyzeBtn = document.getElementById("analyze-btn");
const scanProgress = document.getElementById("scan-progress");
const errorBox = document.getElementById("analyze-error");
const resultBox = document.getElementById("result");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const url = urlInput.value.trim();
  if (!url) return;

  errorBox.hidden = true;
  resultBox.hidden = true;
  scanProgress.hidden = false;
  analyzeBtn.disabled = true;

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();

    if (!res.ok) {
      showError(data.error || "Something went wrong analysing that URL.");
      return;
    }
    renderResult(data);
  } catch (err) {
    showError("Could not reach the server. Is app.py running?");
  } finally {
    scanProgress.hidden = true;
    analyzeBtn.disabled = false;
  }
});

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function renderResult(data) {
  resultBox.hidden = false;

  // Verdict card
  const verdictLabel = document.getElementById("verdict-label");
  verdictLabel.textContent = data.prediction;
  verdictLabel.className = "verdict-card__value " +
    (data.prediction === "Phishing" ? "is-phishing" : "is-legitimate");

  let urlLine = data.normalised_url;
  if (data.scheme_was_assumed) {
    urlLine += "  (no scheme given — assumed http://)";
  }
  document.getElementById("verdict-url").textContent = urlLine;

  // Risk card
  const riskBadge = document.getElementById("risk-badge");
  riskBadge.textContent = data.risk_level;
  riskBadge.className = "risk-card__badge " + riskClass(data.risk_level);

  const pct = data.phishing_probability * 100;
  document.getElementById("gauge-value").textContent = pct.toFixed(1) + "%";

  const fill = document.getElementById("gauge-fill");
  const arcLength = 173;
  fill.style.strokeDashoffset = String(arcLength * (1 - data.phishing_probability));
  fill.style.stroke = riskColor(data.risk_level);

  // Explanation
  const methodBox = document.getElementById("explanation-method");
  methodBox.textContent = data.explanation.method === "shap"
    ? "Explanation generated with SHAP (exact attribution from the trained model)."
    : "SHAP is not installed — showing an approximate, model-based feature importance instead. Run 'pip install shap' for exact per-prediction attributions.";

  const reasonsList = document.getElementById("reasons-list");
  reasonsList.innerHTML = "";
  data.explanation.plain_language.forEach(sentence => {
    const li = document.createElement("li");
    li.textContent = sentence;
    reasonsList.appendChild(li);
  });

  renderContributions(data.explanation.contributions);
  renderThreatIntel(data.threat_intel);
  renderFeatureTable(data.features);
}

const TI_LABELS = {
  malicious: "Known Malicious", suspicious: "Suspicious", clean: "Clean (previously scanned)",
  unknown: "Not Previously Scanned", not_configured: "Not Configured", error: "Check Unavailable",
};

function renderThreatIntel(ti) {
  const box = document.getElementById("threat-intel-box");
  const label = TI_LABELS[ti.verdict] || ti.verdict;
  let html = `<div class="ti-row"><span class="ti-badge ${ti.verdict}">${label}</span>`;
  if (ti.checked && ti.found) {
    html += `<span style="font-family:var(--font-mono);font-size:.85rem;color:var(--text-muted)">${ti.malicious_count} malicious / ${ti.suspicious_count} suspicious / ${ti.harmless_count} harmless of ${ti.total_engines} vendors</span>`;
  }
  html += `</div><p class="ti-msg">${ti.message}</p>`;
  if (ti.permalink) {
    html += `<a class="ti-link" href="${ti.permalink}" target="_blank" rel="noopener">View full VirusTotal report →</a>`;
  }
  box.innerHTML = html;
}

function riskClass(riskLevel) {
  if (riskLevel === "Low Risk") return "is-low";
  if (riskLevel === "Medium Risk") return "is-medium";
  return "is-high";
}
function riskColor(riskLevel) {
  if (riskLevel === "Low Risk") return "#2FBF71";
  if (riskLevel === "Medium Risk") return "#F2B705";
  return "#E5484D";
}

function renderContributions(contributions) {
  const chart = document.getElementById("contrib-chart");
  chart.innerHTML = "";
  const maxAbs = Math.max(...contributions.map(c => Math.abs(c.impact)), 1e-9);

  contributions.slice(0, 8).forEach(c => {
    const row = document.createElement("div");
    row.className = "contrib-row";

    const name = document.createElement("div");
    name.className = "contrib-row__name";
    name.textContent = c.feature;

    const track = document.createElement("div");
    track.className = "contrib-row__track";

    const bar = document.createElement("div");
    const widthPct = (Math.abs(c.impact) / maxAbs) * 48; // max 48% of half-track
    bar.className = "contrib-row__bar " + (c.impact >= 0 ? "positive" : "negative");
    bar.style.width = widthPct + "%";

    track.appendChild(bar);
    row.appendChild(name);
    row.appendChild(track);
    chart.appendChild(row);
  });
}

function renderFeatureTable(features) {
  const table = document.getElementById("feature-table");
  table.innerHTML = "";
  Object.entries(features).forEach(([key, value]) => {
    const cell = document.createElement("div");
    cell.className = "feature-table__cell";
    const displayVal = typeof value === "number" && !Number.isInteger(value)
      ? value.toFixed(3)
      : value;
    cell.innerHTML = `<span>${key}</span><span>${displayVal}</span>`;
    table.appendChild(cell);
  });
}

// ---------------------------------------------------------------------
// Model performance page
// ---------------------------------------------------------------------
let performanceLoaded = false;

async function loadModelPerformance() {
  if (performanceLoaded) return;
  try {
    const res = await fetch("/api/model-performance");
    const data = await res.json();
    if (!res.ok) {
      console.warn(data.error);
      return;
    }
    document.getElementById("m-accuracy").textContent = formatPct(data.accuracy);
    document.getElementById("m-precision").textContent = formatPct(data.precision);
    document.getElementById("m-recall").textContent = formatPct(data.recall);
    document.getElementById("m-f1").textContent = formatPct(data.f1_score);
    document.getElementById("m-auc").textContent = data.roc_auc ? data.roc_auc.toFixed(4) : "n/a";

    renderComparisonTable(data.model_comparison);
    performanceLoaded = true;
  } catch (err) {
    console.warn("Could not load model performance:", err);
  }
}

function renderComparisonTable(comparison) {
  const container = document.getElementById("compare-table");
  if (!comparison || Object.keys(comparison).length === 0) {
    container.textContent = "No comparison data found.";
    return;
  }
  const best = Object.entries(comparison).sort((a, b) => b[1].cv_f1_mean - a[1].cv_f1_mean)[0][0];

  let html = "<table><thead><tr><th>Model</th><th>CV F1 (mean)</th><th>CV F1 (std)</th><th>Sample size</th></tr></thead><tbody>";
  Object.entries(comparison).forEach(([name, stats]) => {
    const rowClass = name === best ? "is-winner" : "";
    html += `<tr class="${rowClass}">
      <td>${name}${name === best ? " (selected)" : ""}</td>
      <td>${stats.cv_f1_mean.toFixed(4)}</td>
      <td>${stats.cv_f1_std.toFixed(4)}</td>
      <td>${stats.comparison_sample_size.toLocaleString()}</td>
    </tr>`;
  });
  html += "</tbody></table>";
  container.innerHTML = html;
}

function formatPct(value) {
  if (value === null || value === undefined) return "n/a";
  return (value * 100).toFixed(2) + "%";
}

// ---------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------
loadHeroStats();
