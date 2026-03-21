/**
 * popup.js — SentimentScope Chrome Extension
 * Communicates with Flask API, renders charts and tables.
 */

// ── Config ────────────────────────────────────────────
const API_BASE = "http://localhost:5000"; // replace with EC2 URL in production

// ── DOM refs ─────────────────────────────────────────
const analyzeBtn    = document.getElementById("analyzeBtn");
const loadingState  = document.getElementById("loadingState");
const loadingProg   = document.getElementById("loadingProgress");
const resultsDiv    = document.getElementById("results");
const emptyState    = document.getElementById("emptyState");
const errorBanner   = document.getElementById("errorBanner");
const errorMsg      = document.getElementById("errorMsg");

let pieChartInst  = null;
let trendChartInst = null;

// ── Helpers ───────────────────────────────────────────
function show(el)  { el.classList.remove("hidden"); }
function hide(el)  { el.classList.add("hidden");    }

function setProgress(msg) { loadingProg.textContent = msg; }

function showError(msg) {
  errorMsg.textContent = msg;
  show(errorBanner);
}

function clearError() { hide(errorBanner); }

// ── Get current tab video ID ──────────────────────────
async function getVideoId() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.url) return null;
  const url = new URL(tab.url);
  return url.hostname.includes("youtube.com") ? url.searchParams.get("v") : null;
}

// ── Analyze ───────────────────────────────────────────
analyzeBtn.addEventListener("click", async () => {
  clearError();
  const videoId = await getVideoId();
  if (!videoId) {
    showError("Please open a YouTube video tab first.");
    return;
  }

  // UI: loading
  hide(emptyState);
  hide(resultsDiv);
  show(loadingState);
  analyzeBtn.disabled = true;

  try {
    setProgress("Fetching comments from YouTube…");
    const res = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: videoId }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `Server error ${res.status}`);
    }

    setProgress("Running sentiment model…");
    const data = await res.json();

    setProgress("Rendering charts…");
    await renderResults(data);

    hide(loadingState);
    show(resultsDiv);
  } catch (err) {
    hide(loadingState);
    show(emptyState);
    showError(err.message || "Unknown error. Is the API running?");
  } finally {
    analyzeBtn.disabled = false;
  }
});

// ── Render everything ─────────────────────────────────
async function renderResults(data) {
  renderStats(data.stats);
  renderPie(data.sentiment_counts);
  renderTrend(data.trend);
  renderWordCloud(data.word_freq);
  renderComments(data.top_comments);
}

// ── Stats ─────────────────────────────────────────────
function renderStats(stats) {
  document.getElementById("statTotal").textContent  = stats.total ?? "—";
  document.getElementById("statUnique").textContent = stats.unique ?? "—";
  document.getElementById("statAvgLen").textContent = stats.avg_length ? `${stats.avg_length}w` : "—";
  const score = stats.avg_sentiment_score;
  document.getElementById("statScore").textContent  = score != null ? score.toFixed(1) : "—";
}

// ── Pie Chart ─────────────────────────────────────────
function renderPie(counts) {
  if (pieChartInst) pieChartInst.destroy();

  const ctx = document.getElementById("pieChart").getContext("2d");
  const pos = counts.positive  ?? 0;
  const neg = counts.negative  ?? 0;
  const neu = counts.neutral   ?? 0;
  const total = pos + neg + neu || 1;

  pieChartInst = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Positive", "Negative", "Neutral"],
      datasets: [{
        data: [pos, neg, neu],
        backgroundColor: ["#4ade80", "#f87171", "#60a5fa"],
        borderColor: "#141417",
        borderWidth: 3,
        hoverOffset: 6,
      }],
    },
    options: {
      responsive: false,
      cutout: "68%",
      plugins: { legend: { display: false }, tooltip: { enabled: true } },
    },
  });

  // Legend
  const legend = document.getElementById("pieLegend");
  legend.innerHTML = [
    { label: "Positive", color: "#4ade80", val: pos },
    { label: "Negative", color: "#f87171", val: neg },
    { label: "Neutral",  color: "#60a5fa", val: neu },
  ].map(({ label, color, val }) => `
    <div class="legend-item">
      <div class="legend-dot" style="background:${color}"></div>
      <span>${label}</span>
      <span class="legend-pct">${((val / total) * 100).toFixed(0)}%</span>
    </div>
  `).join("");
}

// ── Trend Chart ───────────────────────────────────────
function renderTrend(trend) {
  if (trendChartInst) trendChartInst.destroy();

  const ctx = document.getElementById("trendChart").getContext("2d");
  const labels = trend.map(d => d.month);

  trendChartInst = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Positive",
          data: trend.map(d => d.positive),
          borderColor: "#4ade80",
          backgroundColor: "rgba(74,222,128,0.08)",
          tension: 0.4,
          fill: true,
          pointRadius: 3,
          pointHoverRadius: 5,
        },
        {
          label: "Negative",
          data: trend.map(d => d.negative),
          borderColor: "#f87171",
          backgroundColor: "rgba(248,113,113,0.08)",
          tension: 0.4,
          fill: true,
          pointRadius: 3,
          pointHoverRadius: 5,
        },
        {
          label: "Neutral",
          data: trend.map(d => d.neutral),
          borderColor: "#60a5fa",
          backgroundColor: "rgba(96,165,250,0.08)",
          tension: 0.4,
          fill: true,
          pointRadius: 3,
          pointHoverRadius: 5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          display: true,
          position: "bottom",
          labels: {
            color: "#7a7a8a",
            font: { size: 9, family: "Space Mono" },
            boxWidth: 8,
            usePointStyle: true,
            padding: 10,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#3d3d4d", font: { size: 9 } },
          grid: { color: "rgba(255,255,255,0.04)" },
        },
        y: {
          ticks: { color: "#3d3d4d", font: { size: 9 } },
          grid: { color: "rgba(255,255,255,0.04)" },
          beginAtZero: true,
        },
      },
    },
  });
}

// ── Word Cloud ────────────────────────────────────────
function renderWordCloud(wordFreq) {
  const container = document.getElementById("wordCloud");
  container.innerHTML = "";

  const words = Object.entries(wordFreq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 60)
    .map(([text, size]) => ({ text, size }));

  const maxFreq = words[0]?.size || 1;
  const colors  = ["#a78bfa", "#60a5fa", "#4ade80", "#f87171", "#fbbf24", "#e879f9"];

  const width  = container.offsetWidth  || 520;
  const height = container.offsetHeight || 160;

  const layout = d3.layout.cloud()
    .size([width, height])
    .words(words.map(w => ({
      text: w.text,
      size: 10 + (w.size / maxFreq) * 26,
    })))
    .padding(3)
    .rotate(() => 0)
    .font("DM Sans")
    .fontSize(d => d.size)
    .on("end", draw);

  layout.start();

  function draw(words) {
    const svg = d3.select(container)
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", `translate(${width / 2},${height / 2})`);

    svg.selectAll("text")
      .data(words)
      .enter()
      .append("text")
      .style("font-size", d => `${d.size}px`)
      .style("font-family", "DM Sans")
      .style("font-weight", d => d.size > 24 ? "600" : "400")
      .style("fill", (_, i) => colors[i % colors.length])
      .style("opacity", 0)
      .attr("text-anchor", "middle")
      .attr("transform", d => `translate(${d.x},${d.y})`)
      .text(d => d.text)
      .transition()
      .duration(300)
      .delay((_, i) => i * 15)
      .style("opacity", 1);
  }
}

// ── Comments Table ────────────────────────────────────
function renderComments(comments) {
  const tbody = document.getElementById("commentsBody");
  tbody.innerHTML = comments.slice(0, 25).map((c, i) => {
    const label = c.sentiment === 1 ? "pos" : c.sentiment === -1 ? "neg" : "neu";
    const text  = c.sentiment === 1 ? "Positive" : c.sentiment === -1 ? "Negative" : "Neutral";
    const scoreStr = c.score != null ? c.score.toFixed(2) : "—";
    const escaped  = escapeHtml(c.comment);
    return `
      <tr>
        <td>${i + 1}</td>
        <td title="${escaped}">${escaped}</td>
        <td><span class="badge badge-${label}">${text}</span></td>
        <td><span class="score-val">${scoreStr}</span></td>
      </tr>
    `;
  }).join("");
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
