/* app.js — CoverBuild RL Frontend Logic */

const API_BASE = "http://localhost:8000";

// ─── Startup ────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  checkAPIStatus();
  bindSliders();
  bindCharCounters();
  setInterval(checkAPIStatus, 15000);
});

// ─── API Status Check ────────────────────────────────────────────────────────
async function checkAPIStatus() {
  const dot = document.getElementById("status-dot");
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      const data = await res.json();
      dot.className = "status-dot online";
      updateMetrics(data);
      fetchModelInfo();
    } else {
      dot.className = "status-dot offline";
      setMetricOffline();
    }
  } catch {
    dot.className = "status-dot offline";
    setMetricOffline();
  }
}

async function fetchModelInfo() {
  try {
    const res = await fetch(`${API_BASE}/model/info`);
    if (!res.ok) return;
    const data = await res.json();
    document.getElementById("val-model").textContent = data.model_name || "—";
    document.getElementById("val-vocab").textContent = data.vocab_size?.toLocaleString() || "—";
  } catch {}
}

function updateMetrics(healthData) {
  document.getElementById("val-status").textContent = healthData.status === "healthy" ? "✅ Online" : "⚠️ Degraded";
  document.getElementById("val-device").textContent = healthData.device?.toUpperCase() || "—";
}

function setMetricOffline() {
  document.getElementById("val-status").textContent = "🔴 Offline";
  document.getElementById("val-device").textContent = "—";
  document.getElementById("val-model").textContent = "—";
  document.getElementById("val-vocab").textContent = "—";
}

// ─── Sliders ─────────────────────────────────────────────────────────────────
function bindSliders() {
  const tempSlider = document.getElementById("temperature");
  const lenSlider  = document.getElementById("max-length");
  tempSlider.addEventListener("input", () => {
    document.getElementById("temp-val").textContent = parseFloat(tempSlider.value).toFixed(1);
  });
  lenSlider.addEventListener("input", () => {
    document.getElementById("len-val").textContent = lenSlider.value;
  });
}

// ─── Char Counters ────────────────────────────────────────────────────────────
function bindCharCounters() {
  bindCounter("job-description",  "job-char-count",     2000);
  bindCounter("applicant-profile","profile-char-count", 1000);
}
function bindCounter(inputId, counterId, max) {
  const el = document.getElementById(inputId);
  const counter = document.getElementById(counterId);
  el.addEventListener("input", () => {
    const n = el.value.length;
    counter.textContent = `${n} / ${max}`;
    counter.style.color = n > max * 0.9 ? "#ff6b9d" : "var(--text-3)";
  });
}

// ─── Load Sample Inputs ───────────────────────────────────────────────────────
const SAMPLES = [
  {
    job: "Software Engineer at a fintech startup. Requirements: Python, REST APIs, SQL, collaborative team player.",
    profile: "3 years Python experience, built production REST APIs, B.Sc. Computer Science, strong GitHub portfolio.",
  },
  {
    job: "Machine Learning Engineer at an AI lab. Requirements: PyTorch, HuggingFace transformers, CUDA, prior research experience.",
    profile: "PhD candidate in ML, published 2 papers on LLM alignment, contributed to open-source HuggingFace projects.",
  },
  {
    job: "DevOps Engineer at a SaaS company. Requirements: Kubernetes, Docker, CI/CD pipelines, Linux, Terraform.",
    profile: "4 years DevOps, reduced deployment time by 60%, Certified Kubernetes Administrator (CKA).",
  },
];

function loadSample() {
  const s = SAMPLES[Math.floor(Math.random() * SAMPLES.length)];
  typeLine("job-description", s.job);
  typeLine("applicant-profile", s.profile);
  document.getElementById("job-char-count").textContent = `${s.job.length} / 2000`;
  document.getElementById("profile-char-count").textContent = `${s.profile.length} / 1000`;
}

function typeLine(elId, text) {
  const el = document.getElementById(elId);
  el.value = "";
  let i = 0;
  const timer = setInterval(() => {
    el.value += text[i++];
    if (i >= text.length) clearInterval(timer);
  }, 8);
}

// ─── Generate Cover Letter ────────────────────────────────────────────────────
let isGenerating = false;

async function generateLetter() {
  if (isGenerating) return;

  const job     = document.getElementById("job-description").value.trim();
  const profile = document.getElementById("applicant-profile").value.trim();
  const temp    = parseFloat(document.getElementById("temperature").value);
  const maxLen  = parseInt(document.getElementById("max-length").value);

  if (!job)     { showToast("⚠️ Please enter a job description"); return; }
  if (!profile) { showToast("⚠️ Please enter an applicant profile"); return; }

  isGenerating = true;
  const btn = document.getElementById("generate-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span> Generating...';

  hideOutput();
  showLoading();
  animateSteps();

  try {
    const res = await fetch(`${API_BASE}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_description:    job,
        applicant_profile:  profile,
        max_length:         maxLen,
        temperature:        temp,
        top_p:              0.95,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Generation failed");
    }

    const data = await res.json();
    hideLoading();
    showOutput(data);

  } catch (err) {
    hideLoading();
    showToast(`❌ ${err.message}`);
    console.error(err);

    // Fallback: show a demo if API is offline
    if (err.message.includes("fetch") || err.message.includes("Failed")) {
      showDemoOutput(job, profile);
    }
  } finally {
    isGenerating = false;
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">✨</span> Generate Cover Letter';
  }
}

// ─── Loading animation ────────────────────────────────────────────────────────
let stepTimer = null;
function animateSteps() {
  const steps = ["step-1","step-2","step-3"];
  let i = 0;
  steps.forEach(id => document.getElementById(id).className = "step");
  document.getElementById(steps[0]).className = "step active";
  stepTimer = setInterval(() => {
    if (i < steps.length - 1) {
      document.getElementById(steps[i]).className = "step";
      i++;
      document.getElementById(steps[i]).className = "step active";
    }
  }, 1200);
}

function showLoading()  { document.getElementById("loading-state").style.display = "block"; document.getElementById("empty-state").style.display = "none"; }
function hideLoading()  { document.getElementById("loading-state").style.display = "none"; if (stepTimer) clearInterval(stepTimer); }
function hideOutput()   {
  document.getElementById("output-text").style.display = "none";
  document.getElementById("score-panel").style.display = "none";
  document.getElementById("copy-btn").style.display = "none";
  document.getElementById("empty-state").style.display = "none";
}

// ─── Render Output ────────────────────────────────────────────────────────────
function showOutput(data) {
  const outputEl = document.getElementById("output-text");
  outputEl.style.display = "block";
  typewriterEffect(outputEl, data.cover_letter);

  // Score panel
  renderScorePanel(data);
  document.getElementById("score-panel").style.display = "block";
  document.getElementById("copy-btn").style.display = "inline-block";
}

function typewriterEffect(el, text) {
  el.textContent = "";
  let i = 0;
  const chunk = 4;
  const timer = setInterval(() => {
    el.textContent += text.slice(i, i + chunk);
    i += chunk;
    el.scrollTop = el.scrollHeight;
    if (i >= text.length) { el.textContent = text; clearInterval(timer); }
  }, 16);
}

function renderScorePanel(data) {
  const bd = data.score_breakdown;
  const score = data.reward_score;

  // Badge
  const badge = document.getElementById("score-badge");
  badge.textContent = score.toFixed(3);
  if (score >= 0.5) { badge.className = "score-badge excellent"; }
  else if (score >= 0.1) { badge.className = "score-badge good"; }
  else if (score >= -0.2) { badge.className = "score-badge average"; }
  else { badge.className = "score-badge poor"; }

  // Bars
  setDimBar("structure",   bd.structure);
  setDimBar("relevance",   bd.relevance);
  setDimBar("tone",        bd.tone);
  setDimBar("conciseness", bd.conciseness);

  // Meta
  document.getElementById("meta-time").textContent   = `⏱ ${Math.round(data.generation_time_ms)} ms`;
  document.getElementById("meta-tokens").textContent = `🔢 ${data.generated_tokens} tokens`;
}

function setDimBar(dim, val) {
  const pct = Math.max(0, Math.min(1, val)) * 100;
  document.getElementById(`bar-${dim}`).style.width = `${pct}%`;
  document.getElementById(`val-${dim}`).textContent = val.toFixed(2);
}

// ─── Demo fallback (no API) ───────────────────────────────────────────────────
function showDemoOutput(job, profile) {
  const demo = {
    cover_letter: `Dear Hiring Manager,

I am excited to apply for this opportunity. Having reviewed the requirements carefully, I believe my background aligns well with what you are seeking.

Throughout my career, I have demonstrated the ability to deliver high-quality results while collaborating effectively with cross-functional teams. My technical expertise and passion for continuous learning make me a strong candidate.

I would welcome the opportunity to contribute to your team's mission and grow alongside talented colleagues.

Thank you sincerely for your time and consideration.

Sincerely,
[Applicant Name]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  DEMO MODE — Start the API with:
   python scripts/serve.py
Then refresh this page for live generation.`,
    reward_score: 0.42,
    score_breakdown: { structure: 0.7, relevance: 0.4, tone: 0.6, conciseness: 0.8, aggregate: 0.42 },
    generation_time_ms: 0,
    generated_tokens: 0,
    model_version: "demo",
    prompt_tokens: 0,
  };
  showOutput(demo);
  showToast("📡 Demo mode — start the API for live generation");
}

// ─── Copy letter ─────────────────────────────────────────────────────────────
function copyLetter() {
  const text = document.getElementById("output-text").textContent;
  navigator.clipboard.writeText(text).then(() => showToast("✅ Copied to clipboard!"));
}

// ─── Score Existing Letter ────────────────────────────────────────────────────
async function scoreLetter() {
  const job    = document.getElementById("score-job").value.trim();
  const letter = document.getElementById("score-letter").value.trim();
  if (!job || !letter) { showToast("⚠️ Please fill in both fields"); return; }

  try {
    const res = await fetch(`${API_BASE}/reward`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_description: job, cover_letter: letter }),
    });
    if (!res.ok) throw new Error("Scoring failed");
    const data = await res.json();
    showScoreResult(data);
  } catch (e) {
    // fallback heuristic demo
    showScoreResult({
      score: 0.35,
      breakdown: { structure: 0.6, relevance: 0.3, tone: 0.5, conciseness: 0.7, aggregate: 0.35 },
      interpretation: "Demo score — start API for real scoring",
    });
  }
}

function showScoreResult(data) {
  document.getElementById("score-empty").style.display = "none";
  const resEl = document.getElementById("score-result");
  resEl.style.display = "block";

  document.getElementById("big-score-number").textContent = data.score.toFixed(3);
  document.getElementById("big-score-interp").textContent = data.interpretation;

  const barsEl = document.getElementById("scorer-bars");
  const bd = data.breakdown;
  barsEl.innerHTML = ["structure","relevance","tone","conciseness"].map(dim => `
    <div class="score-bar-row">
      <span class="dim-label">${capitalize(dim)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max(0,Math.min(1,bd[dim]))*100}%"></div></div>
      <span class="dim-val">${bd[dim].toFixed(2)}</span>
    </div>
  `).join("");
}

// ─── Toast ────────────────────────────────────────────────────────────────────
let toastEl = null;
function showToast(msg) {
  if (toastEl) document.body.removeChild(toastEl);
  toastEl = document.createElement("div");
  toastEl.className = "toast";
  toastEl.textContent = msg;
  document.body.appendChild(toastEl);
  requestAnimationFrame(() => { requestAnimationFrame(() => { toastEl.classList.add("show"); }); });
  setTimeout(() => {
    toastEl.classList.remove("show");
    setTimeout(() => { if (toastEl) { document.body.removeChild(toastEl); toastEl = null; } }, 400);
  }, 3000);
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }
