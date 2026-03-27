/* ── Sample texts ───────────────────────────────────────── */
const SAMPLES = {
  human: "okay so i was literally dying at work today lmao my boss called a meeting at 4:58pm on a friday?? who does that!! anyway i couldnt focus bcz i kept thinking about the pizza i ordered. got home, pizza was cold. 10/10 still ate it all ngl",
  ai: "The implementation of machine learning algorithms has demonstrated remarkable potential across a diverse range of industries, including healthcare, finance, and transportation.",
  tricky: "okay so i've been really into productivity systems lately lol. i wake up at 6am and follow a structured morning routine."
};

/* ── Char counter ───────────────────────────────────────── */
const textInput = document.getElementById("textInput");
const charCount = document.getElementById("charCount");

textInput.addEventListener("input", () => {
  charCount.textContent = textInput.value.length;
});

/* ── Load sample ────────────────────────────────────────── */
window.loadSample = function (type) {
  const sample = SAMPLES[type] || "";
  textInput.value = sample;
  charCount.textContent = sample.length;
};

/* ── Load stats ─────────────────────────────────────────── */
async function loadStats() {
  try {
    const res = await fetch("/stats");
    const data = await res.json();
    const fmt = v => (v != null && !isNaN(v)) ? (v * 100).toFixed(1) + "%" : "—";

    document.getElementById("lrAcc").textContent = fmt(data.lr_accuracy);
    document.getElementById("nbAcc").textContent = fmt(data.nb_accuracy);
    document.getElementById("rfAcc").textContent = fmt(data.rf_accuracy);
    document.getElementById("svmAcc").textContent = fmt(data.svm_accuracy);
    document.getElementById("ensAcc").textContent = fmt(data.ensemble_accuracy);
  } catch (e) {
    console.warn("Stats load failed");
  }
}
loadStats();

/* ── Analyze ─────────────────────────────────────────────── */
window.analyze = async function () {
  const text = textInput.value.trim();
  if (!text) {
    alert("Please enter text!");
    return;
  }

  const btn = document.getElementById("analyzeBtn");
  const spinner = document.getElementById("btnSpinner");

  btn.disabled = true;
  spinner.classList.remove("hidden");

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    const data = await res.json();
    renderResults(data);

  } catch (err) {
    console.error("Error:", err);
  } finally {
    btn.disabled = false;
    spinner.classList.add("hidden");
  }
};

/* ── Render Results ──────────────────────────────────────── */
function renderResults(data) {
  if (!data) return;

  document.getElementById("resultPlaceholder").classList.add("hidden");
  document.getElementById("resultContent").classList.remove("hidden");

  const isAI = data.prediction === 1;
  const isAIH = data.detection_type === "AI-Humanized";

  /* Prediction Badge */
  const badge = document.getElementById("predictionBadge");
  badge.className = "prediction-badge " + (isAIH ? "aih" : isAI ? "ai" : "human");

  document.getElementById("predIcon").textContent =
    isAIH ? "🎭" : isAI ? "🤖" : "👤";

  document.getElementById("predLabel").textContent =
    isAIH ? "AI-Humanized Text" : (data.label || "—");

  /* Verdict */
  document.getElementById("verdictSurface").textContent = data.label || "—";
  document.getElementById("verdictDeep").textContent = "Analyzed";
  document.getElementById("verdictFinal").textContent =
    isAIH ? "AI-Humanized" : isAI ? "AI" : "Human";

  /* Confidence */
  const conf = data.confidence || 0;
  document.getElementById("confValue").textContent = conf + "%";
  document.getElementById("confBarFill").style.width = conf + "%";

  const ph = data.proba_human || 0;
  const pa = data.proba_ai || 0;

  document.getElementById("probaHuman").textContent = ph + "%";
  document.getElementById("probaAI").textContent = pa + "%";

  document.getElementById("splitBarHuman").style.width = ph + "%";
  document.getElementById("splitBarAI").style.width = pa + "%";

  /* Decision */
  const dec = data.decision || {};
  document.getElementById("decisionBadge").textContent = dec.label || "";
  document.getElementById("decisionMsg").textContent = dec.message || "";

  /* Insight */
  document.getElementById("insightBox").textContent = "💡 Analysis complete";

  /* ===========================
     ✅ FIXED MODEL TABLE
  =========================== */
  const tbody = document.getElementById("modelsTableBody");
  tbody.innerHTML = "";

  let models = data.all_models;

  /* fallback */
  if (!models || models.length === 0) {
    models = [
      {
        model_name: "Ensemble",
        prediction: data.prediction,
        label: data.label,
        proba_human: data.proba_human,
        proba_ai: data.proba_ai,
        confidence: data.confidence
      }
    ];
  }

  models.forEach(m => {
    const tr = document.createElement("tr");

    const cls = (m.prediction === 0) ? "tag-human" : "tag-ai";
    const isEns = (m.model_name || "").toUpperCase() === "ENSEMBLE";

    if (isEns) tr.className = "ensemble-row";

    tr.innerHTML = `
      <td>${isEns ? "🏆 " : ""}${m.model_name || "Model"}</td>
      <td class="${cls}">${m.label || "-"}</td>
      <td>${m.proba_human != null ? m.proba_human + "%" : "-"}</td>
      <td>${m.proba_ai != null ? m.proba_ai + "%" : "-"}</td>
      <td>
        ${m.confidence != null ? m.confidence + "%" : "-"}
        <span class="mini-bar-wrap">
          <span class="mini-bar" style="width:${m.confidence || 0}%${isEns ? ";background:#a371f7" : ""}"></span>
        </span>
      </td>
    `;

    tbody.appendChild(tr);
  });

  /* Model Used */
  document.getElementById("modelUsed").textContent =
    data.model_used || "Ensemble";
}