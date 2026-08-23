(() => {
  const $ = (sel) => document.querySelector(sel);

  const pageConfig = $("#page-config");
  const pageSearch = $("#page-search");
  const form = $("#config-form");
  const prepPanel = $("#prep-panel");
  const modelEl = $("#model");
  const datasetEl = $("#dataset");
  const splitEl = $("#split");
  const pctEl = $("#sample_pct");
  const pctOut = $("#sample_pct_out");
  const prepBar = $("#prep-bar");
  const prepMsg = $("#prep-msg");
  const prepEta = $("#prep-eta");
  const prepError = $("#prep-error");
  const btnNext = $("#btn-next");

  let config = null;
  let pollTimer = null;

  function fmtEta(sec) {
    if (sec == null || !Number.isFinite(sec)) return "estimating time…";
    if (sec <= 0) return "almost done";
    if (sec < 60) return `~${Math.ceil(sec)}s left`;
    const m = Math.floor(sec / 60);
    const s = Math.ceil(sec % 60);
    return `~${m}m ${s}s left`;
  }

  function showSearch() {
    pageConfig.classList.add("hidden");
    pageSearch.classList.remove("hidden");
    history.replaceState(null, "", "/search");
  }

  function showConfig() {
    pageSearch.classList.add("hidden");
    pageConfig.classList.remove("hidden");
    history.replaceState(null, "", "/");
  }

  function fillSplits() {
    const ds = config.datasets.find((d) => d.id === datasetEl.value);
    splitEl.innerHTML = "";
    (ds?.splits || []).forEach((s) => {
      const o = document.createElement("option");
      o.value = s;
      o.textContent = s;
      splitEl.appendChild(o);
    });
  }

  function renderExamples(list) {
    const box = $("#examples");
    box.innerHTML = "";
    list.forEach((q) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "chip";
      b.textContent = q;
      b.addEventListener("click", () => {
        $("#query").value = q;
        $("#query").focus();
      });
      box.appendChild(b);
    });
  }

  function fillSidebar(session) {
    $("#side-model").textContent = session.model_label || "—";
    $("#side-dataset").textContent = session.dataset_label || "—";
    $("#side-split").textContent = session.split || "—";
    $("#side-sample").textContent =
      session.sample_pct != null ? `${session.sample_pct}%` : "—";
    $("#side-seed").textContent =
      session.sample_seed != null ? String(session.sample_seed) : "—";
    $("#side-n").textContent = session.n_videos ?? "—";
  }

  async function loadConfig() {
    const res = await fetch("/api/config");
    config = await res.json();
    modelEl.innerHTML = config.models
      .map((m) => `<option value="${m.id}">${m.label}</option>`)
      .join("");
    datasetEl.innerHTML = config.datasets
      .map((d) => `<option value="${d.id}">${d.label}</option>`)
      .join("");
    pctEl.value = config.default_sample_pct;
    pctOut.textContent = `${config.default_sample_pct}%`;
    $("#sample_seed").value = config.default_sample_seed;
    $("#top_k").value = config.default_top_k;
    fillSplits();
    renderExamples(config.example_queries);
  }

  async function refreshSession() {
    const res = await fetch("/api/session");
    const session = await res.json();
    if (session.ready) {
      fillSidebar(session);
      if (location.pathname.startsWith("/search")) showSearch();
    }
    return session;
  }

  function stopPoll() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function startPoll() {
    stopPoll();
    pollTimer = setInterval(async () => {
      const res = await fetch("/api/preprocess/status");
      const job = await res.json();
      prepBar.style.width = `${Math.round((job.progress || 0) * 100)}%`;
      prepMsg.textContent = job.message || "Working…";
      prepEta.textContent = fmtEta(job.eta_sec);

      if (job.status === "done") {
        stopPoll();
        const session = await refreshSession();
        fillSidebar(session);
        showSearch();
      } else if (job.status === "error") {
        stopPoll();
        prepError.textContent = job.error || "Preprocessing failed";
        prepError.classList.remove("hidden");
        btnNext.disabled = false;
        form.classList.remove("hidden");
      }
    }, 800);
  }

  pctEl.addEventListener("input", () => {
    pctOut.textContent = `${pctEl.value}%`;
  });
  datasetEl.addEventListener("change", fillSplits);

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    prepError.classList.add("hidden");
    form.classList.add("hidden");
    prepPanel.classList.remove("hidden");
    btnNext.disabled = true;
    prepBar.style.width = "0%";
    prepMsg.textContent = "Starting…";
    prepEta.textContent = "estimating time…";

    const body = {
      model_id: modelEl.value,
      dataset_id: datasetEl.value,
      split: splitEl.value,
      sample_pct: Number(pctEl.value),
      sample_seed: Number($("#sample_seed").value) || 42,
    };
    const res = await fetch("/api/preprocess", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const t = await res.text();
      prepError.textContent = t;
      prepError.classList.remove("hidden");
      form.classList.remove("hidden");
      btnNext.disabled = false;
      return;
    }
    startPoll();
  });

  $("#btn-reconfig").addEventListener("click", () => {
    form.classList.remove("hidden");
    prepPanel.classList.add("hidden");
    btnNext.disabled = false;
    showConfig();
  });

  $("#search-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = $("#query").value.trim();
    if (!query) return;
    const top_k = Number($("#top_k").value) || 12;
    $("#search-status").textContent = "Searching…";
    $("#results").innerHTML = "";
    const res = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k }),
    });
    if (!res.ok) {
      $("#search-status").textContent = await res.text();
      return;
    }
    const data = await res.json();
    $("#search-status").textContent = `${data.results.length} results`;
    const box = $("#results");
    box.innerHTML = data.results
      .map(
        (r) => `
      <article class="card">
        <video
          src="${r.video_url}"
          muted
          playsinline
          loop
          preload="metadata"
        ></video>
        <div class="meta">
          <span class="score">${r.score.toFixed(3)}</span>
          <span class="vid">${r.video_id}</span>
          <div class="cap">${(r.caption || "").replace(/</g, "&lt;")}</div>
        </div>
      </article>`
      )
      .join("");

    box.querySelectorAll("video").forEach((v) => {
      const card = v.closest(".card");
      card.addEventListener("mouseenter", () => {
        v.play().catch(() => {});
      });
      card.addEventListener("mouseleave", () => {
        v.pause();
        v.currentTime = 0;
      });
    });
  });

  (async () => {
    await loadConfig();
    const session = await refreshSession();
    if (session.job?.status === "running") {
      form.classList.add("hidden");
      prepPanel.classList.remove("hidden");
      startPoll();
    } else if (session.ready && location.pathname.startsWith("/search")) {
      showSearch();
    }
  })();
})();
