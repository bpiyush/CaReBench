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
  let activeJobId = null;

  let lastResults = [];

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

      // Ignore stale status from a previous run.
      if (activeJobId != null && job.job_id !== activeJobId) {
        return;
      }

      prepBar.style.width = `${Math.round((job.progress || 0) * 100)}%`;
      prepMsg.textContent = job.message || "Working…";
      prepEta.textContent = fmtEta(job.eta_sec);

      if (job.status === "done" && job.job_id === activeJobId) {
        stopPoll();
        const session = await refreshSession();
        fillSidebar(session);
        showSearch();
      } else if (job.status === "error" && job.job_id === activeJobId) {
        stopPoll();
        prepError.textContent = job.error || "Preprocessing failed";
        prepError.classList.remove("hidden");
        btnNext.disabled = false;
        form.classList.remove("hidden");
        activeJobId = null;
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
      activeJobId = null;
      return;
    }
    const data = await res.json();
    activeJobId = data.job_id;
    startPoll();
  });

  $("#btn-reconfig").addEventListener("click", async () => {
    stopPoll();
    activeJobId = null;
    try {
      await fetch("/api/reset", { method: "POST" });
    } catch (_) {
      /* ignore */
    }
    form.classList.remove("hidden");
    prepPanel.classList.add("hidden");
    prepError.classList.add("hidden");
    btnNext.disabled = false;
    prepBar.style.width = "0%";
    $("#results").innerHTML = "";
    $("#search-status").textContent = "";
    lastResults = [];
    const rec = $("#btn-record");
    if (rec) {
      rec.disabled = true;
      rec.textContent = "Save GIF";
    }
    showConfig();
  });

  $("#search-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = $("#query").value.trim();
    if (!query) return;
    const top_k = Number($("#top_k").value) || 12;
    $("#search-status").textContent = "Searching…";
    $("#results").innerHTML = "";
    $("#btn-record").disabled = true;
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
    lastResults = data.results || [];
    $("#search-status").textContent = `${lastResults.length} results`;
    $("#btn-record").disabled = lastResults.length === 0;
    const box = $("#results");
    box.innerHTML = lastResults
      .map(
        (r) => `
      <article class="card">
        <video
          src="${r.video_url}"
          muted
          autoplay
          playsinline
          loop
          preload="auto"
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
      v.muted = true;
      const play = () => v.play().catch(() => {});
      if (v.readyState >= 2) play();
      else v.addEventListener("canplay", play, { once: true });
    });
  });

  $("#btn-record").addEventListener("click", async () => {
    const top = lastResults.slice(0, 9);
    if (!top.length) return;
    const btn = $("#btn-record");
    const prev = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Saving…";
    try {
      const resp = await fetch("/api/record", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: $("#query").value.trim(),
          video_ids: top.map((r) => r.video_id),
          scores: top.map((r) => r.score),
          captions: top.map((r) => r.caption || ""),
          duration: 4,
        }),
      });
      if (!resp.ok) {
        let msg = `HTTP ${resp.status}`;
        try {
          const err = await resp.json();
          msg = err.detail || JSON.stringify(err);
        } catch (_) {
          msg = await resp.text();
        }
        throw new Error(msg);
      }
      const blob = await resp.blob();
      const a = document.createElement("a");
      const disp = resp.headers.get("content-disposition") || "";
      const match = disp.match(/filename="?([^"]+)"?/);
      a.href = URL.createObjectURL(blob);
      a.download = match ? match[1] : "results.gif";
      a.click();
      URL.revokeObjectURL(a.href);
      btn.textContent = "Saved";
    } catch (err) {
      btn.textContent = "Failed";
      $("#search-status").textContent = String(err.message || err).slice(0, 220);
      console.error(err);
    }
    setTimeout(() => {
      btn.disabled = lastResults.length === 0;
      btn.textContent = prev;
    }, 1600);
  });

  (async () => {
    await loadConfig();
    const session = await refreshSession();
    if (session.job?.status === "running") {
      activeJobId = session.job.job_id;
      form.classList.add("hidden");
      prepPanel.classList.remove("hidden");
      startPoll();
    } else if (session.ready && location.pathname.startsWith("/search")) {
      showSearch();
    }
  })();
})();
