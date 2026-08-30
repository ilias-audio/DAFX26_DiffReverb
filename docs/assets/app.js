/* DAFX26_DiffReverb audio-example matrix.
 * Vanilla JS, no dependencies. Reads assets/manifest.json and renders a
 * room x method x instrument grid of players. A single shared <audio> element
 * plays one clip at a time (ideal for A/B), and switching method within the
 * same instrument row preserves playback position so you hear the same musical
 * moment through a different reverberator. */
(() => {
  "use strict";

  const ICON_PLAY = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
  const ICON_PAUSE = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>';

  const audio = new Audio();
  audio.preload = "none";

  let M = null;            // manifest
  let roomId = null;       // current room
  let showAbl = false;     // ablation toggle
  const state = { key: null, instr: null }; // currently loaded cell

  const $ = (sel, el = document) => el.querySelector(sel);
  const el = (tag, cls, html) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  };
  const fmt = (t) => {
    if (!isFinite(t)) return "0:00";
    const m = Math.floor(t / 60), s = Math.floor(t % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  };

  // ---- path helpers ----
  const audioPath = (room, method, instr) => {
    const file = instr === "ir" ? "ir.mp3" : `${instr}.mp3`;
    return `${M.paths.audio}/${room}/${method}/${file}`;
  };
  const cellKey = (room, method, instr) => `${room}|${method}|${instr}`;

  // ---- active method list for the current toggle ----
  // Headline view keeps the paper's order (Reference, Proposed, baselines).
  // Ablation view stable-sorts by group so column-group headers stay contiguous.
  const GROUP_ORDER = { reference: 0, proposed: 1, atten: 2, baseline: 3 };
  const activeMethods = () => {
    if (!showAbl) return M.methods.headline.slice();
    const all = M.methods.headline.concat(M.methods.ablation);
    return all
      .map((m, i) => [m, i])
      .sort((a, b) =>
        (GROUP_ORDER[a[0].group] - GROUP_ORDER[b[0].group]) || (a[1] - b[1]))
      .map((x) => x[0]);
  };

  // =====================================================================
  // Rendering
  // =====================================================================
  function renderRoomPills() {
    const host = $("#roompills");
    host.innerHTML = "";
    M.rooms.forEach((r) => {
      const p = el("button", "pill" + (r.id === roomId ? " active" : ""));
      p.innerHTML = `${r.name}${r.t30 ? `<span class="t30">${r.t30}s</span>` : ""}`;
      p.onclick = () => {
        roomId = r.id; stop(); renderRoomPills(); renderMatrix(); renderSpectra();
      };
      host.appendChild(p);
    });
  }

  // Per-room spectrum comparison (target vs proposed FDN + difference), as both
  // 1/6-octave magnitude spectra and full time-frequency-energy spectrograms.
  // Images are pre-rendered by build_site.py --mode figures; here we just point
  // each <img> at the current room's files.
  function renderSpectra() {
    if (!M || !M.paths || !M.paths.spectra || !roomId) return;
    const base = `${M.paths.spectra}/${roomId}`;
    const set = (id, file, label) => {
      const img = document.getElementById(id);
      if (!img) return;
      img.src = `${base}/${file}`;
      img.alt = `${label}, ${roomId}`;
    };
    // magnitude spectra
    set("spec-target", "target.png", "Target RIR magnitude spectrum");
    set("spec-proposed", "proposed.png", "Proposed FDN magnitude spectrum");
    set("spec-diff", "diff.png", "Proposed minus target spectral difference");
    // time-frequency-energy spectrograms
    set("spec-target-stft", "target_spec.png", "Target RIR spectrogram");
    set("spec-proposed-stft", "proposed_spec.png", "Proposed FDN spectrogram");
    set("spec-diff-stft", "diff_spec.png", "Proposed minus target spectrogram difference");
    // early-time waveform (target vs ER vs late branches)
    set("spec-early", "early_time.png", "Early-time waveform: target, early reflections, late reverb");
    const rn = M.rooms.find((r) => r.id === roomId);
    const cap = $("#spec-room");
    if (cap && rn) cap.textContent = rn.name;
  }

  function renderMatrix() {
    const methods = activeMethods();
    const avail = new Set(M.available[roomId] || []);
    const gLabel = M.groups;

    const scroll = $("#matrix");
    scroll.innerHTML = "";
    const table = el("table", "matrix");

    // <colgroup> to subtly highlight the proposed and recommended columns
    const colg = el("colgroup");
    colg.appendChild(el("col")); // rowhead
    methods.forEach((m) => {
      const cls = m.id === "FDN_DiffER_PEQ10" ? "col-hl" : (m.recommended ? "col-rec" : "");
      colg.appendChild(el("col", cls));
    });
    table.appendChild(colg);

    const thead = el("thead");

    // group header row (spans consecutive same-group methods)
    const grow = el("tr", "grouprow");
    grow.appendChild(el("th", "rowhead-corner", ""));
    let i = 0;
    while (i < methods.length) {
      let j = i;
      while (j + 1 < methods.length && methods[j + 1].group === methods[i].group) j++;
      const th = el("th", "g-" + methods[i].group, gLabel[methods[i].group] || methods[i].group);
      th.colSpan = j - i + 1;
      grow.appendChild(th);
      i = j + 1;
    }
    thead.appendChild(grow);

    // method header row
    const mrow = el("tr", "methrow");
    const corner = el("th", "rowhead-corner");
    corner.innerHTML = `<span class="rname">${M.rooms.find((r) => r.id === roomId).name}</span>`;
    mrow.appendChild(corner);
    methods.forEach((m) => {
      const th = el("th");
      th.title = m.desc || "";
      th.innerHTML = m.name + (m.recommended ? '<span class="mrec">recommended</span>' : "");
      mrow.appendChild(th);
    });
    thead.appendChild(mrow);
    table.appendChild(thead);

    // body: one row per instrument
    const tbody = el("tbody");
    M.instruments.forEach((inst) => {
      const tr = el("tr");
      const rh = el("td", "rowhead");
      rh.innerHTML =
        `<div class="rname">${inst.name}</div>` +
        (inst.id === "ir"
          ? `<div class="rsub">impulse response</div>`
          : `<div class="rsub">dry input convolved</div>`);
      if (inst.dry) {
        const dry = el("div", "dry");
        const b = miniPlay(`${inst.dry}`, `dry|${inst.id}`, inst.id, inst.name, "dry input");
        b.title = "Play the dry (anechoic) input";
        dry.appendChild(b);
        rh.appendChild(dry);
      }
      tr.appendChild(rh);

      methods.forEach((m) => {
        const td = el("td", "cell");
        td.dataset.key = cellKey(roomId, m.id, inst.id);
        if (!avail.has(m.id)) {
          td.className = "cell empty";
          td.textContent = "n/a";
        } else {
          const src = audioPath(roomId, m.id, inst.id);
          const btn = miniPlay(src, cellKey(roomId, m.id, inst.id), inst.id,
            m.name, inst.name);
          td.appendChild(btn);
          // Raw-IR WAV/MP3 downloads are no longer per-cell; the lossless WAVs
          // live in the repo's ir_wav/ folder (linked from the page text).
        }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    scroll.appendChild(table);

    // restore playing highlight if still visible
    refreshPlayingUI();
  }

  // a single circular play button bound to (src, key, instr)
  function miniPlay(src, key, instr, title, sub) {
    const b = el("button", "play");
    b.innerHTML = ICON_PLAY;
    b.dataset.key = key;
    b.setAttribute("aria-label", "Play " + title + (sub ? " " + sub : ""));
    b.onclick = () => toggle(src, key, instr, title, sub);
    return b;
  }

  // =====================================================================
  // Playback
  // =====================================================================
  function toggle(src, key, instr, title, sub) {
    if (state.key === key && !audio.paused) { audio.pause(); return; }
    if (state.key === key && audio.paused && audio.src) { audio.play(); return; }

    // preserve position when switching within the same instrument row
    let t = 0;
    if (state.instr === instr && isFinite(audio.currentTime)) t = audio.currentTime;

    state.key = key; state.instr = instr;
    nowMeta(title, sub);
    audio.src = src;
    const onMeta = () => {
      audio.currentTime = Math.min(t, (audio.duration || 0) - 0.02) || 0;
      audio.play().catch(() => {});
      audio.removeEventListener("loadedmetadata", onMeta);
    };
    audio.addEventListener("loadedmetadata", onMeta);
    audio.load();
    refreshPlayingUI();
  }

  function stop() {
    audio.pause();
    state.key = null; state.instr = null;
    refreshPlayingUI();
    $("#nowbar").classList.remove("show");
  }

  function refreshPlayingUI() {
    const playing = state.key && !audio.paused;
    document.querySelectorAll(".play").forEach((b) => {
      const on = b.dataset.key === state.key && playing;
      b.classList.toggle("playing", on);
      b.innerHTML = on ? ICON_PAUSE : ICON_PLAY;
    });
    document.querySelectorAll("td.cell").forEach((td) => {
      td.classList.toggle("playing", td.dataset.key === state.key && !!playing);
    });
    const np = $("#np-btn");
    if (np) np.innerHTML = playing ? ICON_PAUSE : ICON_PLAY;
  }

  function nowMeta(title, sub) {
    const bar = $("#nowbar");
    bar.classList.add("show");
    $("#np-title").textContent = title || "";
    const room = M.rooms.find((r) => r.id === roomId)?.name || "";
    $("#np-sub").textContent = sub ? `${sub}, ${room}` : room;
  }

  // =====================================================================
  // Wiring
  // =====================================================================
  function wireNowBar() {
    $("#np-btn").onclick = () => { if (audio.paused) audio.play(); else audio.pause(); };
    $("#np-x").onclick = stop;
    const seek = $("#np-seek");
    seek.onclick = (e) => {
      const r = seek.getBoundingClientRect();
      if (audio.duration) audio.currentTime = ((e.clientX - r.left) / r.width) * audio.duration;
    };
    audio.addEventListener("timeupdate", () => {
      const pct = audio.duration ? (audio.currentTime / audio.duration) * 100 : 0;
      $("#np-fill").style.width = pct + "%";
      $("#np-time").textContent = `${fmt(audio.currentTime)} / ${fmt(audio.duration)}`;
    });
    audio.addEventListener("play", refreshPlayingUI);
    audio.addEventListener("pause", refreshPlayingUI);
    audio.addEventListener("ended", refreshPlayingUI);
  }

  async function main() {
    wireNowBar();
    try {
      const res = await fetch("assets/manifest.json", { cache: "no-cache" });
      M = await res.json();
    } catch (e) {
      $("#matrix").innerHTML =
        '<p style="padding:20px;color:#f88">Could not load assets/manifest.json. ' +
        "If viewing locally, serve the folder over HTTP " +
        "(e.g. <code>python3 -m http.server</code>).</p>";
      return;
    }
    const prefer = M.rooms.find((r) => /lecture/i.test(r.id) || /lecture/i.test(r.name));
    roomId = (prefer || M.rooms[0]).id;
    const tgl = $("#ablToggle");
    tgl.checked = showAbl;
    tgl.onchange = () => { showAbl = tgl.checked; renderMatrix(); };
    const ablLabel = $("#ablLabel");
    if (ablLabel) ablLabel.textContent =
      `Show ablation variants (+${M.methods.ablation.length} columns)`;
    renderRoomPills();
    renderMatrix();
    renderSpectra();
  }

  document.addEventListener("DOMContentLoaded", main);
})();
