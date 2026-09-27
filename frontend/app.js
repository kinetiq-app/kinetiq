// app.js — Kinetiq v4 live camera PWA.
//
// Pipeline (privacy invariant: pixels NEVER leave this device):
//   webcam ─▶ MediaPipe PoseLandmarker (in-browser) ─▶ 33 keypoints
//          ─▶ frame {t_ms, pose_model, people:[{track_id, kp:[[x,y,z,vis]×33]}]}
//          ─▶ POST /prototype/assess (only keypoints cross the wire)
//          ─▶ render rep_count / phase / flags / coaching_cue / subject_lock_ok
//
// The frame shape and endpoint are read from the REAL backend
// (evals/gate0/golden_loader.py validate_frame_schema + prototype_api/main.py + schemas.py),
// not assumed — see docs/EVAL_HARNESS_STAGE0_SPEC.md §5.

import {
  PoseLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { createSetTracker, boundQueue } from "./segments.js";

const CFG = window.KINETIQ_CONFIG;
const urlParams = typeof window !== "undefined" && window.location ? new URLSearchParams(window.location.search) : null;
const queryApi = urlParams ? urlParams.get("api") : null;
const isLocalHost = typeof window !== "undefined" && window.location && (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1");
let resolvedApi = queryApi || (CFG && CFG.API_BASE_URL) || "";
if (!queryApi && typeof window !== "undefined" && window.location) {
  const host = window.location.hostname;
  // If hosted via Cloudflare Pages (*.pages.dev), tunnel, or local host, use same-origin!
  if (
    host.endsWith("pages.dev") ||
    host.includes("trycloudflare") ||
    host === "localhost" ||
    host === "127.0.0.1"
  ) {
    resolvedApi = window.location.origin;
  }
}
const API = resolvedApi.replace(/\/+$/, "");

// ---- DOM ----
const $ = (id) => document.getElementById(id);
const screens = {
  picker: $("screen-picker"),
  live: $("screen-live"),
  summary: $("screen-summary"),
};
const video = $("video");
const overlay = $("overlay");
const octx = overlay.getContext("2d");

// ---- session state ----
let landmarker = null;
let severities = {};
let stream = null;
let running = false;
let exercise = null;
let sessionId = null;
let firstPost = true;
let frameBuffer = []; // frames accumulated since last successful POST
let lastResponse = null;
let lastVideoTs = -1;
let postTimer = null;

// ---- connection resilience -------------------------------------------------
// Render's free tier puts a web service to sleep after ~15 min idle, and the
// next request pays a 30-60s cold start. That lands on the FIRST request of a
// session -- exactly when someone is standing in front of the camera. Without
// this, one failed POST during wake-up dumped the user on a blocking error.
//
// Two defences: wake the server before the set starts, and treat early POST
// failures as "not awake yet" (retry with backoff, keep recording) rather than
// as a dead end. Frames are never dropped -- they re-queue and replay, so the
// rep count catches up once the server answers.
let apiWarm = false;         // has /health answered since page load?
let sessionMaxFrames = null; // server's per-session frame cap, from /health (null = unknown)
let tracker = null;          // keeps one visible set continuous across server sessions
let droppedFrames = 0;       // frames shed from an over-long outage queue (see boundQueue)
let flushInFlight = false;   // a POST is outstanding -- don't stack another on it
let failStreak = 0;          // consecutive failed flushes
let nextAttemptAt = 0;       // backoff gate, performance.now() ms
const HEALTH_TIMEOUT_MS = 12000;   // one /health probe
const WAKE_TIMEOUT_MS = 90000;     // total budget for waking a sleeping instance
const HARD_FAIL_AFTER = 10;        // give up quietly retrying, ask the user
const BACKOFF_MAX_MS = 8000;

// ---------------------------------------------------------------------------
// screens
function show(name) {
  for (const s of Object.values(screens)) s.classList.remove("active");
  screens[name].classList.add("active");
}
function setState(title, msg, actionLabel, actionFn, allowDismiss = false) {
  $("state-title").textContent = title;
  $("state-msg").textContent = msg || "";
  const btn = $("state-action");
  if (actionLabel) {
    btn.textContent = actionLabel;
    btn.hidden = false;
    btn.style.display = "";
    btn.onclick = actionFn;
  } else {
    btn.hidden = true;
    btn.style.display = "none";
  }
  const dismissBtn = $("state-dismiss");
  if (dismissBtn) {
    if (allowDismiss || running) {
      dismissBtn.hidden = false;
      dismissBtn.style.display = "";
      dismissBtn.onclick = () => clearState();
    } else {
      dismissBtn.hidden = true;
      dismissBtn.style.display = "none";
    }
  }
  const el = $("state-overlay");
  el.hidden = false;
  el.style.display = "flex";
}
function clearState() {
  const el = $("state-overlay");
  el.hidden = true;
  el.style.display = "none";
  const dismissBtn = $("state-dismiss");
  if (dismissBtn) {
    dismissBtn.hidden = true;
    dismissBtn.style.display = "none";
  }
}

// ---------------------------------------------------------------------------
// one-time load of the pose model + the derived severity table
async function loadModel() {
  if (landmarker) return;
  const files = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  landmarker = await PoseLandmarker.createFromOptions(files, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1, // single subject in-browser; subject-lock is exercised in the offline bake-off
  });
}
async function loadSeverities() {
  try {
    severities = await fetch("severities.json").then((r) => r.json());
  } catch {
    severities = {}; // coloring degrades gracefully to "med" if this is missing
  }
}

// ---------------------------------------------------------------------------
// camera & permissions
async function getCameraPermissionState() {
  if (navigator.permissions && navigator.permissions.query) {
    try {
      const p = await navigator.permissions.query({ name: "camera" });
      return p.state; // 'granted', 'prompt', 'denied'
    } catch {
      return "prompt";
    }
  }
  return "prompt";
}

let streamIdleTimer = null;
function scheduleCameraRelease() {
  clearTimeout(streamIdleTimer);
  streamIdleTimer = setTimeout(() => {
    if (!running) stopCamera(true);
  }, 45000);
}

function syncOverlayDimensions() {
  if (video.videoWidth > 0 && video.videoHeight > 0) {
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
  } else {
    video.onloadedmetadata = () => {
      overlay.width = video.videoWidth || 640;
      overlay.height = video.videoHeight || 480;
    };
  }
}

async function startCamera() {
  // If active stream already exists with live tracks, reuse it instantly!
  if (stream && stream.active && stream.getVideoTracks().some((t) => t.readyState === "live")) {
    if (video.srcObject !== stream) {
      video.srcObject = stream;
    }
    try {
      await video.play();
    } catch {}
    syncOverlayDimensions();
    return;
  }

  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  syncOverlayDimensions();
}

function stopCamera(releaseTracks = false) {
  if (video) {
    try {
      video.pause();
    } catch {}
  }
  if (releaseTracks && stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
    if (video) video.srcObject = null;
  }
}

// ---------------------------------------------------------------------------
// build a backend-shaped frame from a MediaPipe result
function buildFrame(landmarks, tMs) {
  // MediaPipe normalized landmarks: {x, y, z, visibility}, 33 of them, BlazePose order.
  const kp = landmarks.map((p) => [
    +p.x.toFixed(5),
    +p.y.toFixed(5),
    p.z == null ? 0 : +p.z.toFixed(5),
    +(p.visibility ?? 0).toFixed(4),
  ]);
  return {
    t_ms: Math.round(tMs),
    pose_model: CFG.POSE_MODEL,
    people: [{ track_id: 0, kp, box: bboxOf(kp) }],
  };
}

// The detector's subject-lock picks its subject by bounding-box area
// (detector/subject_lock.py -> geometry.bbox_area), so `box` is REQUIRED even
// with a single person in frame. validate_frame_schema does not check for it,
// so omitting it passed validation and then raised KeyError('box') deep in the
// detector -- surfacing in the browser as a bare 500 with no CORS header, i.e.
// "Failed to fetch". MediaPipe gives landmarks but no box, so we derive one.
//
// Format is [x, y, w, h], normalized, matching the golden fixtures.
function bboxOf(kp) {
  let x0 = 1, y0 = 1, x1 = 0, y1 = 0, seen = 0;
  for (const [x, y, , vis] of kp) {
    if (vis < 0.3) continue; // ignore landmarks MediaPipe is guessing at
    seen++;
    if (x < x0) x0 = x;
    if (y < y0) y0 = y;
    if (x > x1) x1 = x;
    if (y > y1) y1 = y;
  }
  if (seen === 0) return [0, 0, 0, 0]; // nobody visible; subject-lock treats it as no subject
  const clamp = (v) => Math.max(0, Math.min(1, v));
  x0 = clamp(x0); y0 = clamp(y0); x1 = clamp(x1); y1 = clamp(y1);
  return [
    +x0.toFixed(5),
    +y0.toFixed(5),
    +(x1 - x0).toFixed(5),
    +(y1 - y0).toFixed(5),
  ];
}

// ---------------------------------------------------------------------------
// detection loop
function loop() {
  if (!running) return;
  // Failsafe auto-dismiss: if video is actively playing frames, never keep a "Camera" or "Warming up" overlay visible
  if (video.currentTime > 0) {
    const title = $("state-title").textContent;
    if (title === "Camera" || title === "Warming up") {
      clearState();
    }
  }
  const now = performance.now();
  if (video.currentTime !== lastVideoTs) {
    lastVideoTs = video.currentTime;
    const result = landmarker.detectForVideo(video, now);
    if (result.landmarks && result.landmarks.length > 0) {
      frameBuffer.push(buildFrame(result.landmarks[0], now));
      drawSkeleton(result.landmarks[0]);
    } else {
      octx.clearRect(0, 0, overlay.width, overlay.height);
    }
  }
  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------------------
// non-blocking connection notice (the set keeps running underneath it)
function banner(msg) {
  const el = $("conn-banner");
  if (!el) return;
  if (msg) {
    el.textContent = msg;
    el.hidden = false;
  } else {
    el.hidden = true;
  }
}

// A cheap GET that tells us whether the instance is awake. Uses AbortController
// because a sleeping Render instance holds the connection open rather than
// refusing it -- without a timeout this would hang instead of retrying.
async function pingHealth(timeoutMs = HEALTH_TIMEOUT_MS) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${API}/health`, { signal: ctrl.signal, cache: "no-store" });
    if (!res.ok) return false;
    try {
      const h = await res.json();
      if (h && h.status === "ok") {
        if (Number.isFinite(h.session_max_frames)) sessionMaxFrames = h.session_max_frames;
        return true;
      }
    } catch {
      // Fall through to res.ok if json parsing fails
    }
    return res.ok;
  } catch (err) {
    console.warn("[kinetiq] pingHealth probe failed:", err && err.message);
    return false;
  } finally {
    clearTimeout(t);
  }
}

// Poll /health until the instance answers or the budget runs out.
async function wakeApi(onProgress) {
  if (apiWarm) return true;
  const deadline = performance.now() + WAKE_TIMEOUT_MS;
  let attempt = 0;
  while (performance.now() < deadline) {
    attempt++;
    if (onProgress) {
      const secs = Math.round((performance.now() - (deadline - WAKE_TIMEOUT_MS)) / 1000);
      onProgress(
        attempt === 1
          ? "Waking the server — this can take up to a minute on the free plan."
          : `Still waking… ${secs}s. Your reps are being recorded either way.`
      );
    }
    if (await pingHealth()) {
      apiWarm = true;
      return true;
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
  return false;
}

// ---------------------------------------------------------------------------
// server sessions: one visible set may span several (see segments.js)
function newSessionId() {
  return `sess-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}
function rollSession() {
  tracker.roll();
  sessionId = newSessionId();
  firstPost = true; // reset:true opens the fresh server session cleanly
}

// ---------------------------------------------------------------------------
// POST buffered frames to the detector API
async function flush({ final = false } = {}) {
  // `final` lets stopSet deliver the last frames after `running` is already false; without it
  // the "final flush" returned immediately and the tail of every set was never scored.
  if (!running && !final) return;
  if (frameBuffer.length === 0) return;

  // A cold start can take 30-60s while the post timer keeps firing every 400ms.
  // Without this guard we would stack dozens of concurrent POSTs onto a server
  // that is still booting, and each would carry a different slice of the buffer.
  if (flushInFlight) return;
  if (performance.now() < nextAttemptAt) return;

  // Never let an outage grow the queue past one server session's worth: beyond that every
  // retry is a bigger body than the last, and those frames could never be scored anyway.
  const bounded = boundQueue(frameBuffer, sessionMaxFrames);
  frameBuffer = bounded.frames;
  droppedFrames += bounded.dropped;

  // Roll to a fresh server session BEFORE this one fills (between reps where possible),
  // instead of discovering the cap as a 413 that no retry can get past.
  if (tracker.shouldRoll(frameBuffer.length)) rollSession();

  // Send only what the current session can still accept; the rest waits for the next flush.
  const take = Math.min(frameBuffer.length, tracker.capacity());
  if (take <= 0) {
    rollSession();
    return;
  }
  const frames = frameBuffer.slice(0, take);
  frameBuffer = frameBuffer.slice(take);
  flushInFlight = true;
  const body = {
    session_id: sessionId,
    exercise_id: exercise,
    frames,
    reset: firstPost,
  };
  try {
    const res = await fetch(`${API}/prototype/assess`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const txt = await res.text();
      const e = new Error(`API ${res.status}: ${txt.slice(0, 140)}`);
      e.status = res.status;
      throw e;
    }
    firstPost = false;
    lastResponse = await res.json();
    tracker.acknowledge(frames.length, lastResponse);
    render(lastResponse);

    // Recovered: drop the retry state and any notice we were showing.
    apiWarm = true;
    failStreak = 0;
    nextAttemptAt = 0;
    banner(null);
    clearState();
  } catch (err) {
    // Re-queue the frames we pulled so nothing is silently lost (the whole
    // project's discipline). They replay on the next successful POST and the
    // rep count catches up.
    frameBuffer = frames.concat(frameBuffer);

    // Log the REAL reason to the console. The postmortem's hardest lesson: a
    // failure that shows only as "Failed to fetch" on screen, with the actual
    // status/body buried in the Network tab, cost days. `err.message` already
    // carries "API <status>: <body>" for a non-2xx; for a true network failure
    // it's the fetch error. Either way it's now in the console, not just the UI.
    console.error("[kinetiq] assess failed:", err && (err.status || "network"), "-", err && err.message);

    if (err && err.status === 413) {
      // The server session is full (e.g. the cap was unknown, or a stale session). Nothing
      // is wrong with the connection: open a fresh session and resend on the next tick. The
      // server appends nothing on a 413, so no frame is double-counted.
      rollSession();
      return;
    }

    failStreak++;

    if (failStreak < HARD_FAIL_AFTER) {
      // Probably a cold start, not a dead server. Back off and keep recording
      // instead of blocking the user on an error screen.
      const delay = Math.min(1000 * 2 ** (failStreak - 1), BACKOFF_MAX_MS);
      nextAttemptAt = performance.now() + delay;
      banner(
        apiWarm
          ? "Lost the coach — still recording, will catch up."
          : "Waking the server — still recording, your reps will catch up."
      );
    } else {
      // Sustained failure: now it is worth interrupting.
      banner(null);
      setState(
        "Can't reach the trainer",
        String(err.message || err),
        "Retry",
        async () => {
          failStreak = 0;
          nextAttemptAt = 0;
          clearState();
          banner("Reconnecting…");
          const ok = await wakeApi((m) => banner(m));
          banner(ok ? null : "Still no answer — check your connection.");
        }
      );
    }
  } finally {
    flushInFlight = false;
  }
}

// ---------------------------------------------------------------------------
// render a response
function render(r) {
  $("rep-count").textContent = tracker ? tracker.totalReps() : r.rep_count;
  $("phase").textContent = r.phase || "—";

  const lock = $("hud-lock");
  lock.className = "lock-pill " + (r.subject_lock_ok ? "lock-ok" : "lock-lost");
  lock.textContent = r.subject_lock_ok ? "locked on you" : "can't see you";

  const cueEl = $("cue");
  if (r.coaching_cue) {
    cueEl.textContent = r.coaching_cue;
    cueEl.hidden = false;
  } else {
    cueEl.hidden = true;
  }

  const flagsEl = $("flags");
  flagsEl.innerHTML = "";
  for (const f of r.current_flags || []) {
    const sev = (severities[exercise] && severities[exercise][f]) || "med";
    const el = document.createElement("span");
    el.className = "flag " + sev;
    el.textContent = prettyFlag(f);
    flagsEl.appendChild(el);
  }
}
function prettyFlag(f) {
  return f.replace(/_/g, " ");
}

// ---------------------------------------------------------------------------
// skeleton overlay (cosmetic — helps a user frame themselves)
const CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [24, 26], [26, 28],
];
function drawSkeleton(lm) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  octx.lineWidth = 3;
  octx.strokeStyle = "rgba(16,185,129,0.9)";
  octx.fillStyle = "rgba(16,185,129,0.9)";
  for (const [a, b] of CONNECTIONS) {
    if (!lm[a] || !lm[b]) continue;
    octx.beginPath();
    octx.moveTo(lm[a].x * overlay.width, lm[a].y * overlay.height);
    octx.lineTo(lm[b].x * overlay.width, lm[b].y * overlay.height);
    octx.stroke();
  }
  for (const p of lm) {
    octx.beginPath();
    octx.arc(p.x * overlay.width, p.y * overlay.height, 4, 0, Math.PI * 2);
    octx.fill();
  }
}

// ---------------------------------------------------------------------------
let isStartingSet = false;

// ---------------------------------------------------------------------------
// start / stop a set
async function startSet(ex) {
  if (isStartingSet) return;
  isStartingSet = true;

  try {
    clearTimeout(streamIdleTimer);
    exercise = ex;
    sessionId = newSessionId();
    firstPost = true;
    droppedFrames = 0;
    tracker = createSetTracker(() => sessionMaxFrames, {
      rollAt: CFG.SESSION_ROLL_AT,
      forceRollAt: CFG.SESSION_FORCE_ROLL_AT,
    });
    frameBuffer = [];
    lastResponse = null;
    lastVideoTs = -1;
    failStreak = 0;
    nextAttemptAt = 0;
    flushInFlight = false;
    banner(null);
    clearState();

    $("hud-exercise").textContent = ex;
    $("rep-count").textContent = "0";
    $("phase").textContent = "—";
    $("cue").hidden = true;
    $("flags").innerHTML = "";
    show("live");

    // 1. Pose model: only show warmup overlay if landmarker is not yet in memory
    if (!landmarker) {
      setState("Warming up", "Loading the on-device pose model…");
      await loadModel();
      clearState();
    }

    // 2. Camera setup: only display camera permission prompt if permission is not yet granted
    const isStreamActive = stream && stream.active && stream.getVideoTracks().some((t) => t.readyState === "live");
    if (!isStreamActive) {
      const permState = await getCameraPermissionState();
      if (permState !== "granted") {
        setState("Camera", "Allow camera access to begin. Video stays on your device.", null, null, true);
      }
      await startCamera();
      clearState();
    } else {
      await startCamera();
      clearState();
    }

    // Wake the detector in the background if not already awake.
    // Use the non-blocking banner strip so the camera view and skeleton remain fully visible!
    if (!apiWarm) {
      banner("Waking the server — your reps are being recorded and will catch up.");
      wakeApi((m) => banner(m)).then((awake) => {
        if (awake) banner(null);
      }).catch(() => {});
    }

    running = true;
    requestAnimationFrame(loop);
    if (postTimer) clearInterval(postTimer);
    postTimer = setInterval(flush, CFG.POST_INTERVAL_MS);
  } catch (err) {
    if (err && (err.name === "NotAllowedError" || err.name === "SecurityError")) {
      setState(
        "Camera blocked",
        "Kinetiq needs the camera to see your form. Enable it in your browser settings, then retry.",
        "Retry",
        () => startSet(ex)
      );
    } else {
      setState("Couldn't start", String(err.message || err), "Back", backToPicker);
    }
  } finally {
    isStartingSet = false;
  }
}

async function stopSet() {
  running = false;
  if (postTimer) {
    clearInterval(postTimer);
    postTimer = null;
  }
  clearState();
  // Wait out any POST already in flight, then deliver whatever is still queued.
  for (let i = 0; i < 50 && flushInFlight; i++) await new Promise((r) => setTimeout(r, 100));
  await flush({ final: true }); // final flush so the last reps are counted
  stopCamera(false); // keep stream warm for next exercise, pause video
  scheduleCameraRelease();
  octx.clearRect(0, 0, overlay.width, overlay.height);
  renderSummary(lastResponse);
  show("summary");
}

function backToPicker() {
  running = false;
  if (postTimer) {
    clearInterval(postTimer);
    postTimer = null;
  }
  stopCamera(false);
  scheduleCameraRelease();
  clearState();
  banner(null);
  show("picker");
}

// ---------------------------------------------------------------------------
// summary from the accumulated reps[]
function renderSummary(r) {
  // A long set may have spanned several server sessions; the tracker holds all of them.
  const reps = tracker ? tracker.allReps() : (r && r.reps) || [];
  const total = tracker ? tracker.totalReps() : r ? r.rep_count : 0;
  $("sum-reps").textContent = total;

  const flagged = reps.filter((x) => x.flags && x.flags.length > 0);
  const clean = reps.length - flagged.length;
  $("sum-clean").textContent = reps.length
    ? `${clean} clean · ${flagged.length} flagged`
    : "No completed reps detected.";

  // tally flags across the set
  const tally = {};
  for (const x of flagged) for (const f of x.flags) tally[f] = (tally[f] || 0) + 1;
  const list = $("sum-flags");
  list.innerHTML = "";
  for (const [f, n] of Object.entries(tally)) {
    const sev = (severities[exercise] && severities[exercise][f]) || "med";
    const el = document.createElement("span");
    el.className = "flag " + sev;
    el.textContent = `${prettyFlag(f)} ×${n}`;
    list.appendChild(el);
  }
  let note = reps.length
    ? "Keypoints only were sent to the detector — no video left your device."
    : "Try again — make sure your whole body is in frame.";
  // Say so if a long outage forced us to shed frames; a quietly low count would be dishonest.
  if (droppedFrames > 0) {
    note += ` Connection dropped out for a while, so ~${Math.round(droppedFrames / 30)}s of` +
      " movement couldn't be scored — the count may be low.";
  }
  $("sum-cue").textContent = note;
}

// ---------------------------------------------------------------------------
// wire up
document.querySelectorAll(".exercise-card").forEach((btn) => {
  btn.addEventListener("click", () => startSet(btn.dataset.exercise));
});
$("btn-stop").addEventListener("click", stopSet);
$("btn-back").addEventListener("click", backToPicker);
$("btn-again").addEventListener("click", () => {
  clearState();
  banner(null);
  show("picker");
});

window.addEventListener("pagehide", () => stopCamera(true));

loadSeverities();

// Pre-warm the detector the moment the app opens. By the time someone has read
// the picker and chosen an exercise, a sleeping instance has usually finished
// booting -- which turns the most common cold start into no wait at all.
wakeApi().catch(() => {});

// register service worker (offline shell; pose model + API still need network)
//
// update() is forced on every load, and a new worker that takes control triggers
// exactly one reload. Without this, a browser that already had the old
// cache-first worker could keep serving a stale app.js/config.js indefinitely --
// which is precisely how a deployed API URL fix stayed invisible for days.
if ("serviceWorker" in navigator) {
  let reloadedForUpdate = false;
  navigator.serviceWorker.addEventListener("controllerchange", () => {
    if (reloadedForUpdate || running || isStartingSet) return;
    reloadedForUpdate = true;
    window.location.reload();
  });
  navigator.serviceWorker
    .register("sw.js")
    .then((reg) => reg.update().catch(() => {}))
    .catch(() => {});
}
