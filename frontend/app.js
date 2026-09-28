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
    host.endsWith("workers.dev") ||
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

// ---- positioning & framing state ----
let sessionPhase = "positioning"; // "positioning" | "countdown" | "active"
let steadyFrameCount = 0;
let countdownTimer = null;
let countdownVal = 3;
let smoothedLandmarks = null;
let audioCtx = null;

// ---- plank hold tracking ----
let isHoldExercise = false;
let holdStartTime = null;
let holdElapsedMs = 0;
let holdGoodMs = 0;
let holdWarnMs = 0;
let holdLastTickMs = null;
let plankHoldActive = false;
let plankBreakFrames = 0;
let plankDominantFaults = {};

// Bicep Curl on-device tracking state (supports front & side view rep counting)
let bicepCurlReps = 0;
let bicepCurlArms = {
  left: { phase: "ready", restElbow: null, peakDisplacement: 0, repStartMs: 0 },
  right: { phase: "ready", restElbow: null, peakDisplacement: 0, repStartMs: 0 },
};
let lastCurlRepMs = 0;

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
  // When a stream is reused (srcObject re-assigned), loadedmetadata does NOT
  // re-fire, so the old event-listener approach silently leaves the canvas at
  // its previous dimensions (or 0×0 on first load). Poll via rAF until ready.
  function applyDimensions() {
    if (video.videoWidth > 0 && video.videoHeight > 0) {
      overlay.width  = video.videoWidth;
      overlay.height = video.videoHeight;
    } else {
      requestAnimationFrame(applyDimensions);
    }
  }
  applyDimensions();
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
// audio tone helper (Web Audio API - synthetic, zero external assets)
function playBeep(freq = 440, duration = 0.12) {
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) return;
    if (!audioCtx) audioCtx = new AudioContextClass();
    if (audioCtx.state === "suspended") audioCtx.resume();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    gain.gain.setValueAtTime(0.18, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
  } catch {}
}

// ---------------------------------------------------------------------------
// 2D angle calculation helper (angle at vertex b in degrees)
function calcAngleDeg(a, b, c) {
  const rad = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let deg = Math.abs((rad * 180.0) / Math.PI);
  if (deg > 180.0) deg = 360.0 - deg;
  return deg;
}

// Format milliseconds into MM:SS or Xs for clean display
function formatHoldTime(ms) {
  const totalSecs = Math.floor(ms / 1000);
  const mins = Math.floor(totalSecs / 60);
  const secs = totalSecs % 60;
  return mins > 0 ? `${mins}:${secs.toString().padStart(2, "0")}` : `${secs}s`;
}

// ---------------------------------------------------------------------------
// Real-time Plank (hold) posture evaluator
// Analyzes vector alignment between Shoulder, Hip, and Ankle.
function evaluatePlankPosture(lm) {
  if (!lm || lm.length < 33) return { state: "broken", angle: 0, cue: "Can't see body" };

  const ls = lm[11], rs = lm[12]; // shoulders
  const lh = lm[23], rh = lm[24]; // hips
  const la = lm[27], ra = lm[28]; // ankles

  // Choose side with highest confidence
  const leftVis = (ls.visibility ?? 0) + (lh.visibility ?? 0) + (la.visibility ?? 0);
  const rightVis = (rs.visibility ?? 0) + (rh.visibility ?? 0) + (ra.visibility ?? 0);

  const s = leftVis >= rightVis ? ls : rs;
  const h = leftVis >= rightVis ? lh : rh;
  const a = leftVis >= rightVis ? la : ra;

  const minVis = Math.min(s.visibility ?? 0, h.visibility ?? 0);
  if (minVis < 0.40) {
    return { state: "broken", angle: 0, cue: "Step back into view" };
  }

  // Orientation check: in a plank, body must be roughly horizontal
  const dx = Math.abs(s.x - a.x);
  const dy = Math.abs(s.y - a.y);
  const isHorizontal = dx > dy * 0.65;

  if (!isHorizontal) {
    return { state: "broken", angle: 0, cue: "Get down into plank posture" };
  }

  const angle = calcAngleDeg(s, h, a);

  // Sag vs Pike deviation relative to line connecting shoulder and ankle
  const t = (h.x - s.x) / (a.x - s.x || 0.0001);
  const lineY = s.y + t * (a.y - s.y);
  const diffY = h.y - lineY; // positive = hip sagging down toward floor

  if (angle >= 158 && angle <= 180 && Math.abs(diffY) <= 0.038) {
    return { state: "good", angle, cue: "Great line — hold steady!" };
  }

  if (diffY > 0.035 || (angle < 158 && diffY > 0)) {
    return { state: "sag", angle, cue: "Raise your hips to align with core" };
  }

  if (diffY < -0.035 || (angle < 158 && diffY < 0)) {
    return { state: "pike", angle, cue: "Lower your hips to a straight line" };
  }

  if (angle >= 148 && angle <= 188) {
    return { state: "good", angle, cue: "Good posture — keep holding" };
  }

  return { state: "broken", angle, cue: "Plank posture broken" };
}

// ---------------------------------------------------------------------------
// framing & positioning evaluator
// Checks if the user has set the phone down and stepped into full view.
function evaluateFraming(lm, ex) {
  if (!lm || lm.length < 33) {
    return { ok: false, reason: "Step into camera view", isHoldingPhone: false };
  }

  const ls = lm[11], rs = lm[12]; // shoulders
  const lh = lm[23], rh = lm[24]; // hips
  const lk = lm[25], rk = lm[26]; // knees

  const sVisMin = Math.min(ls.visibility ?? 0, rs.visibility ?? 0);
  const sVisMax = Math.max(ls.visibility ?? 0, rs.visibility ?? 0);
  const hVisMin = Math.min(lh.visibility ?? 0, rh.visibility ?? 0);
  const hVisMax = Math.max(lh.visibility ?? 0, rh.visibility ?? 0);
  const kVisMax = Math.max(lk.visibility ?? 0, rk.visibility ?? 0);

  // 1. Proximity / Holding phone check:
  const shoulderWidth = Math.abs(ls.x - rs.x);
  const shoulderDist = Math.hypot(ls.x - rs.x, ls.y - rs.y);
  const isSideProfile = shoulderWidth < 0.12;

  const kp = lm.map((p) => [p.x, p.y, p.z ?? 0, p.visibility ?? 0]);
  const box = bboxOf(kp);
  const isTooClose = shoulderWidth > 0.42 || shoulderDist > 0.45 || box[3] > 0.88 || (sVisMax > 0.5 && Math.min(ls.y, rs.y) < 0.04);

  if (isTooClose) {
    return { ok: false, reason: "Set phone down & step back", isHoldingPhone: true };
  }

  // 2. Shoulders check
  if (isSideProfile || ex === "bicep_curl") {
    // In side profile or bicep curl, at least one shoulder must be clearly visible
    if (sVisMax < 0.45) {
      return { ok: false, reason: "Step back — shoulders not in frame", isHoldingPhone: false };
    }
  } else {
    // Front/angled squat/lunge/pushup need both shoulders
    if (sVisMin < 0.45) {
      return { ok: false, reason: "Step back — shoulders not in frame", isHoldingPhone: false };
    }
  }

  // 3. Hips check
  if (isSideProfile || ex === "bicep_curl") {
    if (hVisMax < 0.38) {
      return { ok: false, reason: "Step back — hips not in frame", isHoldingPhone: false };
    }
  } else {
    if (hVisMin < 0.38) {
      return { ok: false, reason: "Step back — hips not in frame", isHoldingPhone: false };
    }
  }

  // 4. Exercise-specific requirements:
  if (ex === "squat" || ex === "lunge") {
    if (kVisMax < 0.40) {
      return { ok: false, reason: "Step back — knees must be in frame", isHoldingPhone: false };
    }
  }

  if (ex === "bicep_curl") {
    const le = lm[13], re = lm[14]; // elbows
    const lw = lm[15], rw = lm[16]; // wrists
    const leftArmVis = Math.min(ls.visibility ?? 0, le.visibility ?? 0, lw.visibility ?? 0);
    const rightArmVis = Math.min(rs.visibility ?? 0, re.visibility ?? 0, rw.visibility ?? 0);
    const bestArmVis = Math.max(leftArmVis, rightArmVis);
    if (bestArmVis < 0.38) {
      return { ok: false, reason: "Front or side view — keep elbows & wrists in frame", isHoldingPhone: false };
    }
  }

  if (ex === "plank") {
    return { ok: true, reason: "Place phone on floor, step back into plank", isHoldingPhone: false };
  }

  return { ok: true, reason: "In position! Hold still...", isHoldingPhone: false };
}

// ---------------------------------------------------------------------------
// Real-time Bicep Curl Evaluator & Rep Counter
// Tracks fist elevation relative to shoulder level and monitors elbow stability.
// Supports both FRONT and SIDE views, with tolerance for natural movement.
function evaluateBicepCurlLive(lm, now) {
  if (!lm || lm.length < 33) return;

  const arms = [
    { side: "left", s: lm[11], e: lm[13], w: lm[15], state: bicepCurlArms.left },
    { side: "right", s: lm[12], e: lm[14], w: lm[16], state: bicepCurlArms.right },
  ];

  let repCompletedThisFrame = false;
  let activePhaseText = "READY";
  let activeCue = null;

  for (const arm of arms) {
    const s = arm.s, e = arm.e, w = arm.w;
    const vis = Math.min(s.visibility ?? 0, e.visibility ?? 0, w.visibility ?? 0);
    if (vis < 0.35) continue;

    const uLen = Math.hypot(s.x - e.x, s.y - e.y);
    if (uLen < 0.05) continue;

    const angle = calcAngleDeg(s, e, w);

    // Fist proximity to shoulder level:
    // When fist is near shoulder level, (w.y - s.y) is small (within 40% of upper arm length)
    // or elbow angle is flexed <= 75 degrees.
    const distToShoulderY = w.y - s.y;
    const isAtShoulder = distToShoulderY <= 0.40 * uLen || angle <= 75.0;

    // Full extension at bottom: wrist below elbow or angle near 140+ degrees
    const isExtendedAtBottom = (w.y >= e.y + 0.10 * uLen) || angle >= 140.0;

    const st = arm.state;

    if (st.phase === "ready") {
      st.restElbow = { x: e.x, y: e.y };
      st.peakDisplacement = 0;
      st.repStartMs = 0;

      // User starts curling: arm leaves full bottom extension
      if (!isExtendedAtBottom && angle < 135.0) {
        st.phase = "curling";
        st.repStartMs = now;
        st.peakDisplacement = 0;
      }
    } else if (st.phase === "curling") {
      activePhaseText = "CURLING UP";

      if (st.restElbow) {
        const drift = Math.hypot(e.x - st.restElbow.x, e.y - st.restElbow.y) / uLen;
        if (drift > st.peakDisplacement) st.peakDisplacement = drift;
      }

      if (isAtShoulder) {
        st.phase = "top";
      } else if (isExtendedAtBottom) {
        st.phase = "ready";
      }
    } else if (st.phase === "top") {
      activePhaseText = "TOP CONTRACTION";

      if (st.restElbow) {
        const drift = Math.hypot(e.x - st.restElbow.x, e.y - st.restElbow.y) / uLen;
        if (drift > st.peakDisplacement) st.peakDisplacement = drift;
      }

      // Descending away from shoulder level
      if (distToShoulderY > 0.45 * uLen && angle > 80.0) {
        st.phase = "lowering";
      }
    } else if (st.phase === "lowering") {
      activePhaseText = "LOWERING";

      if (st.restElbow) {
        const drift = Math.hypot(e.x - st.restElbow.x, e.y - st.restElbow.y) / uLen;
        if (drift > st.peakDisplacement) st.peakDisplacement = drift;
      }

      // Reached bottom extension: validate and count rep
      if (isExtendedAtBottom) {
        const repDuration = now - (st.repStartMs || now);
        if (repDuration >= 400 && (now - lastCurlRepMs > 350)) {
          repCompletedThisFrame = true;
          lastCurlRepMs = now;
          bicepCurlReps++;

          // Form assessment: elbow stability check with room for error
          const excessiveElbowMove = st.peakDisplacement > 0.45;
          if (excessiveElbowMove) {
            activeCue = "Rep counted! Next rep, keep elbows more pinned";
          } else {
            activeCue = "Good rep! Elbows stayed stationary";
          }
        }
        st.phase = "ready";
        st.restElbow = { x: e.x, y: e.y };
        st.peakDisplacement = 0;
      }
    }
  }

  if (repCompletedThisFrame) {
    playBeep(880, 0.15);
    const totalReps = tracker ? Math.max(tracker.totalReps(), bicepCurlReps) : bicepCurlReps;
    $("rep-count").textContent = String(totalReps);
    $("phase").textContent = "REP COMPLETED";
    $("phase").className = "rep-phase good";
    if (activeCue) {
      $("cue").textContent = activeCue;
      $("cue").hidden = false;
    }
  } else {
    if (sessionPhase === "active" && !$("phase").textContent.includes("COMPLETED")) {
      $("phase").textContent = activePhaseText;
      $("phase").className = "rep-phase";
    }
  }
}


// ---------------------------------------------------------------------------
// temporal exponential moving average (EMA) filter
// Eliminates jitter/glitching from landmarks while staying responsive to movement
function smoothLandmarks(rawLm) {
  if (!rawLm || rawLm.length < 33) {
    smoothedLandmarks = null;
    return null;
  }
  if (!smoothedLandmarks) {
    smoothedLandmarks = rawLm.map((p) => ({
      x: p.x,
      y: p.y,
      z: p.z ?? 0,
      visibility: p.visibility ?? 0,
    }));
    return smoothedLandmarks;
  }

  const ALPHA = 0.65;
  for (let i = 0; i < 33; i++) {
    const raw = rawLm[i];
    const prev = smoothedLandmarks[i];
    const rawVis = raw.visibility ?? 0;

    prev.visibility = 0.7 * prev.visibility + 0.3 * rawVis;

    const dx = raw.x - prev.x;
    const dy = raw.y - prev.y;
    const distSq = dx * dx + dy * dy;

    // If landmark jumped too far (sudden flip or recovery), snap to raw
    if (distSq > 0.06) {
      prev.x = raw.x;
      prev.y = raw.y;
    } else {
      prev.x = ALPHA * raw.x + (1 - ALPHA) * prev.x;
      prev.y = ALPHA * raw.y + (1 - ALPHA) * prev.y;
    }
  }
  return smoothedLandmarks;
}

// ---------------------------------------------------------------------------
// guidance pill & countdown management
function showGuidePill(text, isReady = false, icon = null) {
  const el = $("hud-guide");
  if (!el) return;
  const iconEl = $("guide-icon");
  const textEl = $("guide-text");
  if (iconEl) iconEl.textContent = icon || (isReady ? "✓" : "📱");
  if (textEl) textEl.textContent = text;
  el.className = "hud-guide" + (isReady ? " ready" : "");
  el.hidden = false;
}

function hideGuidePill() {
  const el = $("hud-guide");
  if (el) el.hidden = true;
}

function startCountdown() {
  if (sessionPhase === "countdown" || sessionPhase === "active") return;
  sessionPhase = "countdown";
  countdownVal = 3;

  const countdownEl = $("hud-countdown");
  const numEl = $("countdown-num");
  const msgEl = $("countdown-msg");

  showGuidePill("In position! Hold still...", true, "✓");

  countdownEl.hidden = false;
  numEl.textContent = "3";
  numEl.style.color = "#ffffff";
  msgEl.textContent = "Get Ready";
  numEl.style.animation = "none";
  numEl.offsetHeight; // trigger reflow
  numEl.style.animation = "popIn 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275)";

  playBeep(440, 0.12);

  clearInterval(countdownTimer);
  countdownTimer = setInterval(() => {
    countdownVal--;
    if (countdownVal > 0) {
      numEl.textContent = String(countdownVal);
      numEl.style.animation = "none";
      numEl.offsetHeight;
      numEl.style.animation = "popIn 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275)";
      playBeep(440, 0.12);
    } else if (countdownVal === 0) {
      numEl.textContent = "GO!";
      numEl.style.color = "var(--good)";
      numEl.style.animation = "none";
      numEl.offsetHeight;
      numEl.style.animation = "popIn 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275)";
      msgEl.textContent = "Start " + (exercise || "exercise");
      playBeep(880, 0.25);
    } else {
      clearInterval(countdownTimer);
      countdownTimer = null;
      beginActiveWorkout();
    }
  }, 1000);
}

function cancelCountdown(reason) {
  if (sessionPhase !== "countdown") return;
  clearInterval(countdownTimer);
  countdownTimer = null;
  sessionPhase = "positioning";
  steadyFrameCount = 0;
  const countdownEl = $("hud-countdown");
  if (countdownEl) countdownEl.hidden = true;
  showGuidePill(reason || "Set phone down & step back", false);
}

function beginActiveWorkout() {
  sessionPhase = "active";
  const countdownEl = $("hud-countdown");
  if (countdownEl) countdownEl.hidden = true;
  hideGuidePill();

  // Reset session cleanly for the active workout
  sessionId = newSessionId();
  firstPost = true;
  frameBuffer = [];
  lastVideoTs = -1;
  tracker = createSetTracker(() => sessionMaxFrames, {
    rollAt: CFG.SESSION_ROLL_AT,
    forceRollAt: CFG.SESSION_FORCE_ROLL_AT,
  });

  $("rep-count").textContent = "0";
  $("phase").textContent = "—";
  $("hud-lock").className = "lock-pill lock-ok";
  $("hud-lock").textContent = "locked on you";
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
  if (video.readyState >= 2 && video.currentTime !== lastVideoTs) {
    lastVideoTs = video.currentTime;
    const result = landmarker.detectForVideo(video, now);
    if (result.landmarks && result.landmarks.length > 0) {
      const rawLm = result.landmarks[0];
      const smoothed = smoothLandmarks(rawLm);
      const lmToUse = smoothed || rawLm;

      if (exercise === "plank") {
        // ==========================================
        // ---- PLANK (HOLD) ENGINE ----
        // ==========================================
        const posture = evaluatePlankPosture(lmToUse);
        const isDecentForm = posture.state === "good" || posture.state === "sag" || posture.state === "pike";

        // Dynamic theme for plank wireframe:
        // Green (good) / Yellow (sag or pike) / Red (broken)
        let plankTheme = { r: 16, g: 185, b: 129 }; // Green
        if (posture.state === "sag" || posture.state === "pike") {
          plankTheme = { r: 250, g: 204, b: 21 }; // Yellow
        } else if (posture.state === "broken") {
          plankTheme = { r: 239, g: 68, b: 68 }; // Red
        }

        drawSkeleton(lmToUse, plankTheme);

        if (!plankHoldActive) {
          // Automatic Start: starts automatically when user assumes a valid plank position
          if (isDecentForm) {
            steadyFrameCount++;
            showGuidePill(posture.state === "good" ? "Starting hold!" : "Starting hold — adjust posture", true, "✓");
            if (steadyFrameCount >= 8) {
              plankHoldActive = true;
              holdStartTime = now;
              holdLastTickMs = now;
              playBeep(880, 0.2); // Start chime
              hideGuidePill();
              $("hud-lock").className = "lock-pill lock-ok";
              $("hud-lock").textContent = "holding plank";
            }
          } else {
            steadyFrameCount = 0;
            showGuidePill(posture.cue, false);
            $("hud-lock").className = "lock-pill lock-unknown";
            $("hud-lock").textContent = "get ready";
          }
        } else {
          // Active hold in progress
          if (isDecentForm) {
            const dt = now - (holdLastTickMs || now);
            holdElapsedMs += dt;
            if (posture.state === "good") {
              holdGoodMs += dt;
              $("phase").textContent = "PERFECT FORM";
              $("phase").className = "rep-phase good";
              $("cue").hidden = true;
            } else {
              holdWarnMs += dt;
              plankDominantFaults[posture.state] = (plankDominantFaults[posture.state] || 0) + dt;
              $("phase").textContent = posture.state === "sag" ? "HIP SAG" : "HIP PIKE";
              $("phase").className = "rep-phase warn";
              $("cue").textContent = posture.cue;
              $("cue").hidden = false;
            }
            holdLastTickMs = now;
            $("rep-count").textContent = formatHoldTime(holdElapsedMs);
            plankBreakFrames = 0;
          } else {
            // Posture broken (collapsed, knees on floor, or stood up)
            plankBreakFrames++;
            $("phase").textContent = "FORM BROKEN";
            $("phase").className = "rep-phase bad";
            $("cue").textContent = "Posture broken — hold straight line";
            $("cue").hidden = false;
            holdLastTickMs = now;

            // Automatic Stop on exhaustion (>1.2s break)
            if (plankBreakFrames > 35) {
              playBeep(440, 0.35); // finish tone
              stopSet();
              return;
            }
          }
        }
      } else {
        // ==========================================
        // ---- REP-BASED EXERCISES ----
        // ==========================================
        const framing = evaluateFraming(rawLm, exercise);
        const inFrame = framing.ok;

        // Draw wireframe only if shoulders are visible and user is not holding phone close
        const sVisMax = Math.max(rawLm[11].visibility ?? 0, rawLm[12].visibility ?? 0);
        const sVisMin = Math.min(rawLm[11].visibility ?? 0, rawLm[12].visibility ?? 0);
        const isSide = Math.abs(rawLm[11].x - rawLm[12].x) < 0.12;
        const canDraw = ((isSide || exercise === "bicep_curl") ? sVisMax >= 0.45 : sVisMin >= 0.45) && !framing.isHoldingPhone;

        if (canDraw) {
          drawSkeleton(lmToUse, { r: 16, g: 185, b: 129 });
        } else {
          smoothedLandmarks = null;
          octx.clearRect(0, 0, overlay.width, overlay.height);
        }

        // STATE MACHINE:
        if (sessionPhase === "positioning") {
          $("hud-lock").className = "lock-pill lock-unknown";
          $("hud-lock").textContent = framing.isHoldingPhone ? "set phone down" : (inFrame ? "in position" : "step back");
          showGuidePill(framing.reason, inFrame);

          if (inFrame) {
            steadyFrameCount++;
            // Stably positioned for ~18 frames (~0.5-0.6s)
            if (steadyFrameCount >= 18) {
              startCountdown();
            }
          } else {
            steadyFrameCount = 0;
          }
        } else if (sessionPhase === "countdown") {
          if (!inFrame && !framing.isHoldingPhone) {
            steadyFrameCount = 0;
            cancelCountdown("Stepped out of frame — step back to restart");
          }
        } else if (sessionPhase === "active") {
          if (inFrame) {
            // Send frame to detector
            frameBuffer.push(buildFrame(lmToUse, now));
            $("hud-lock").className = "lock-pill lock-ok";
            $("hud-lock").textContent = "locked on you";
            hideGuidePill();

            if (exercise === "bicep_curl") {
              evaluateBicepCurlLive(lmToUse, now);
            }
          } else {
            // User walked away, phone tilted, or user walking up to phone to end set.
            // CRITICAL: Do NOT push frames to buffer! Prevents end-of-set false reps.
            $("hud-lock").className = "lock-pill lock-lost";
            $("hud-lock").textContent = framing.isHoldingPhone ? "too close" : "can't see you";
            showGuidePill(framing.reason, false);
          }
        }
      }
    } else {
      // Nobody detected in frame
      smoothedLandmarks = null;
      octx.clearRect(0, 0, overlay.width, overlay.height);

      if (exercise === "plank") {
        if (plankHoldActive) {
          plankBreakFrames++;
          if (plankBreakFrames > 35) {
            playBeep(440, 0.35);
            stopSet();
            return;
          }
        } else {
          showGuidePill("Place phone on floor, step back into plank", false);
          $("hud-lock").className = "lock-pill lock-lost";
          $("hud-lock").textContent = "can't see you";
        }
      } else {
        if (sessionPhase === "positioning") {
          steadyFrameCount = 0;
          showGuidePill("Step into camera view", false);
          $("hud-lock").className = "lock-pill lock-lost";
          $("hud-lock").textContent = "can't see you";
        } else if (sessionPhase === "countdown") {
          cancelCountdown("Can't see you — step back into frame");
        } else if (sessionPhase === "active") {
          $("hud-lock").className = "lock-pill lock-lost";
          $("hud-lock").textContent = "can't see you";
          showGuidePill("Step back into frame", false);
        }
      }
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
  const backendReps = tracker ? tracker.totalReps() : r.rep_count;
  const displayReps = exercise === "bicep_curl" ? Math.max(backendReps, bicepCurlReps) : backendReps;
  $("rep-count").textContent = displayReps;
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
// Clean 14-bone biomechanical graph for fitness tracking (eliminates finger/face clutter)
const SKELETON_BONES = [
  // Shoulders & Torso
  [11, 12], // shoulder-to-shoulder
  [11, 23], // left shoulder to left hip
  [12, 24], // right shoulder to right hip
  [23, 24], // hip-to-hip

  // Left arm
  [11, 13], // left shoulder to left elbow
  [13, 15], // left elbow to left wrist

  // Right arm
  [12, 14], // right shoulder to right elbow
  [14, 16], // right elbow to right wrist

  // Left leg
  [23, 25], // left hip to left knee
  [25, 27], // left knee to left ankle
  [27, 31], // left ankle to left foot

  // Right leg
  [24, 26], // right hip to right knee
  [26, 28], // right knee to right ankle
  [28, 32], // right ankle to right foot
];

const KEY_JOINTS = [
  0,                  // Nose (head anchor)
  11, 12,             // Shoulders
  13, 14,             // Elbows
  15, 16,             // Wrists
  23, 24,             // Hips
  25, 26,             // Knees
  27, 28,             // Ankles
];

function drawSkeleton(lm, theme = { r: 16, g: 185, b: 129 }) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (!lm || lm.length < 33) return;

  // Anchor check: at least one shoulder must be visible (supports side profile view and front view)
  const sVisMax = Math.max(lm[11].visibility ?? 0, lm[12].visibility ?? 0);
  const sVisMin = Math.min(lm[11].visibility ?? 0, lm[12].visibility ?? 0);
  const isSide = Math.abs(lm[11].x - lm[12].x) < 0.12;
  if (isSide ? sVisMax < 0.40 : sVisMin < 0.35) return;

  octx.lineCap = "round";

  // 1. Draw bones
  for (const [a, b] of SKELETON_BONES) {
    const pa = lm[a], pb = lm[b];
    if (!pa || !pb) continue;
    const visA = pa.visibility ?? 0;
    const visB = pb.visibility ?? 0;
    if (visA < 0.45 || visB < 0.45) continue;

    const alpha = Math.min(visA, visB);
    octx.beginPath();
    octx.lineWidth = 3.5;
    octx.strokeStyle = `rgba(${theme.r}, ${theme.g}, ${theme.b}, ${alpha.toFixed(2)})`;
    octx.moveTo(pa.x * overlay.width, pa.y * overlay.height);
    octx.lineTo(pb.x * overlay.width, pb.y * overlay.height);
    octx.stroke();
  }

  // 2. Draw key biomechanical joints
  for (const idx of KEY_JOINTS) {
    const p = lm[idx];
    if (!p) continue;
    const vis = p.visibility ?? 0;
    if (vis < 0.45) continue;

    const x = p.x * overlay.width;
    const y = p.y * overlay.height;

    // Color halo
    octx.beginPath();
    octx.fillStyle = `rgba(${theme.r}, ${theme.g}, ${theme.b}, ${vis.toFixed(2)})`;
    octx.arc(x, y, 5, 0, Math.PI * 2);
    octx.fill();

    // Crisp white inner core
    octx.beginPath();
    octx.fillStyle = `rgba(255, 255, 255, ${(vis * 0.9).toFixed(2)})`;
    octx.arc(x, y, 2, 0, Math.PI * 2);
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

    isHoldExercise = (ex === "plank");
    holdStartTime = null;
    holdElapsedMs = 0;
    holdGoodMs = 0;
    holdWarnMs = 0;
    holdLastTickMs = null;
    plankHoldActive = false;
    plankBreakFrames = 0;
    plankDominantFaults = {};

    bicepCurlReps = 0;
    bicepCurlArms = {
      left: { phase: "ready", restElbow: null, peakDisplacement: 0, repStartMs: 0 },
      right: { phase: "ready", restElbow: null, peakDisplacement: 0, repStartMs: 0 },
    };
    lastCurlRepMs = 0;

    sessionPhase = "positioning";
    steadyFrameCount = 0;
    smoothedLandmarks = null;
    clearInterval(countdownTimer);
    countdownTimer = null;
    hideGuidePill();
    if ($("hud-countdown")) $("hud-countdown").hidden = true;

    // Initialize Web Audio on user gesture
    try {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass && !audioCtx) audioCtx = new AudioContextClass();
      if (audioCtx && audioCtx.state === "suspended") audioCtx.resume();
    } catch {}

    let displayName = ex;
    if (ex === "plank") displayName = "Plank (hold)";
    else if (ex === "bicep_curl") displayName = "Bicep Curl";
    else if (ex === "pushup") displayName = "Push-up";
    $("hud-exercise").textContent = displayName;

    const repUnitEl = $("rep-unit");
    if (repUnitEl) repUnitEl.textContent = isHoldExercise ? "HOLD" : "REPS";
    $("rep-count").textContent = isHoldExercise ? "0s" : "0";
    $("phase").textContent = isHoldExercise ? "GET READY" : "—";
    $("phase").className = "rep-phase";
    $("hud-lock").className = "lock-pill lock-unknown";
    $("hud-lock").textContent = isHoldExercise ? "get ready" : "step back";
    $("cue").hidden = true;
    $("flags").innerHTML = "";
    show("live");

    let initialGuide = "Set phone down & step back";
    if (ex === "plank") initialGuide = "Place phone on floor, step back into plank";
    else if (ex === "bicep_curl") initialGuide = "Front or side view — keep elbows & fists in frame";
    showGuidePill(initialGuide, false);

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
  plankHoldActive = false;
  clearInterval(countdownTimer);
  countdownTimer = null;
  sessionPhase = "positioning";
  smoothedLandmarks = null;
  hideGuidePill();
  if ($("hud-countdown")) $("hud-countdown").hidden = true;

  if (postTimer) {
    clearInterval(postTimer);
    postTimer = null;
  }
  clearState();

  if (!isHoldExercise) {
    // Wait out any POST already in flight, then deliver whatever is still queued.
    for (let i = 0; i < 50 && flushInFlight; i++) await new Promise((r) => setTimeout(r, 100));
    await flush({ final: true }); // final flush so the last reps are counted
  }
  stopCamera(false); // keep stream warm for next exercise, pause video
  scheduleCameraRelease();
  octx.clearRect(0, 0, overlay.width, overlay.height);
  renderSummary(lastResponse);
  show("summary");
}

function backToPicker() {
  running = false;
  plankHoldActive = false;
  clearInterval(countdownTimer);
  countdownTimer = null;
  sessionPhase = "positioning";
  smoothedLandmarks = null;
  hideGuidePill();
  if ($("hud-countdown")) $("hud-countdown").hidden = true;

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
// summary from the accumulated reps[] or hold time
function renderSummary(r) {
  if (isHoldExercise) {
    $("sum-reps").textContent = formatHoldTime(holdElapsedMs);
    const sumLabel = $("sum-label");
    if (sumLabel) sumLabel.textContent = "total hold time";

    const totalHoldSecs = Math.round(holdElapsedMs / 1000);
    const goodSecs = Math.round(holdGoodMs / 1000);
    const warnSecs = Math.max(0, totalHoldSecs - goodSecs);
    const goodPct = totalHoldSecs > 0 ? Math.round((holdGoodMs / holdElapsedMs) * 100) : 0;
    const warnPct = totalHoldSecs > 0 ? 100 - goodPct : 0;

    $("sum-clean").textContent = totalHoldSecs >= 3
      ? (goodPct >= 70 ? "Strong hold! Excellent alignment." : "Good effort — room to improve core alignment.")
      : "Hold was too short to score.";

    const breakdownEl = $("sum-hold-breakdown");
    if (breakdownEl) {
      breakdownEl.hidden = false;
      breakdownEl.innerHTML = `
        <div class="hold-progress-bar">
          <div class="hold-bar-good" style="width: ${goodPct}%"></div>
          <div class="hold-bar-warn" style="width: ${warnPct}%"></div>
        </div>
        <div class="hold-metrics-list">
          <div class="hold-metric-row">
            <span class="hold-tag"><span class="hold-dot good"></span> Perfect Form</span>
            <span class="hold-val">${goodSecs}s (${goodPct}%)</span>
          </div>
          <div class="hold-metric-row">
            <span class="hold-tag"><span class="hold-dot warn"></span> Adjusted Form</span>
            <span class="hold-val">${warnSecs}s (${warnPct}%)</span>
          </div>
        </div>
      `;
    }

    $("sum-flags").innerHTML = "";
    let cueText = "Great effort.";
    if (plankDominantFaults["sag"] && plankDominantFaults["sag"] > (plankDominantFaults["pike"] || 0)) {
      cueText = "Primary deviation: Hip Sag — focus on engaging your core and lifting hips to align with shoulders.";
    } else if (plankDominantFaults["pike"]) {
      cueText = "Primary deviation: Hip Pike — focus on lowering your hips slightly to maintain a flat, neutral spine.";
    } else if (goodPct >= 80) {
      cueText = "Excellent stamina and straight-line posture throughout your hold!";
    }
    $("sum-cue").textContent = cueText;
    return;
  }

  // Rep-based summary
  const breakdownEl = $("sum-hold-breakdown");
  if (breakdownEl) breakdownEl.hidden = true;
  const sumLabel = $("sum-label");
  if (sumLabel) sumLabel.textContent = "reps counted";

  // A long set may have spanned several server sessions; the tracker holds all of them.
  const reps = tracker ? tracker.allReps() : (r && r.reps) || [];
  const backendTotal = tracker ? tracker.totalReps() : r ? r.rep_count : 0;
  const total = exercise === "bicep_curl" ? Math.max(backendTotal, bicepCurlReps) : backendTotal;
  $("sum-reps").textContent = total;

  const flagged = reps.filter((x) => x.flags && x.flags.length > 0);
  const clean = reps.length - flagged.length;
  let cleanText = reps.length
    ? `${clean} clean · ${flagged.length} flagged`
    : (total > 0 ? `${total} clean curls completed` : "No completed reps detected.");
  $("sum-clean").textContent = cleanText;

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
  let note = total > 0
    ? (exercise === "bicep_curl" ? "Fists reached shoulder level with controlled elbow position. Keypoints only processed." : "Keypoints only were sent to the detector — no video left your device.")
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
const skipBtn = $("btn-skip-countdown");
if (skipBtn) {
  skipBtn.addEventListener("click", () => {
    clearInterval(countdownTimer);
    countdownTimer = null;
    playBeep(880, 0.2);
    beginActiveWorkout();
  });
}

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
