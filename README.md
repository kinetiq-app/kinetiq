# Kinetiq

**AI-powered movement coaching, entirely on your device.**

Kinetiq uses your webcam and an on-device pose estimation model to count reps, detect form faults, and deliver real-time coaching cues — without a single pixel ever leaving your browser.

---

## Table of Contents

- [How it works](#how-it-works)
- [Features](#features)
- [Supported exercises](#supported-exercises)
- [Architecture](#architecture)
- [Privacy](#privacy)
- [Running locally](#running-locally)
- [Deployment](#deployment)
- [API reference](#api-reference)
- [Testing](#testing)
- [Project structure](#project-structure)
- [Configuration](#configuration)

---

## How it works

```
Webcam  →  MediaPipe PoseLandmarker (in-browser, GPU)
                ↓
        33 body keypoints  [x, y, z, visibility]
                ↓
        POST /prototype/assess   ← only keypoints cross the wire
                ↓
        FastAPI detector (Render)
          • rep state machine
          • form fault detection
          • coaching cues
                ↓
        rep_count · phase · flags · coaching_cue  →  HUD overlay
```

The pose model runs entirely inside the browser using WebAssembly + WebGL. The backend never receives video — it receives a compact array of 33 floating-point landmark coordinates per frame, typically ~600 bytes.

---

## Features

### Movement Detection
- **Real-time rep counting** — state-machine driven; counts are stable and don't flicker mid-rep
- **Phase tracking** — reports the current phase of each exercise (e.g. *descent*, *hold*, *ascent*)
- **Subject lock** — the detector locks onto the person in frame and tracks them across the session; the HUD shows a green "locked on you" / red "can't see you" pill in real time
- **Session continuity** — a single set can span multiple server sessions without losing the rep count; the frontend rolls sessions automatically at 80% capacity

### Form Analysis
- **Fault detection** — flags biomechanical errors per rep (e.g. *knee_cave*, *forward_lean*, *elbow_flare*)
- **Coaching cues** — plain-language corrective prompts shown live during the set
- **Per-rep breakdown** — summary screen shows total reps, clean reps vs. flagged reps, and a tally of every fault with counts

### Skeleton Overlay
- **Full-body BlazePose wireframe** — 28-connection graph covering torso, full arms (shoulder → elbow → wrist → hand), and full legs (hip → knee → ankle → heel → toe)
- **Confidence-aware rendering** — joints and bones fade with landmark visibility; occluded joints (e.g. feet when out of frame) are hidden rather than guessed at
- **Canvas alignment** — the overlay canvas is kept in sync with the video element at all times, including when switching exercises without releasing the camera

### UX & Performance
- **Camera permission flow** — the permission prompt only appears when permission has not already been granted; subsequent exercise switches reuse the live stream without reinitialising hardware
- **Server wake management** — Render's free tier spins down after idle; Kinetiq pre-warms the API when the picker loads and uses a non-blocking banner during the warm-up so you can start immediately
- **Frame buffering with backoff** — frames accumulate locally during network hiccups and replay once the connection recovers; no reps are silently lost
- **Progressive Web App** — installable, offline-capable shell; static assets are cached by a service worker
- **Privacy overlay dismiss** — a manual dismiss button is always available if the overlay becomes stuck

---

## Supported exercises

| Exercise | File |
|---|---|
| Squat | `exercises/squat.json` |
| Pushup | `exercises/pushup.json` |
| Lunge | `exercises/lunge.json` |
| Deadlift | `exercises/deadlift.json` |
| Bicep Curl | `exercises/bicep_curl.json` |
| Overhead Press | `exercises/overhead_press.json` |
| Arnold Press | `exercises/arnold_press.json` |
| Bench Press | `exercises/bench_press.json` |
| Pull-up | `exercises/pull_up.json` |
| Triceps Pushdown | `exercises/triceps_pushdown.json` |
| Hamstring Curl | `exercises/hamstring_curl.json` |
| Hanging Leg Raise | `exercises/hanging_leg_raise.json` |
| Leg Raise | `exercises/leg_raise.json` |
| Plank (hold) | `exercises/plank.json` |

Each exercise file defines the joint landmarks, angle thresholds, rep phase boundaries, and fault rules used by the detector — no code changes are needed to add a new exercise.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Cloudflare Workers  (worker.js + wrangler.jsonc)        │
│                                                          │
│   /health, /prototype/*  →  proxy  →  Render API        │
│   /*                     →  ASSETS.fetch()               │
│                              └─ frontend/ static files   │
└─────────────────────────────────────────────────────────┘
                              │
                   same-origin requests
                              │
┌─────────────────────────────────────────────────────────┐
│  Browser  (frontend/)                                    │
│                                                          │
│   index.html  ·  app.js  ·  styles.css  ·  config.js    │
│   sw.js  (service worker, cache v4)                      │
│   segments.js  (session rolling / rep aggregation)       │
│                                                          │
│   MediaPipe PoseLandmarker Lite  (GPU delegate)          │
│   — runs entirely in WASM, no server involved —          │
└─────────────────────────────────────────────────────────┘
                              │
                  POST /prototype/assess
              (33 keypoints per frame, ~600 B)
                              │
┌─────────────────────────────────────────────────────────┐
│  Render  (evals/gate0/prototype_api/)                    │
│                                                          │
│   FastAPI  ·  main.py                                    │
│   ├─ validate_frame_schema (golden_loader.py)            │
│   ├─ SessionBufferStore (in-memory, per-session)         │
│   └─ run_detector (detector/adapter.py)                  │
│        ├─ subject_lock.py                                │
│        ├─ rep state machine                              │
│        └─ fault rules (exercises/*.json)                 │
└─────────────────────────────────────────────────────────┘
```

### Key design decisions

- **One detector, two consumers.** `run_detector` in `evals/gate0/detector/adapter.py` is the single source of truth for rep counting and fault detection. The API wraps it; it never reimplements detection logic.
- **Recompute-over-buffer.** The detector re-runs over the entire accumulated session buffer on every API call. This is intentionally simple for a prototype: O(frames so far) per call, bounded by `PROTOTYPE_SESSION_MAX_FRAMES`.
- **Edge reverse proxy.** The Cloudflare Worker proxies `/health` and `/prototype/*` to Render, rewriting `Host` and `Origin` headers. The browser sees a single origin — no CORS, no mixed content.
- **In-memory sessions.** Sessions are stored in process memory on Render. A single instance is pinned (`numInstances: 1`) so a user's frames are never split across two buffers.

---

## Privacy

> **Pixels never leave your device.**

- The webcam feed is processed entirely by MediaPipe running in your browser.
- Only 33 floating-point keypoints per frame are transmitted to the backend.
- No images, video frames, or raw pixel data are ever sent over the network.
- Every incoming request is validated against the frame schema before reaching the detector; any payload that isn't the expected `[x, y, z, visibility] × 33` structure is rejected with a `422`.

---

## Running locally

### Prerequisites

| Tool | Version |
|---|---|
| Python | 3.10 or 3.11 |
| Node.js | 18 or later (for tests) |
| pip | latest |

> You do **not** need Node.js to run the app — only for running the frontend unit tests.

### 1 — Clone the repo

```bash
git clone https://github.com/kinetiq-app/kinetiq.git
cd kinetiq
```

### 2 — Install Python dependencies

```bash
pip install -r evals/gate0/prototype_api/requirements.txt
```

### 3 — Start the API

```bash
cd evals/gate0
uvicorn prototype_api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Verify it is up:

```bash
curl http://localhost:8000/health
# {"status":"ok","session_max_frames":3600, ...}
```

### 4 — Serve the frontend

The frontend is plain static HTML/JS — any static file server works.

**Option A — Python's built-in server (simplest)**

```bash
cd frontend
python -m http.server 3000
```

Open `http://localhost:3000` in your browser.

**Option B — Wrangler (matches production behaviour exactly)**

```bash
# Install Wrangler globally if you haven't already
npm install -g wrangler

# From the repo root
npx wrangler dev
```

This runs the Cloudflare Worker locally, proxying API routes to `http://localhost:8000` automatically if `API` resolves to the same-origin. Open the URL printed by Wrangler (typically `http://localhost:8787`).

> **Note:** When running via `python -m http.server`, the frontend auto-detects `localhost` as same-origin and routes API calls to `window.location.origin`. Since the API is on port 8000 and the frontend is on port 3000, you will need to either use Wrangler (option B) or temporarily set `API_BASE_URL` in `frontend/config.js` to `"http://localhost:8000"`.

### 5 — Open the app

1. Allow camera access when prompted.
2. Choose an exercise from the picker.
3. Stand back so your full body is in frame.
4. Start moving — reps, phase, and coaching cues appear in real time.

---

## Deployment

The production stack is:

| Layer | Service | Config |
|---|---|---|
| Frontend + edge proxy | Cloudflare Workers | `wrangler.jsonc`, `worker.js` |
| Detector API | Render (free tier, Docker) | `render.yaml`, `evals/gate0/prototype_api/Dockerfile` |

### Deploying to Cloudflare Workers

```bash
npx wrangler deploy
```

Wrangler reads `wrangler.jsonc` at the repo root, bundles `worker.js`, and uploads `frontend/` as static assets. The build runs automatically on every push to `main` via the GitHub integration configured in the Cloudflare dashboard.

### Deploying the API to Render

Render picks up `render.yaml` at the repo root and builds the Docker image defined in `evals/gate0/prototype_api/Dockerfile`. No manual steps are needed after the initial wiring:

1. Connect the `kinetiq-app/kinetiq` GitHub repo in the Render dashboard.
2. Render detects `render.yaml` and creates the `kinetiq-v5-api` service.
3. Set the `PROTOTYPE_API_CORS_ORIGINS` environment variable in the Render dashboard to your Cloudflare Workers URL (e.g. `https://kinetiq.your-subdomain.workers.dev`).

> **Free tier note:** Render's free tier spins down services after ~15 minutes of inactivity. Kinetiq automatically pre-warms the API when the picker loads and shows a non-blocking banner if the wake-up is still in progress.

---

## API reference

Base URL (production): `https://kinetiq-v5-api.onrender.com`

### `GET /health`

Returns the server status and session configuration.

**Response**
```json
{
  "status": "ok",
  "session_max_frames": 3600,
  "supported_exercises": ["squat", "pushup", "lunge", ...]
}
```

### `POST /prototype/assess`

Accepts a batch of keypoint frames and returns the current rep count, phase, detected faults, and a coaching cue.

**Request body**
```json
{
  "session_id": "sess-1234567890-abc123",
  "exercise_id": "squat",
  "reset": true,
  "frames": [
    {
      "t_ms": 1234567,
      "pose_model": "blazepose_33",
      "people": [
        {
          "track_id": 0,
          "kp": [[x, y, z, vis], ...],
          "box": [x, y, w, h]
        }
      ]
    }
  ]
}
```

- `reset: true` opens a fresh server session; send it on the first frame of every set.
- `kp` is an array of 33 `[x, y, z, visibility]` tuples in BlazePose landmark order (MediaPipe output).
- `box` is the bounding box `[x, y, width, height]` in normalised coordinates, derived from visible landmarks.

**Response**
```json
{
  "rep_count": 5,
  "phase": "descent",
  "subject_lock_ok": true,
  "current_flags": ["knee_cave"],
  "coaching_cue": "Drive your knees out",
  "reps": [
    { "rep_index": 1, "flags": [] },
    { "rep_index": 2, "flags": ["knee_cave"] }
  ]
}
```

| Field | Description |
|---|---|
| `rep_count` | Total completed reps this session |
| `phase` | Current phase of the movement |
| `subject_lock_ok` | Whether the detector is tracking the user |
| `current_flags` | Active form faults in the current rep |
| `coaching_cue` | Plain-language correction (null if form is clean) |
| `reps` | Per-rep record with flags for the summary screen |

---

## Testing

### Frontend unit tests (Node.js)

Tests the session rolling and frame-queue logic in `segments.js`.

```bash
node --test frontend/segments.test.mjs
```

Expected: **10/10 pass**.

### Backend unit tests (Python)

```bash
# Full eval harness — detector, scorers, golden fixtures
python -m unittest discover -s evals/gate0 -p "test_*.py"
# Expected: 316 tests, 0 failures

# API-layer tests only
python -m unittest discover -s evals/gate0/prototype_api -p "test_*.py"
# Expected: 53 tests, 0 failures
```

### Manual smoke test

```bash
# Check the API is up
curl https://kinetiq-v5-api.onrender.com/health

# Send a minimal frame (replace with real keypoint values)
curl -X POST https://kinetiq-v5-api.onrender.com/prototype/assess \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "smoke-test-1",
    "exercise_id": "squat",
    "reset": true,
    "frames": []
  }'
```

---

## Project structure

```
kinetiq/
├── worker.js                        # Cloudflare Worker — edge proxy + static asset handler
├── wrangler.jsonc                   # Cloudflare Workers config
├── render.yaml                      # Render Blueprint (API service)
│
├── frontend/                        # Camera PWA (static, no build step)
│   ├── index.html                   # App shell — picker · live · summary screens
│   ├── app.js                       # Main app logic: camera, pose loop, skeleton, flush
│   ├── config.js                    # Runtime config (API URL, intervals, model name)
│   ├── segments.js                  # Session rolling and rep aggregation across segments
│   ├── styles.css                   # UI styles
│   ├── sw.js                        # Service worker — offline shell, cache v4
│   ├── severities.json              # Fault → severity mapping (high / med / low)
│   ├── segments.test.mjs            # Node.js unit tests for segments.js
│   └── .assetsignore                # Prevents worker files from being uploaded as assets
│
├── evals/gate0/
│   ├── detector/                    # Eval-validated rep counter and fault detector
│   │   ├── adapter.py               # run_detector() — single entry point for all consumers
│   │   ├── exercise_signals.py      # Maps exercise IDs to signal definitions
│   │   ├── subject_lock.py          # Cross-frame person tracking
│   │   └── ...
│   ├── prototype_api/               # FastAPI wrapper around the detector
│   │   ├── main.py                  # App, CORS, /health, /prototype/assess
│   │   ├── schemas.py               # Pydantic request/response models
│   │   ├── session_buffer.py        # In-memory per-session frame store
│   │   ├── cues.py                  # Interim coaching cue lookup
│   │   ├── Dockerfile               # Docker build (context = repo root)
│   │   └── requirements.txt
│   ├── golden_loader.py             # Golden fixture loader + validate_frame_schema
│   └── test_*.py                    # 316 unit tests
│
└── exercises/                       # Exercise definitions (JSON)
    ├── squat.json
    ├── pushup.json
    ├── lunge.json
    └── ...                          # 14 exercises total
```

---

## Configuration

All frontend runtime config lives in [`frontend/config.js`](frontend/config.js):

| Key | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://kinetiq-v5-api.onrender.com` | Detector API origin. Overridden at runtime when the app is served from `localhost`, `*.workers.dev`, or `*.pages.dev` — same-origin routing kicks in automatically. |
| `POST_INTERVAL_MS` | `400` | How often buffered keypoint frames are flushed to the API (ms). |
| `POSE_MODEL` | `"blazepose_33"` | Landmark format tag; must match the detector's `keypoint_map.py`. |
| `SESSION_ROLL_AT` | `0.8` | Roll to a fresh server session when the buffer reaches 80% of the server's frame cap. |
| `SESSION_FORCE_ROLL_AT` | `0.95` | Force a session roll at 95% even mid-rep, to avoid an unrecoverable 413. |

Backend configuration is set via environment variables on Render:

| Variable | Description |
|---|---|
| `PROTOTYPE_API_CORS_ORIGINS` | Comma-separated list of allowed CORS origins (e.g. your Cloudflare Workers URL). Defaults to `*`. |

---

## Live URL

The latest build is deployed at the Cloudflare Workers URL connected to the `kinetiq-app/kinetiq` GitHub repo. Every push to `main` triggers an automatic redeploy.
