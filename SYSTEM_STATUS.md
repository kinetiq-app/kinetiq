# Kinetiq v5 — System Architecture, Status & Living Context Tracker

> **Single Source of Context**: This document serves as the master engineering status sheet and technical reference for **Kinetiq v5**. Consult and update this sheet across development sessions to avoid re-analyzing the codebase or setting context from scratch.
>
> **Last Verified & Updated**: September 27, 2026  
> **Test Suite Status**: 316/316 Python Tests PASS (100%) | 10/10 Frontend Tests PASS (100%) | Stage 0 Eval Gate PASS | Live Render Deploy: HEALTHY & REACHABLE

---

## 1. Executive Summary & Product Vision

**Kinetiq** is a real-time, privacy-first AI virtual physical trainer designed for mobile browsers (PWA). It transforms a smartphone camera into an automated coach that counts repetitions, tracks movement phases, identifies kinetic/biomechanical form defects, and issues real-time voice/text cues.

### Core Non-Negotiable Invariants
1. **Privacy First (Zero Video Egress)**: Video and image pixels **never** leave the user's phone. Pose estimation executes client-side on-device via WebAssembly/WebGL. Only normalized 3D keypoint arrays cross the wire.
2. **One Detector, N Consumers**: Detection logic lives in a single source (`evals/gate0/detector/`). The live API, offline eval harness, and effectiveness reports all invoke the exact same detector entry point (`run_detector`).
3. **Deterministic Safety Veto**: Safety flags (e.g., severe lumbar flexion, hip collapse, knee valgus) cannot be overridden by learned heuristics or optimistic thresholds.
4. **Config as Single Source of Truth**: All thresholds live in `exercises/*.json` contracts or `backend/app/core/config.py`. Inline magic numbers are strictly forbidden.
5. **GATE G-REAL**: No threshold fine-tuning or learned scoring models until verified against real-world human workout data.

---

## 2. System Architecture & End-to-End Data Flow

```
                                      CLIENT DEVICE (Browser / PWA)
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  [ Camera Feed ]                                                                                        |
|         │                                                                                               |
|         ▼                                                                                               |
|  [ MediaPipe PoseLandmarker (BlazePose Lite float16, WebGL/WASM) ]                                      |
|         │                                                                                               |
|         ▼                                                                                               |
|  [ 33 Keypoints {x, y, z, vis} ] ──▶ [ bboxOf() Derivation ] ──▶ [ Frame Builder: {t_ms, pose, people} ]|
|                                                                                 │                       |
|                                                                                 ▼                       |
|                                                                     [ Frame Buffer Queue ]              |
|                                                                                 │ (boundQueue limit)    |
|                                                                                 ▼                       |
|                                                                     [ createSetTracker (segments.js) ]  |
|                                                                                 │ (roll at 80% / 95%)   |
+---------------------------------------------------------------------------------┼-----------------------+
                                                                                  │ HTTP POST /prototype/assess
                                                                                  │ (Every 400ms, batch keypoints)
                                                                                  ▼
                                                            BACKEND (FastAPI on Render / Cloud)
+---------------------------------------------------------------------------------------------------------+
|                                                                                                         |
|  [ FastAPI Route: /prototype/assess ]                                                                   |
|         │                                                                                               |
|         ▼                                                                                               |
|  [ Schema & Invariant Validator ] (validate_frame_schema + Bounding Box check + Exercise Scope check)  |
|         │                                                                                               |
|         ▼                                                                                               |
|  [ SessionBufferStore (In-Memory Ring/List) ] ──▶ Append incoming frames to session buffer              |
|         │                                                                                               |
|         ▼                                                                                               |
|  [ CORE COMPOUND DETECTOR: run_detector(buffer, exercise_id) ]                                          |
|         │                                                                                               |
|         ├─▶ 1. subject_lock.py: Bounding box tracking & bystander filtering                             |
|         ├─▶ 2. plausibility.py: Biomechanical skeleton plausibility checks (bone proportions)           |
|         ├─▶ 3. exercise_signals.py: Kinematic signal extraction (joint angles, displacement, velocities)|
|         ├─▶ 4. rep_counter.py: Finite State Machine (READY ─▶ DESCENDING ─▶ INFLECTION ─▶ ASCENDING)     |
|         ├─▶ 5. faults.py: Form defect evaluators (depth, hip sag, knee cave, trunk angle)               |
|         ├─▶ 6. flag_hysteresis.py: Sustained vs transient fault filtering & confidence gating           |
|         └─▶ 7. Deterministic Safety Veto: Immutable rejection of invalid/unsafe reps                    |
|         │                                                                                               |
|         ▼                                                                                               |
|  [ Cue Generator: cues.py ] ──▶ Maps active faults to exercise library copy (word-cap limited)          |
|         │                                                                                               |
|         ▼                                                                                               |
|  [ AssessResponse JSON ] ──▶ { rep_count, phase, current_flags, coaching_cue, subject_lock_ok, reps }  |
|                                                                                                         |
+---------------------------------------------------------------------------------┬-----------------------+
                                                                                  │ HTTP 200 JSON
                                                                                  │ (With full CORS headers)
                                                                                  ▼
+---------------------------------------------------------------------------------------------------------+
|  CLIENT UI (Live HUD / Canvas)                                                                          |
|  - Updates Rep Counter HUD & Phase indicator                                                            |
|  - Renders Skeleton Overlay on Canvas                                                                   |
|  - Displays Severity-Colored Fault Pills (emerald/warn/danger) & Coaching Cue bubble                    |
|  - Set Completion: Generates Clean vs Flagged breakdown & post-set summary                             |
+---------------------------------------------------------------------------------------------------------+
```

---

## 3. Component Inventory & Directory Layout

| Directory / File | Type | Purpose | Production Readiness |
| :--- | :--- | :--- | :--- |
| `frontend/` | Vanilla JS / PWA | Camera capture, MediaPipe inference, resilient HTTP batching, HUD rendering, Service Worker cache. | Functional prototype. Connected to deployed API. |
| `evals/gate0/prototype_api/` | Python / FastAPI | Real-time detector API wrapper (`main.py`), in-memory session buffer (`session_buffer.py`), deploy verifier (`verify_deploy.py`). | Serving live on Render. Single-instance only (in-memory). |
| `evals/gate0/detector/` | Python Engine | Core compound deterministic motion intelligence: subject lock, angles, rep counter state machine, fault rules. | 100% test-green. Eval validated on synthetic datasets. |
| `exercises/` | JSON Contracts | Formal schema definitions for 14 exercises: joint landmarks, depth thresholds, fault definitions, coaching cues. | 3 vision-live (squat, pushup, lunge); 11 defined & gated. |
| `backend/app/core/` | Python Config | Single source of truth for global constants, thresholds, floors, and Pydantic schemas (`config.py`, `schemas.py`). | Fully referenced by harness and engine. |
| `evals/gate0/golden/` | Ground Truth Data | Frozen keypoint recordings (`.keypoints.jsonl`) and verified labels (`.labels.json`) across view angles. | Golden set active and passing regression checks. |
| `evals/gate0/` | Eval Harness | Automated quality scorers (`form_pr.py`, `phantom.py`, `rep_match.py`, `subject_lock.py`, `view.py`, `aggregate.py`). | Stage 0 gate fully automated and passing in CI. |
| `.claude/` | Agentic Tooling | In-repo read-only verifiers (`contract-verifier`, `deployment-verifier`, `error-surface-auditor`) & skills. | Operating discipline active. |

---

## 4. Critical Assessment: Root Causes of Historical & Discovered Flaws

### Historical "Failed to Fetch" Saga (Four Compounding Flaws)
1. **Unwired Environment Target**: `config.js` shipped targeting `http://localhost:8000`. On a mobile device, this caused loopback connection failure and HTTPS mixed-content blockage.
2. **Aggressive Stale Service Worker**: Cache-first worker cached `app.js` and `config.js` without revalidation; updates were invisible to returning users.
3. **The `box` Key KeyError & CORS Dropping**: MediaPipe returned landmarks without bounding boxes. `validate_frame_schema` didn't enforce bounding box existence. In Python, `subject_lock` threw unhandled `KeyError: 'box'`. Because Starlette handles unhandled 500s above CORS middleware, the 500 reached the browser with no CORS headers, masked as a generic browser `Failed to fetch`.
4. **Session Buffer OOM & 413 Cascades**: Sets longer than 2 minutes exceeded `PROTOTYPE_SESSION_MAX_FRAMES` (3600 frames). API returned 413, but the client retried the whole backlog repeatedly, compounding the payload until memory crash.

### Newly Uncovered & Resolved Flaws in this Prototype
* **Flaw A: Unhandled `RuntimeError` in `BlazePoseAdapter.is_available()`**  
  * *Symptom*: Running `python -m unittest discover -s evals/gate0 -p "test_*.py"` failed on test 316: `test_capture_without_runtime_returns_nonzero_and_suggests_dry_run`.
  * *Root Cause*: In environments where MediaPipe is installed via pip/conda, `is_available()` returned `True`, but `pose_landmarker.task` was not present on disk. When `infer_frames` ran, it threw a C++ `RuntimeError` instead of raising `PoseRuntimeUnavailable`.
  * *Resolution*: Updated `BlazePoseAdapter.is_available()` in `evals/gate0/detector/pose_capture/blazepose_adapter.py` to verify both `cv2`/`mediapipe` imports and `Path(model_path).is_file()`. All 316 tests now pass.
* **Flaw B: Node.js ES Module Test Runner Failure**  
  * *Symptom*: Running `node --test frontend/segments.test.mjs` crashed with `SyntaxError: Named export 'boundQueue' not found`.
  * *Root Cause*: `frontend/` lacked a `package.json` declaring `"type": "module"`. Node.js defaulted to CommonJS parsing for `segments.js`, breaking ESM named imports in tests.
  * *Resolution*: Created `frontend/package.json` with `"type": "module"`. All 10 node unit tests now pass natively.
* **Flaw C: In-Memory Single-Process Scalability Trap**  
  * *Root Cause*: `SessionBufferStore` holds all active session frames in a local Python dictionary in process memory.
  * *Impact*: Deploying more than 1 container or worker process on Render or AWS ECS will immediately corrupt workout tracking due to split requests. Furthermore, running `run_detector()` over the entire history on every frame batch creates an $O(N)$ CPU cost per poll.
* **Flaw E: Camera Instruction Overlay Lock on Refresh & Exercise Switch**
  * *Symptom*: When switching exercises or refreshing the page after camera permissions were already granted, the instruction overlay ("Allow camera access to begin...") reappeared and remained stuck on the screen, even while reps were actively counting in the background.
  * *Root Cause Analysis*:
    1. **Lack of Permissions API Query**: `startSet()` unconditionally invoked `setState("Camera", "Allow camera access...")` before `startCamera()`, regardless of whether permission was already granted or stream was active.
    2. **Premature Hardware Tear-down & Re-negotiation**: Switching exercises called `stopCamera()`, tearing down video tracks and re-invoking `getUserMedia()`. On mobile OS, camera hardware release is asynchronous, causing hardware lock latency and browser race conditions.
    3. **Unprotected Asynchronous Re-entry**: `startSet()` had no concurrency guard (`isStartingSet`), allowing double taps or quick clicks to spawn concurrent initialization tasks where a secondary call rewrote the state overlay after `clearState()` had already run.
    4. **No Escape Hatch in UI**: `#state-overlay` lacked a dismiss button, completely trapping users when an overlay lingered.
  * *Resolution*:
    1. Added `getCameraPermissionState()` using the Permissions API; only display the camera instruction card if permissions are not already granted (`'prompt'`).
    2. Implemented active stream reuse across exercise switches: if `stream` has live tracks, `startCamera()` re-attaches and plays instantly without renegotiating `getUserMedia` or resetting camera hardware.
    3. Added 45s idle release timer (`scheduleCameraRelease()`) to shut down tracks after inactivity on the picker.
    4. Added `isStartingSet` concurrency lock preventing parallel setup races.
    5. Added auto-dismiss failsafe in `loop()`: if video is actively playing frames (`currentTime > 0`), obsolete "Camera" or "Warming up" overlays are immediately cleared.
    6. Added an explicit dismiss (`×`) button on `#state-overlay`.
    * **Flaw F: Cloudflare Pages CORS Rejection & Lock-On Failure**
  * *Symptom*: Live PWA deployed on Cloudflare Pages displays "locating you…" indefinitely; reps are not counted; server cannot be woken; and banner reports waking server forever.
  * *Root Cause Analysis*:
    1. **Strict Origin Mismatch on Render Backend**: `kinetiq-v5-api.onrender.com` had `PROTOTYPE_API_CORS_ORIGINS` configured strictly to `https://kinetiq-v5-pwa.onrender.com`. When the PWA was moved to Cloudflare Pages (`https://kinetiq.pages.dev`), browser requests sent `Origin: https://kinetiq.pages.dev`. The backend responded with `400 Bad Request: Disallowed CORS origin`.
    2. **Silent Failure in PWA**: Because `/health` probes were rejected by CORS, `apiWarm` stayed `false`, frames could not be posted via `/prototype/assess`, and subject lock responses never arrived to transition the HUD pill from "locating you…" to "locked on".
    3. **Service Worker Fallback Pitfall**: `sw.js` was caching same-origin routes and falling back to `index.html` on failed GET requests, which could return HTML instead of JSON for `/health`.
  * *Resolution*:
    1. Created `frontend/_worker.js` (Cloudflare Pages Advanced Mode) and `functions/` (Standard Mode) implementing an edge reverse-proxy for `/health` and `/prototype/*` directly to `https://kinetiq-v5-api.onrender.com` rewriting origin headers.
    2. Updated `frontend/app.js` to route `*.pages.dev` to same-origin (`window.location.origin`), completely eliminating browser CORS preflight overhead.
    3. Updated `frontend/sw.js` to explicitly bypass Service Worker caching for `/health` and `/prototype/*`.
    4. Updated `evals/gate0/prototype_api/main.py` CORS middleware with origin regex supporting `*.pages.dev`, `*.onrender.com`, `*.trycloudflare.com`, and localhost.
    5. Bumped Service Worker cache to `kinetiq-v5-shell-v4` and cache-buster in `index.html` to `v=5.0.4`.

* **Flaw G: Cloudflare Worker Assets Build Error (`Uploading Pages _worker.js as an asset`)**
  * *Symptom*: Build pipeline failed during `npx wrangler deploy` with: `✘ [ERROR] Uploading a Pages _worker.js file as an asset... This could expose your private server-side code to the public Internet.`
  * *Root Cause Analysis*:
    1. The project uses Cloudflare Workers with Static Assets (`npx wrangler deploy`).
    2. In the absence of an explicit `wrangler.jsonc`, Wrangler fell back to non-interactive defaults in CI and treated `frontend/` strictly as an assets directory.
    3. Placing `_worker.js` inside `frontend/` triggered Wrangler's asset scanner error.
    4. Having a top-level `functions/` directory prompted Wrangler for interactive confirmation, which defaulted to "no" in CI.
  * *Resolution*:
    1. Added an explicit root `wrangler.jsonc` declaring `"main": "worker.js"` and `"assets": { "directory": "frontend", "binding": "ASSETS" }`, preventing non-interactive prompt fallbacks.
    2. Moved the edge reverse proxy entry point to root `worker.js` (outside `frontend/`).
    3. Removed `frontend/_worker.js` and `functions/`.
    4. Added `frontend/.assetsignore` ignoring `_worker.js`.
    5. Updated `frontend/app.js` to recognize `*.workers.dev` origins for same-origin routing.

---

## 5. Current Verification & Health Matrix

| Test / Check | Command / URL | Result | Notes |
| :--- | :--- | :--- | :--- |
| **Cloudflare Production Pipeline** | `https://github.com/kinetiq-app/kinetiq.git` -> Wrangler | **TRACKED (main)** | Auto-deploys via root `wrangler.jsonc` + `worker.js`. |
| **Python Unittest Suite** | `python -m unittest discover -s evals/gate0 -p "test_*.py"` | **PASS (316/316)** | Run time ~21s. Covers detector, scorers, labeling, CLI, schemas, and cues. |
| **Prototype API Test Suite** | `python -m unittest discover -s evals/gate0/prototype_api -p "test_*.py"` | **PASS (53/53)** | Run time ~1.5s. Covers endpoints, CORS, sessions, cues. |
| **Frontend Segment Tests** | `node --test frontend/segments.test.mjs` | **PASS (10/10)** | Validates session rolling, queue bounding, frame acknowledgment. |
| **Stage 0 Golden Set Eval** | `python evals/gate0/aggregate.py --golden evals/gate0/golden --mode full` | **PASS** | 100% rep accuracy, 0 phantom reps, 100% subject lock, form precision/recall met. |

---

## 6. Living Change Log & Decisions

| Date | Author / Agent | Changes Made | Rationale |
| :--- | :--- | :--- | :--- |
| **2026-09-28** | Antigravity AI | - **Cloudflare Worker Static Assets Configuration**: Added root `wrangler.jsonc` and `worker.js`, removed `frontend/_worker.js` and `functions/`, and added `frontend/.assetsignore`.<br>- **CI Non-Interactive Clean Build**: Prevents Wrangler from prompting in CI and eliminates the `Uploading a Pages _worker.js file as an asset` fatal error.<br>- **Supported `workers.dev` Origins**: Updated `frontend/app.js` to resolve `*.workers.dev` to `window.location.origin`. | Resolve Cloudflare deployment build failure while preserving edge reverse-proxying of detector API endpoints. |
| **2026-09-28** | Antigravity AI | - **Cloudflare Pages Edge Reverse Proxy**: Created `frontend/_worker.js` and `functions/` to proxy `/health` and `/prototype/*` to the detector API on Render, eliminating CORS mismatch.<br>- **PWA Same-Origin Routing**: Updated `frontend/app.js` to resolve `*.pages.dev` to `window.location.origin`.<br>- **Protected API Traffic in SW**: Excluded `/health` and `/prototype/*` from `frontend/sw.js`.<br>- **Hardened `pingHealth`**: Added status validation and error logging.<br>- **Bumped SW to v4**: Cache version `kinetiq-v5-shell-v4` and `index.html` asset tags `v=5.0.4`. | Fix server wake-up failure and lock-on failure caused by Cloudflare Pages to Render CORS origin mismatch. |
| **2026-09-27** | Antigravity AI | - **Camera Permission & Overlay UX Overhaul**: Added `getCameraPermissionState()`, active stream reuse across exercise switches, concurrency lock on `startSet`, auto-dismiss failsafe in `loop()`, and manual dismiss button.<br>- **Cache Busted**: Bumped Service Worker cache to `kinetiq-v5-shell-v3` and appended `?v=5.0.3` to asset tags in `index.html`.<br>- **Full Verification**: Node tests (10/10) and Python gate0 unit tests (316/316) passing green. | Eliminate overlay locking bug on refresh and exercise switch, provide instantaneous zero-lag exercise switching, and guarantee full UI usability on mobile devices. |
| **2026-09-27** | Antigravity AI | - **Detached deployment pipeline**: Mounted `frontend/` statically directly on `prototype_api` (`evals/gate0/prototype_api/main.py`), unifying PWA & API into a single same-origin server.<br>- **Instant live development tunnel**: Set up portable `cloudflared.exe` and `dev_tunnel.ps1` allowing instant HTTPS sharing on `https://*.trycloudflare.com` without needing external GitHub repository push access.<br>- **Fixed Camera overlay UI bug**: Added `[hidden] { display: none !important; }` in `styles.css` and explicit `style.display = "none"` in `app.js` so the camera prompt dismisses immediately upon grant.<br>- Verified all 316 Python unit tests and 10 Node.js unit tests green. | Enable 100% independent development and live instant testing on mobile devices without external GitHub permissions. |
| **2026-09-20** | Engineering Team | Initial prototype scaffold carryover from v4; setup in-repo `.claude/` verifiers, render blueprint, and initial docs. | Transition to self-contained v5 prototype root. |

---

## 7. Immediate Next Steps & Strategic Roadmap

1. **Gate G-LIVE (First Real Human Test)**:
   * Perform one complete set of 10 squats in front of the live mobile PWA in an Incognito window.
   * Observe frame rate, MediaPipe landmark jitter, and latency of rep increments on the phone screen.
2. **Backend Scalability Decoupling (Pre-SaaS Transition)**:
   * Move from full session recalculation ($O(N)$) to an incremental streaming state machine ($O(1)$ per frame batch).
   * Replace in-memory `SessionBufferStore` with a lightweight Redis instance or stateful session tokens so multiple backend workers can run concurrently.
3. **PWA Native Polish**:
   * Add screen wake lock API (`navigator.wakeLock.request('screen')`) so the phone display does not dim or sleep during sets.
   * Add Audio / Haptic feedback for rep milestones and safety warnings (audio cues mid-rep without having to look at the screen).
