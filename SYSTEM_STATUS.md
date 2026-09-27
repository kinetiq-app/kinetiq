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
* **Flaw D: The "Synthetic Data" Mirage (GATE G-REAL)**  
  * *Current Reality*: While 100% of test suites and evals are green, **0 real human workouts** have been verified end-to-end on camera in a gym. Real-world variables (lighting shifts, baggy gym clothing, camera wobble, background lifters) are untested.

---

## 5. Current Verification & Health Matrix

| Test / Check | Command / URL | Result | Notes |
| :--- | :--- | :--- | :--- |
| **Python Unittest Suite** | `python -m unittest discover -s evals/gate0 -p "test_*.py"` | **PASS (316/316)** | Run time ~31s. Covers detector, scorers, labeling, CLI, schemas, and cues. |
| **Frontend Segment Tests** | `node --test frontend/segments.test.mjs` | **PASS (10/10)** | Validates session rolling, queue bounding, frame acknowledgment. |
| **Stage 0 Golden Set Eval** | `python evals/gate0/aggregate.py --golden evals/gate0/golden --mode full` | **PASS** | 100% rep accuracy, 0 phantom reps, 100% subject lock, form precision/recall met. |
| **Deployed API Health** | `GET https://kinetiq-v5-api.onrender.com/health` | **PASS (200 OK)** | Returns version SHA, session max frames (3600), supported exercises. |
| **Deployed End-to-End CORS & Contract** | `python evals/gate0/prototype_api/verify_deploy.py https://kinetiq-v5-api.onrender.com --origin https://kinetiq-v5-pwa.onrender.com` | **ALL CHECKS PASS** | Simulates real browser requests, valid frames, and malformed frames (CORS preserved). |
| **Deployed PWA** | `https://kinetiq-v5-pwa.onrender.com` | **LIVE (200 OK)** | Serves PWA bundle with correct backend API configuration. |

---

## 6. Living Change Log & Decisions

| Date | Author / Agent | Changes Made | Rationale |
| :--- | :--- | :--- | :--- |
| **2026-09-27** | Antigravity AI | - Fixed `BlazePoseAdapter.is_available()` to verify `pose_landmarker.task` file existence (all 316 unit tests green).<br>- Created `frontend/package.json` with `"type": "module"` (all 10 Node tests green).<br>- Fixed local testing origin mismatch in `frontend/app.js` to dynamically detect `localhost`/`127.0.0.1` and route to local API while preserving remote deployed endpoint.<br>- Bumped Service Worker cache to `kinetiq-v5-shell-v1` in `frontend/sw.js`.<br>- Verified local MVP stack end-to-end (`prototype_api` on `:8000` + PWA on `:8080`).<br>- Confirmed live Render API & PWA health via `verify_deploy.py`.<br>- Created master `SYSTEM_STATUS.md`. | Resolve test suite regression, enable seamless frontend testing, eliminate CORS traps for local development, and bring prototype to verified working MVP stage. |
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
