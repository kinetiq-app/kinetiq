# Kinetiq

**AI-powered movement coaching, entirely on your device.**

Kinetiq uses your webcam and on-device pose estimation to count reps, measure hold durations, detect biomechanical faults, and deliver real-time audio and visual coaching cues — without a single pixel ever leaving your browser.

---

## Table of Contents

- [How it works](#how-it-works)
- [Current Version Features (v5.0)](#current-version-features-v50)
- [Supported Exercises & Biomechanics](#supported-exercises--biomechanics)
- [System Architecture](#system-architecture)
- [Privacy & Security](#privacy--security)
- [Running Locally](#running-locally)
- [Collaborator's Guide](#collaborators-guide)
- [Roadmap & Next Steps](#roadmap--next-steps)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [Deployment & Environments](#deployment--environments)
- [API Reference](#api-reference)
- [Project Structure & Configuration](#project-structure--configuration)

---

## How it works

```
Webcam Video  →  MediaPipe PoseLandmarker (In-browser, WebGL / GPU)
                      ↓
              33 Body Landmarks  [x, y, z, visibility]
                      ↓
    ┌─────────────────┴─────────────────┐
    ↓                                   ↓
On-Device Kinematics Engine      Background Sync Engine
  • Bicep curl 2D/3D tracking      • Session rolling (80% buffer)
  • Forearm plank ground line      • POST /prototype/assess (JSON keypoints only)
  • Zero-latency rep counter       • Render FastAPI detector
  • Web Audio chimes & TTS         • Golden eval benchmarking
    ↓                                   ↓
HUD Overlay & Voice Audio        Long-term Stats & Rep Breakdown
```

The pose model runs entirely inside the browser using WebAssembly and WebGL. Video streams are never transmitted — only compact arrays of 33 floating-point coordinates (~600 bytes per frame) are processed for detection and optional session sync.

---

## Current Version Features (v5.0)

### 🎨 Humanist Studio Luxury SaaS Design System
- **2-Tab Layout**: Seamless switching between **Workouts** (exercise studio) and **Dashboard** (activity analytics).
- **Centered Brand Identity**: Clean `.wordmark` brand placement with dynamic time-of-day greeting (*"Good morning"*, *"Good afternoon"*, *"Good evening"*).
- **Minimalist Workout Cards**: Pure typography hierarchy displaying exercise titles, capsules (`REPS` vs `HOLD`), targeted muscle groups, and camera positioning guidance (SVG stickmen removed for a luxury aesthetic).
- **GitHub-Style Contribution Heatmap**: Interactive, authentic monthly grid (`16px × 16px` tiles, `5px` gap, 4 intensity levels) with Monday–Sunday headers.
- **Weekly Minutes Trend Chart**: Responsive SVG area and line chart tracking daily minutes with active date range badges.
- **User Profile Management**: Top-right avatar dropdown on Dashboard opening an interactive modal to view and edit profile details (name, fitness goals, height, weight, experience level) with smooth saved-state feedback.

### 📐 Real-Time Biomechanics & Movement Analysis
- **Bicep Curl (Front & Side Views)**:
  - Supports both frontal and sagittal camera angles.
  - Full range-of-motion validation: requires deep top flexion and complete bottom extension.
  - Robust cheat rejection: invalidates straight-arm front raises, lateral raises, chicken-wing elbow flares, and foreshortened half-reps.
- **Forearm Plank (Hold)**:
  - Enforces strict horizontal bridge alignment across shoulders, spine, hips, knees, and ankles.
  - Rejects push-ups and straight-arm high planks (requires forearms grounded with elbow angle $\approx 90^\circ$).
  - **Dynamic Perspective Ground Clearance**: Computes the exact 2D floor line between forearms and toes. Rejects resting prone/sphinx poses (hips or knees on floor) as `broken` rather than `sag`.
  - **Fixed Hip Height Baseline**: Records initial suspended hip height; vertical drift past threshold ($>0.10 \times \text{bodyLen}$) pauses the timer.
  - **Hold Initiation Gating**: Timer only starts when user establishes clean, fully suspended form.
- **Squats, Push-Ups & Lunges**:
  - Instant on-device rep counting with zero latency and angle tracking.

### 🔊 Audio & Voice Feedback System
- **Female Voice Audio Output**: Built-in Web Speech API native synthesis speaking aloud *"Set completed!"* with automatic detection of platform female voices (iOS *Samantha*, Windows *Jenny/Zira*, Chrome *Google US English Female*).
- **3-Stage Harmonic Fanfare**: Ascending triangle-wave chord progression (C5+G5 $\to$ E5+B5 $\to$ G5+C6) signaling set completion with resonant acoustic sustain.
- **Form-Break Warning Tone**: Low dual tone (330 Hz $\to$ 260 Hz) sounding immediately when plank form breaks or the timer pauses to prevent wasted user energy.

### 📱 Camera HUD & Mobile Experience
- **Camera Flip Toggle**: In-HUD button to toggle between front selfie camera and rear environment camera on mobile devices (`facingMode: user` $\leftrightarrow$ `environment`).
- **Pixel-Perfect Canvas Sync**: Overlay canvas styled with `object-fit: cover` to prevent coordinate drift across different aspect ratios.
- **Live Guidance Pills & Lock Pill**: Real-time feedback pills indicating tracking status (*"locked on you"*, *"can't see you"*), countdowns, and form cues.

---

## Supported Exercises & Biomechanics

| Exercise | Mode | Primary Joints Tracked | Rejection Rules |
|---|---|---|---|
| **Bicep Curl** | Reps | Shoulder, Elbow, Wrist, Hip | Straight-arm raises, lateral raises, cheat curls, incomplete extension |
| **Plank** | Hold | Shoulder, Elbow, Wrist, Hip, Knee, Ankle | Push-ups (arm $>120^\circ$), hips on floor, knees on floor, excessive drift |
| **Squat** | Reps | Hip, Knee, Ankle | Incomplete depth ($>105^\circ$), incomplete lockout ($<155^\circ$) |
| **Push-up** | Reps | Shoulder, Elbow, Wrist | Incomplete depth ($>100^\circ$), incomplete extension ($<150^\circ$) |
| **Lunge** | Reps | Hip, Knee, Ankle | Incomplete knee flexion ($>105^\circ$), incomplete extension ($<150^\circ$) |

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│  Cloudflare Edge (Workers + Assets)                                    │
│                                                                        │
│   Production:  main branch   →  https://kinetiq.kinetiq.workers.dev    │
│   Staging/Dev: dev branch    →  https://kinetiq-dev.workers.dev        │
│                                                                        │
│   /*                    →  ASSETS.fetch() (frontend/ static files)     │
│   /health, /prototype/* →  Proxy to Render API (rewriting headers)     │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                         Same-Origin HTTPS
                                   │
┌──────────────────────────────────┴─────────────────────────────────────┐
│  Browser PWA (frontend/)                                               │
│                                                                        │
│   UI Layer:       index.html · styles.css · app.js (Design System)     │
│   Offline Engine: sw.js (Cache Storage v4)                             │
│   Vision Engine:  MediaPipe PoseLandmarker Lite (WebGL/GPU Delegate)   │
│   Audio Engine:   Web Audio API (Fanfare/Tones) + Web Speech API (TTS) │
│   Kinematics:     evaluatePlankPosture · evaluateBicepCurlLive         │
│   Buffer Sync:    segments.js (Rolling set tracking)                   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                   POST /prototype/assess (JSON keypoints)
                                   │
┌──────────────────────────────────┴─────────────────────────────────────┐
│  Render API Service (evals/gate0/prototype_api/)                       │
│                                                                        │
│   FastAPI Engine  (Python 3.10 / 3.11, Docker)                         │
│   ├─ In-Memory Session Store (SessionBufferStore)                      │
│   ├─ Keypoint Validator (golden_loader.py)                             │
│   └─ Eval-Backed Detector (evals/gate0/detector/adapter.py)            │
│        └─ Golden Test Fixtures (316 passing unit tests)                │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Privacy & Security

> **Pixels never leave your device.**

1. **Client-Side Vision**: The camera feed is processed exclusively by MediaPipe in browser memory.
2. **Zero Video Transmission**: No video, image frames, or canvas streams are ever uploaded or stored.
3. **Payload Inspection**: The edge proxy and API accept only validated 33-point keypoint arrays (`[x, y, z, visibility] × 33`). Any malformed payload is rejected with HTTP `422`.
4. **Local Hardware Ownership**: Camera streams are paused when navigating away and fully stopped upon session exit or tab closure.

---

## Running Locally

### Prerequisites
- **Node.js**: v18.0.0 or later (v20+ recommended)
- **Python**: 3.10 or 3.11 (only if running the optional Python eval API locally)
- **Wrangler**: Cloudflare CLI (`npm install -g wrangler` or via `npx wrangler`)

### 1. Clone the Repository
```bash
git clone https://github.com/kinetiq-app/kinetiq.git
cd kinetiq
```

### 2. Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### 3. Run with Wrangler (Recommended)
Running Wrangler emulates Cloudflare Workers and static asset bindings locally:
```bash
npx wrangler dev
```
Open the local URL displayed (typically `http://localhost:8787`).

---

## Collaborator's Guide

We welcome contributions to Kinetiq! To ensure codebase stability and clean deployments, follow this branching and development guide.

### Branch Structure

| Branch | Purpose | Deployment Target | Access Policy |
|---|---|---|---|
| **`main`** | Production release branch | `kinetiq.workers.dev` (Production) | Protected; merged only from `dev` |
| **`dev`** | Primary active development & integration | `kinetiq-dev.workers.dev` (Staging) | Active working branch |
| **`test`** | Sandbox mirror of `dev` for peer testing | Testing / Peer Review | Sandboxed collaboration |
| **`feature/*`** | Specific feature or bugfix branches | Local dev environments | Branch off `dev`, PR to `dev` |

### Setting Up Specific Branches Locally

#### Work on the `dev` branch:
```bash
git fetch origin
git checkout dev
git pull origin dev
```

#### Work on or test the `test` branch:
```bash
git fetch origin
git checkout test
git pull origin test
```

#### Create a new feature branch:
Always branch off the latest `dev`:
```bash
git fetch origin
git checkout dev
git pull origin dev
git checkout -b feature/your-feature-name
```

### Pre-Commit Checklist & Quality Gates
Before committing and submitting a pull request, ensure all tests pass:

1. **Run Unit Tests**:
   ```bash
   cd frontend
   npm test
   ```
   *All 38 test suites in `segments.test.mjs` and `kinematics.test.mjs` must pass with 0 failures.*

2. **Commit Message Conventions**:
   Follow conventional commits:
   - `feat(...)`: New feature or capability
   - `fix(...)`: Bug fix or threshold adjustment
   - `docs(...)`: Documentation or guide updates
   - `test(...)`: Adding or updating test cases
   - `refactor(...)`: Code cleanup without functional change

3. **Submitting a Pull Request**:
   - Push your feature branch: `git push -u origin feature/your-feature-name`
   - Open a PR targeting **`dev`** (do not target `main` directly).
   - Once verified and approved on `dev`, release tags are merged into `main`.

---

## Roadmap & Next Steps

### 1. User Profiles & Cloud Persistence
- **Cloudflare D1 / KV Database Integration**:
  - Replace purely local storage with lightweight SQLite on Cloudflare D1.
  - Seamless authentication (magic link / OAuth).
  - Sync user profiles (height, weight, fitness goals, preferences) across devices.
  - Persistent workout history logs and metrics tracking.

### 2. Social & Friends Competition
- **Friend Connections**: Search and add friends via username, QR code, or invite links.
- **Asynchronous Challenges**:
  - Challenge friends to weekly workouts (e.g. *60-Second Plank Challenge*, *50 Clean Squats*).
  - Ghost mode: compete against a friend's recorded rep timeline in real time.
- **Head-to-Head & Leaderboards**:
  - Weekly leaderboard ranking by volume and clean form consistency.
  - Activity feed celebrating PRs, streaks, and completed sets.

### 3. Tier Level Gamification System
- **Tier Hierarchy**:
  - 🥉 **Bronze**: Beginner / Onboarding (0 – 499 Form XP)
  - 🥈 **Silver**: Consistent Practitioner (500 – 1,999 Form XP)
  - 🥇 **Gold**: Form Disciplined (2,000 – 4,999 Form XP)
  - 💎 **Platinum**: Advanced Athlete (5,000 – 9,999 Form XP)
  - ⚡ **Kinetiq Elite**: Master of Biomechanics (10,000+ Form XP)
- **Form-Weighted Scoring**:
  - Clean reps award $1.5\times$ XP; reps with flagged faults reduce XP to incentivize correct technique over reckless speed.
  - Tier badges displayed in the dashboard header, profile modal, and competitive leaderboards.
  - Unlocks exclusive HUD themes and advanced biomechanical analytics.

---

## Testing & Quality Assurance

### Frontend Test Runner (Node.js)
Tests session segment rolling, queue bounding, kinematics angle calculations, and error rejection rules:

```bash
cd frontend
npm test
```

**Test Coverage**:
- `segments.test.mjs`: Buffer roll points, session capping, multi-segment rep continuity.
- `kinematics.test.mjs`:
  - Forearm plank posture, push-up rejection, 90° elbow constraints.
  - Dynamic ground line clearance (hips/knees on floor rejection).
  - Bicep curl 2D/3D flexion, lateral raise rejection, cheat curl rejection.
  - Hip baseline drift threshold handling.

**Current Test Status**: **38 / 38 passing ($100\%$ pass rate)**.

### Backend Python Evals (Optional)
```bash
python -m unittest discover -s evals/gate0 -p "test_*.py"
# 316 detector unit tests
```

---

## Deployment & Environments

### Environments in `wrangler.jsonc`

```jsonc
{
  "name": "kinetiq",
  "main": "worker.js",
  "compatibility_date": "2024-09-23",
  "assets": {
    "directory": "frontend",
    "binding": "ASSETS"
  },
  "env": {
    "dev": {
      "name": "kinetiq-dev",
      "observability": {
        "enabled": true
      },
      "assets": {
        "directory": "frontend",
        "binding": "ASSETS"
      }
    }
  }
}
```

- **Production Deployment** (from `main`):
  ```bash
  npx wrangler deploy
  ```
- **Staging / Dev Deployment** (from `dev`):
  ```bash
  npx wrangler deploy --env dev
  ```

---

## API Reference

Base URL (Production API): `https://kinetiq-v5-api.onrender.com`

### `GET /health`
Returns server status and supported exercises.

### `POST /prototype/assess`
Accepts a batch of keypoint frames and returns rep counts, active phase, detected faults, and real-time coaching cues.

---

## Project Structure & Configuration

```
kinetiq/
├── worker.js                     # Cloudflare Worker edge proxy & asset fetcher
├── wrangler.jsonc                # Cloudflare environments (prod & dev)
├── render.yaml                   # Render deployment blueprint
├── README.md                     # Comprehensive architecture & developer guide
│
├── frontend/                     # Progressive Web App
│   ├── index.html                # App shell, navigation tabs & modals
│   ├── app.js                    # Core logic: kinematics, audio, HUD, camera
│   ├── config.js                 # Runtime API configuration & roll thresholds
│   ├── segments.js               # Session rolling & rep aggregations
│   ├── styles.css                # Humanist Studio Luxury design styles
│   ├── sw.js                     # PWA Service Worker (cache v4)
│   ├── severities.json           # Biomechanical fault severity rankings
│   ├── segments.test.mjs         # Unit tests for session rolling
│   ├── kinematics.test.mjs       # Unit tests for movement kinematics & planks
│   └── package.json              # Test scripts & project metadata
│
├── evals/gate0/                  # Detector evaluation suite
│   ├── detector/                 # Rep state machines & signal definitions
│   └── prototype_api/            # FastAPI wrapper service
│
└── exercises/                    # 14 exercise definitions (JSON)
```

---

## License

Copyright © 2026 Kinetiq Team. All rights reserved.
