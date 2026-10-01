// Kinematic validation unit tests for Bicep Curl & Plank in app.js
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

// Load app.js into a mocked sandbox to test exact production functions
function loadAppSandbox() {
  const code = fs.readFileSync(new URL("./app.js", import.meta.url), "utf8");
  // Strip ES module imports for Node VM execution
  const strippedCode = code
    .replace(/import\s+[\s\S]*?from\s+["'][^"']+["'];?/g, "// [import stripped]")
    + "\n;globalThis.__getBicepCurlState = () => ({ reps: bicepCurlReps, arms: bicepCurlArms });";

  const elements = {};
  function createMockElement() {
    return {
      textContent: "",
      innerHTML: "",
      className: "",
      style: {},
      hidden: false,
      classList: { remove: () => {}, add: () => {}, contains: () => false },
      getContext: () => ({ clearRect: () => {}, beginPath: () => {}, arc: () => {}, fill: () => {}, stroke: () => {} }),
      addEventListener: () => {},
      removeEventListener: () => {},
      querySelector: () => createMockElement(),
      querySelectorAll: () => [],
      appendChild: () => createMockElement(),
      setAttribute: () => {},
      getAttribute: () => null,
    };
  }

  function getEl(id) {
    if (!elements[id]) {
      elements[id] = createMockElement();
    }
    return elements[id];
  }

  const sandbox = {
    window: {
      location: { search: "", hostname: "localhost", origin: "http://localhost" },
      KINETIQ_CONFIG: { POSE_MODEL: "blazepose", SESSION_ROLL_AT: 0.8, SESSION_FORCE_ROLL_AT: 0.95 },
      AudioContext: class {
        createOscillator() { return { frequency: { setValueAtTime: () => {} }, connect: () => {}, start: () => {}, stop: () => {} }; }
        createGain() { return { gain: { setValueAtTime: () => {}, exponentialRampToValueAtTime: () => {} }, connect: () => {} }; }
        get currentTime() { return 0; }
        get destination() { return {}; }
      },
      addEventListener: () => {},
      matchMedia: () => ({ matches: false, addEventListener: () => {} }),
      localStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
    },
    localStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
    document: {
      getElementById: getEl,
      createElement: () => createMockElement(),
      querySelectorAll: () => [],
      querySelector: () => createMockElement(),
      addEventListener: () => {},
      title: "",
      body: { appendChild: () => {} },
    },
    navigator: {
      serviceWorker: { register: () => Promise.resolve(), addEventListener: () => {} },
      mediaDevices: { getUserMedia: () => Promise.resolve() },
    },
    requestAnimationFrame: () => 0,
    cancelAnimationFrame: () => 0,
    createSetTracker: () => ({ totalReps: () => 0 }),
    boundQueue: () => ({ frames: [], dropped: 0 }),
    performance: { now: () => 0 },
    console,
    Math,
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    URLSearchParams,
  };

  vm.createContext(sandbox);
  vm.runInContext(strippedCode, sandbox);
  sandbox.elements = elements;
  Object.defineProperty(sandbox, "bicepCurlReps", { get: () => sandbox.__getBicepCurlState().reps });
  Object.defineProperty(sandbox, "bicepCurlArms", { get: () => sandbox.__getBicepCurlState().arms });
  return sandbox;
}

// Helper to create empty 33 BlazePose landmarks (unmodeled landmarks have visibility 0)
function createBaseLandmarks() {
  const lm = [];
  for (let i = 0; i < 33; i++) {
    lm.push({ x: 0.5, y: 0.5, z: 0, visibility: 0 });
  }
  return lm;
}

// ---------------------------------------------------------------------------
// PLANK TESTS
// ---------------------------------------------------------------------------
test("Plank: perfect form returns state: good", () => {
  const app = loadAppSandbox();
  const lm = createBaseLandmarks();

  // Side view horizontal plank: shoulder -> hip -> knee -> ankle
  // Elbows on floor propping up chest
  lm[11] = { x: 0.25, y: 0.55, z: 0, visibility: 0.95 }; // shoulder
  lm[13] = { x: 0.25, y: 0.70, z: 0, visibility: 0.95 }; // elbow (0.15 below shoulder)
  lm[15] = { x: 0.35, y: 0.70, z: 0, visibility: 0.95 }; // wrist
  lm[23] = { x: 0.50, y: 0.57, z: 0, visibility: 0.95 }; // hip (straight line)
  lm[25] = { x: 0.65, y: 0.58, z: 0, visibility: 0.95 }; // knee
  lm[27] = { x: 0.80, y: 0.60, z: 0, visibility: 0.95 }; // ankle

  const res = app.evaluatePlankPosture(lm);
  assert.equal(res.state, "good");
  assert.ok(res.angle >= 158);
});

test("Plank: mild hip sag returns state: sag (warn)", () => {
  const app = loadAppSandbox();
  const lm = createBaseLandmarks();

  lm[11] = { x: 0.25, y: 0.55, z: 0, visibility: 0.95 }; // shoulder
  lm[13] = { x: 0.25, y: 0.70, z: 0, visibility: 0.95 }; // elbow
  lm[15] = { x: 0.35, y: 0.70, z: 0, visibility: 0.95 }; // wrist
  // Hip sags slightly (diffY ~ 0.045)
  lm[23] = { x: 0.50, y: 0.62, z: 0, visibility: 0.95 }; // hip sagged slightly
  lm[25] = { x: 0.65, y: 0.60, z: 0, visibility: 0.95 }; // knee
  lm[27] = { x: 0.80, y: 0.60, z: 0, visibility: 0.95 }; // ankle

  const res = app.evaluatePlankPosture(lm);
  assert.equal(res.state, "sag");
});

test("Plank: mild hip pike returns state: pike (warn)", () => {
  const app = loadAppSandbox();
  const lm = createBaseLandmarks();

  lm[11] = { x: 0.25, y: 0.60, z: 0, visibility: 0.95 }; // shoulder
  lm[13] = { x: 0.25, y: 0.75, z: 0, visibility: 0.95 }; // elbow
  lm[15] = { x: 0.35, y: 0.75, z: 0, visibility: 0.95 }; // wrist
  // Hip piked slightly (diffY ~ -0.045)
  lm[23] = { x: 0.50, y: 0.55, z: 0, visibility: 0.95 }; // hip elevated slightly
  lm[25] = { x: 0.65, y: 0.58, z: 0, visibility: 0.95 }; // knee
  lm[27] = { x: 0.80, y: 0.60, z: 0, visibility: 0.95 }; // ankle

  const res = app.evaluatePlankPosture(lm);
  assert.equal(res.state, "pike");
});

test("Plank: severe hip sag near floor triggers red/broken state immediately", () => {
  const app = loadAppSandbox();
  const lm = createBaseLandmarks();

  lm[11] = { x: 0.25, y: 0.55, z: 0, visibility: 0.95 }; // shoulder
  lm[13] = { x: 0.25, y: 0.70, z: 0, visibility: 0.95 }; // elbow
  lm[15] = { x: 0.35, y: 0.70, z: 0, visibility: 0.95 }; // wrist
  // Hips severely sagging down near the floor
  lm[23] = { x: 0.50, y: 0.70, z: 0, visibility: 0.95 }; // hip sagged near floor (diffY ~ 0.12)
  lm[25] = { x: 0.65, y: 0.64, z: 0, visibility: 0.95 }; // knee
  lm[27] = { x: 0.80, y: 0.60, z: 0, visibility: 0.95 }; // ankle

  const res = app.evaluatePlankPosture(lm);
  assert.equal(res.state, "broken");
  assert.match(res.cue, /sagging too low/i);
});

test("Plank: severe hip pike in inverted-V triggers red/broken state immediately", () => {
  const app = loadAppSandbox();
  const lm = createBaseLandmarks();

  lm[11] = { x: 0.25, y: 0.65, z: 0, visibility: 0.95 }; // shoulder
  lm[13] = { x: 0.25, y: 0.80, z: 0, visibility: 0.95 }; // elbow
  lm[15] = { x: 0.35, y: 0.80, z: 0, visibility: 0.95 }; // wrist
  // Hips piked high in inverted-V
  lm[23] = { x: 0.50, y: 0.45, z: 0, visibility: 0.95 }; // hip piked high (diffY ~ -0.20)
  lm[25] = { x: 0.65, y: 0.56, z: 0, visibility: 0.95 }; // knee
  lm[27] = { x: 0.80, y: 0.65, z: 0, visibility: 0.95 }; // ankle

  const res = app.evaluatePlankPosture(lm);
  assert.equal(res.state, "broken");
  assert.match(res.cue, /piked too high/i);
});

test("Plank: user lying flat on floor triggers broken posture", () => {
  const app = loadAppSandbox();
  const lm = createBaseLandmarks();

  // User lying flat on belly on floor: chest and shoulders are resting flat on floor,
  // arm elevation has collapsed (elbows and shoulders both at y = 0.80)
  lm[11] = { x: 0.25, y: 0.80, z: 0, visibility: 0.95 }; // shoulder flat on floor
  lm[13] = { x: 0.28, y: 0.82, z: 0, visibility: 0.95 }; // elbow flat beside body (e.y - s.y = 0.02)
  lm[15] = { x: 0.35, y: 0.82, z: 0, visibility: 0.95 }; // wrist flat on floor
  lm[23] = { x: 0.50, y: 0.80, z: 0, visibility: 0.95 }; // hip flat on floor
  lm[25] = { x: 0.65, y: 0.80, z: 0, visibility: 0.95 }; // knee flat on floor
  lm[27] = { x: 0.80, y: 0.80, z: 0, visibility: 0.95 }; // ankle flat on floor

  const res = app.evaluatePlankPosture(lm);
  assert.equal(res.state, "broken");
  assert.match(res.cue, /resting on floor/i);
});

test("Plank: bent knees on floor triggers broken posture", () => {
  const app = loadAppSandbox();
  const lm = createBaseLandmarks();

  lm[11] = { x: 0.25, y: 0.55, z: 0, visibility: 0.95 }; // shoulder
  lm[13] = { x: 0.25, y: 0.70, z: 0, visibility: 0.95 }; // elbow
  lm[15] = { x: 0.35, y: 0.70, z: 0, visibility: 0.95 }; // wrist
  lm[23] = { x: 0.50, y: 0.58, z: 0, visibility: 0.95 }; // hip
  // Knees resting on floor with bent legs (kneeAngle ~ 125°)
  lm[25] = { x: 0.65, y: 0.70, z: 0, visibility: 0.95 }; // knee down on floor
  lm[27] = { x: 0.75, y: 0.58, z: 0, visibility: 0.95 }; // feet/ankles raised or bent

  const res = app.evaluatePlankPosture(lm);
  assert.equal(res.state, "broken");
  assert.match(res.cue, /knees on floor/i);
});

test("Plank: knees on floor with straight legs triggers broken posture", () => {
  const app = loadAppSandbox();
  const lm = createBaseLandmarks();

  lm[11] = { x: 0.25, y: 0.50, z: 0, visibility: 0.95 }; // shoulder
  lm[13] = { x: 0.25, y: 0.65, z: 0, visibility: 0.95 }; // elbow on floor at 0.65
  lm[15] = { x: 0.35, y: 0.65, z: 0, visibility: 0.95 };
  lm[23] = { x: 0.45, y: 0.60, z: 0, visibility: 0.95 };
  lm[25] = { x: 0.65, y: 0.65, z: 0, visibility: 0.95 }; // knee resting on floor at 0.65
  lm[27] = { x: 0.85, y: 0.70, z: 0, visibility: 0.95 }; // ankle

  const res = app.evaluatePlankPosture(lm);
  assert.equal(res.state, "broken");
  assert.match(res.cue, /knees on floor/i);
});

test("Plank: lying flat on floor with forearms extended triggers broken posture", () => {
  const app = loadAppSandbox();
  const lm = createBaseLandmarks();

  lm[11] = { x: 0.25, y: 0.75, z: 0, visibility: 0.95 }; // shoulder flat on floor
  lm[13] = { x: 0.25, y: 0.82, z: 0, visibility: 0.95 }; // elbow on floor
  lm[15] = { x: 0.35, y: 0.82, z: 0, visibility: 0.95 };
  lm[23] = { x: 0.50, y: 0.75, z: 0, visibility: 0.95 }; // hip flat on floor
  lm[25] = { x: 0.65, y: 0.75, z: 0, visibility: 0.95 }; // knee flat on floor
  lm[27] = { x: 0.80, y: 0.75, z: 0, visibility: 0.95 }; // ankle flat on floor

  const res = app.evaluatePlankPosture(lm);
  assert.equal(res.state, "broken");
  assert.match(res.cue, /resting on floor/i);
});

// ---------------------------------------------------------------------------
// BICEP CURL TESTS
// ---------------------------------------------------------------------------
test("Bicep Curl: straight-arm front raises do NOT register reps", () => {
  const app = loadAppSandbox();
  app.sessionPhase = "active";
  app.exercise = "bicep_curl";

  const lm = createBaseLandmarks();
  // Front view: shoulders wide apart
  lm[11] = { x: 0.40, y: 0.30, z: 0, visibility: 0.95 }; // left shoulder
  lm[12] = { x: 0.60, y: 0.30, z: 0, visibility: 0.95 }; // right shoulder
  lm[23] = { x: 0.42, y: 0.65, z: 0, visibility: 0.95 }; // left hip
  lm[24] = { x: 0.58, y: 0.65, z: 0, visibility: 0.95 }; // right hip

  // 1. Initial hanging arms
  lm[13] = { x: 0.40, y: 0.48, z: 0, visibility: 0.95 }; // elbow down
  lm[15] = { x: 0.40, y: 0.65, z: 0, visibility: 0.95 }; // wrist extended down
  app.evaluateBicepCurlLive(lm, 1000);

  // 2. Straight arm raises forward to shoulder height (elbow straight, angle ~180°)
  // Elbow and wrist both lift forward/upward with straight arm
  for (let t = 1100; t <= 1600; t += 100) {
    lm[13] = { x: 0.40, y: 0.35, z: -0.15, visibility: 0.95 };
    lm[15] = { x: 0.40, y: 0.30, z: -0.30, visibility: 0.95 };
    app.evaluateBicepCurlLive(lm, t);
  }

  // 3. Lower straight arm back down
  for (let t = 1700; t <= 2200; t += 100) {
    lm[13] = { x: 0.40, y: 0.48, z: 0, visibility: 0.95 };
    lm[15] = { x: 0.40, y: 0.65, z: 0, visibility: 0.95 };
    app.evaluateBicepCurlLive(lm, t);
  }

  assert.equal(app.bicepCurlReps, 0, "Straight arm front raise must not register a bicep curl rep");
});

test("Bicep Curl: overhead raises do NOT register reps", () => {
  const app = loadAppSandbox();
  app.sessionPhase = "active";
  app.exercise = "bicep_curl";

  const lm = createBaseLandmarks();
  lm[11] = { x: 0.40, y: 0.35, z: 0, visibility: 0.95 }; // shoulder
  lm[12] = { x: 0.60, y: 0.35, z: 0, visibility: 0.95 };
  lm[23] = { x: 0.42, y: 0.70, z: 0, visibility: 0.95 }; // hip
  lm[24] = { x: 0.58, y: 0.70, z: 0, visibility: 0.95 };

  // Arms extended down at ready
  lm[13] = { x: 0.40, y: 0.52, z: 0, visibility: 0.95 };
  lm[15] = { x: 0.40, y: 0.70, z: 0, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1000);

  // Raise arms overhead (elbow rises well above shoulder: e.y < s.y)
  for (let t = 1100; t <= 1600; t += 100) {
    lm[13] = { x: 0.40, y: 0.20, z: 0, visibility: 0.95 }; // elbow above shoulder
    lm[15] = { x: 0.40, y: 0.05, z: 0, visibility: 0.95 }; // wrist overhead
    app.evaluateBicepCurlLive(lm, t);
  }

  // Lower arms back down
  for (let t = 1700; t <= 2200; t += 100) {
    lm[13] = { x: 0.40, y: 0.52, z: 0, visibility: 0.95 };
    lm[15] = { x: 0.40, y: 0.70, z: 0, visibility: 0.95 };
    app.evaluateBicepCurlLive(lm, t);
  }

  assert.equal(app.bicepCurlReps, 0, "Overhead arm raise must not register a bicep curl rep");
});

test("Bicep Curl: front-view half rep (incomplete flexion) is rejected", () => {
  const app = loadAppSandbox();
  app.sessionPhase = "active";
  app.exercise = "bicep_curl";

  const lm = createBaseLandmarks();
  lm[11] = { x: 0.40, y: 0.30, z: 0, visibility: 0.95 };
  lm[12] = { x: 0.60, y: 0.30, z: 0, visibility: 0.95 };
  lm[23] = { x: 0.42, y: 0.65, z: 0, visibility: 0.95 };
  lm[24] = { x: 0.58, y: 0.65, z: 0, visibility: 0.95 };

  // Bottom extension
  lm[13] = { x: 0.40, y: 0.48, z: 0, visibility: 0.95 };
  lm[15] = { x: 0.40, y: 0.65, z: 0, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1000);

  // Curl only halfway up: forearm horizontal (elbow angle ~90° in 3D, wrist at elbow level)
  for (let t = 1100; t <= 1600; t += 100) {
    lm[13] = { x: 0.40, y: 0.48, z: 0, visibility: 0.95 };
    lm[15] = { x: 0.40, y: 0.48, z: -0.17, visibility: 0.95 }; // 90° angle, wrist at elbow level
    app.evaluateBicepCurlLive(lm, t);
  }

  // Lower back down
  for (let t = 1700; t <= 2200; t += 100) {
    lm[13] = { x: 0.40, y: 0.48, z: 0, visibility: 0.95 };
    lm[15] = { x: 0.40, y: 0.65, z: 0, visibility: 0.95 };
    app.evaluateBicepCurlLive(lm, t);
  }

  assert.equal(app.bicepCurlReps, 0, "Half-rep with incomplete flexion must not count as a rep");
});

test("Bicep Curl: front-view half rep (incomplete extension) is rejected", () => {
  const app = loadAppSandbox();
  app.sessionPhase = "active";
  app.exercise = "bicep_curl";

  const lm = createBaseLandmarks();
  lm[11] = { x: 0.40, y: 0.30, z: 0, visibility: 0.95 };
  lm[12] = { x: 0.60, y: 0.30, z: 0, visibility: 0.95 };
  lm[23] = { x: 0.42, y: 0.65, z: 0, visibility: 0.95 };
  lm[24] = { x: 0.58, y: 0.65, z: 0, visibility: 0.95 };

  // 1. Start from extended
  lm[13] = { x: 0.40, y: 0.48, z: 0, visibility: 0.95 };
  lm[15] = { x: 0.40, y: 0.65, z: 0, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1000);

  // 2. Full top contraction (wrist near shoulder level, deep flexion)
  for (let t = 1100; t <= 1500; t += 100) {
    lm[13] = { x: 0.40, y: 0.48, z: 0, visibility: 0.95 };
    lm[15] = { x: 0.40, y: 0.32, z: -0.05, visibility: 0.95 };
    app.evaluateBicepCurlLive(lm, t);
  }

  // 3. Lower only halfway down (angle ~100°, wrist around elbow height)
  for (let t = 1600; t <= 2000; t += 100) {
    lm[13] = { x: 0.40, y: 0.48, z: 0, visibility: 0.95 };
    lm[15] = { x: 0.40, y: 0.47, z: -0.15, visibility: 0.95 };
    app.evaluateBicepCurlLive(lm, t);
  }

  // 4. Curl back up without reaching bottom
  for (let t = 2100; t <= 2500; t += 100) {
    lm[13] = { x: 0.40, y: 0.48, z: 0, visibility: 0.95 };
    lm[15] = { x: 0.40, y: 0.32, z: -0.05, visibility: 0.95 };
    app.evaluateBicepCurlLive(lm, t);
  }

  assert.equal(app.bicepCurlReps, 0, "Rep with incomplete bottom extension must not count");
});

test("Bicep Curl: full curl in front view counts cleanly", () => {
  const app = loadAppSandbox();
  app.sessionPhase = "active";
  app.exercise = "bicep_curl";

  const lm = createBaseLandmarks();
  lm[11] = { x: 0.40, y: 0.30, z: 0, visibility: 0.95 };
  lm[12] = { x: 0.60, y: 0.30, z: 0, visibility: 0.95 };
  lm[23] = { x: 0.42, y: 0.65, z: 0, visibility: 0.95 };
  lm[24] = { x: 0.58, y: 0.65, z: 0, visibility: 0.95 };

  // 1. Full bottom extension (ready)
  lm[13] = { x: 0.40, y: 0.48, z: 0, visibility: 0.95 };
  lm[15] = { x: 0.40, y: 0.66, z: 0, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1000);

  // 2. Curling upward
  lm[15] = { x: 0.40, y: 0.45, z: -0.10, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1200);

  // 3. Top contraction (deep flexion <= 75°, fist near shoulder)
  lm[15] = { x: 0.40, y: 0.33, z: -0.05, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1400);

  // 4. Lowering
  lm[15] = { x: 0.40, y: 0.45, z: -0.10, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1600);

  // 5. Full bottom extension (angle >= 145°, wrist hanging down)
  lm[15] = { x: 0.40, y: 0.66, z: 0, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1800);

  assert.equal(app.bicepCurlReps, 1, "Full range front-view curl must count as 1 rep");
});

test("Bicep Curl: full curl in side view counts cleanly", () => {
  const app = loadAppSandbox();
  app.sessionPhase = "active";
  app.exercise = "bicep_curl";

  const lm = createBaseLandmarks();
  // Side view: shoulders nearly overlapping in x
  lm[11] = { x: 0.50, y: 0.30, z: 0, visibility: 0.95 };
  lm[12] = { x: 0.51, y: 0.30, z: 0.05, visibility: 0.30 };
  lm[23] = { x: 0.50, y: 0.65, z: 0, visibility: 0.95 };
  lm[24] = { x: 0.51, y: 0.65, z: 0.05, visibility: 0.30 };

  // 1. Full bottom extension
  lm[13] = { x: 0.50, y: 0.48, z: 0, visibility: 0.95 };
  lm[15] = { x: 0.50, y: 0.66, z: 0, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1000);

  // 2. Curling upward
  lm[15] = { x: 0.58, y: 0.48, z: 0, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1200);

  // 3. Top contraction in side view (elbow pinned at side, wrist flexed up to shoulder level)
  lm[15] = { x: 0.53, y: 0.33, z: 0, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1400);

  // 4. Lowering
  lm[15] = { x: 0.58, y: 0.48, z: 0, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1600);

  // 5. Full bottom extension
  lm[15] = { x: 0.50, y: 0.66, z: 0, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1800);

  assert.equal(app.bicepCurlReps, 1, "Full range side-view curl must count as 1 rep");
});

test("Bicep Curl: starting set with arms flexed at shoulders and lowering does NOT register rep", () => {
  const app = loadAppSandbox();
  app.sessionPhase = "active";
  app.exercise = "bicep_curl";

  const lm = createBaseLandmarks();
  lm[11] = { x: 0.40, y: 0.30, z: 0, visibility: 0.95 };
  lm[12] = { x: 0.60, y: 0.30, z: 0, visibility: 0.95 };
  lm[23] = { x: 0.42, y: 0.65, z: 0, visibility: 0.95 };
  lm[24] = { x: 0.58, y: 0.65, z: 0, visibility: 0.95 };

  // Frame 1: Hands at shoulders (top contraction position)
  lm[13] = { x: 0.40, y: 0.48, z: 0, visibility: 0.95 };
  lm[15] = { x: 0.40, y: 0.32, z: -0.05, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1000);

  // Frame 2: Drops arms to sides (extension at bottom)
  lm[15] = { x: 0.40, y: 0.66, z: 0, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1500);

  assert.equal(app.bicepCurlReps, 0, "Dropping arms from top without curling must NOT count as a rep");
});

test("Bicep Curl: lowering from overhead raise does NOT register rep", () => {
  const app = loadAppSandbox();
  app.sessionPhase = "active";
  app.exercise = "bicep_curl";

  const lm = createBaseLandmarks();
  lm[11] = { x: 0.40, y: 0.35, z: 0, visibility: 0.95 };
  lm[12] = { x: 0.60, y: 0.35, z: 0, visibility: 0.95 };
  lm[23] = { x: 0.42, y: 0.70, z: 0, visibility: 0.95 };
  lm[24] = { x: 0.58, y: 0.70, z: 0, visibility: 0.95 };

  // Hands overhead
  lm[13] = { x: 0.40, y: 0.20, z: 0, visibility: 0.95 };
  lm[15] = { x: 0.40, y: 0.05, z: 0, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1000);

  // Hands brought down past shoulders
  lm[13] = { x: 0.40, y: 0.52, z: 0, visibility: 0.95 };
  lm[15] = { x: 0.40, y: 0.37, z: -0.05, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1300);

  // Hands lowered to sides
  lm[15] = { x: 0.40, y: 0.70, z: 0, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1800);

  assert.equal(app.bicepCurlReps, 0, "Lowering from overhead raise must NOT count as a rep");
});

test("Bicep Curl: front raise with occluded/invisible hips does NOT register rep", () => {
  const app = loadAppSandbox();
  app.sessionPhase = "active";
  app.exercise = "bicep_curl";

  const lm = createBaseLandmarks();
  lm[11] = { x: 0.40, y: 0.30, z: 0, visibility: 0.95 };
  lm[12] = { x: 0.60, y: 0.30, z: 0, visibility: 0.95 };
  // Hips invisible / occluded:
  lm[23] = { x: 0.50, y: 0.50, z: 0, visibility: 0.0 };
  lm[24] = { x: 0.50, y: 0.50, z: 0, visibility: 0.0 };

  // Bottom extension
  lm[13] = { x: 0.40, y: 0.48, z: 0, visibility: 0.95 };
  lm[15] = { x: 0.40, y: 0.65, z: 0, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1000);

  // Front arm raise with invisible hips
  for (let t = 1100; t <= 1600; t += 100) {
    lm[13] = { x: 0.40, y: 0.35, z: -0.15, visibility: 0.95 };
    lm[15] = { x: 0.40, y: 0.30, z: -0.30, visibility: 0.95 };
    app.evaluateBicepCurlLive(lm, t);
  }

  for (let t = 1700; t <= 2200; t += 100) {
    lm[13] = { x: 0.40, y: 0.48, z: 0, visibility: 0.95 };
    lm[15] = { x: 0.40, y: 0.65, z: 0, visibility: 0.95 };
    app.evaluateBicepCurlLive(lm, t);
  }

  assert.equal(app.bicepCurlReps, 0, "Front raise with invisible hips must not count");
});

test("Bicep Curl: reversing upward from lowering phase without reaching bottom counts only 1 full rep", () => {
  const app = loadAppSandbox();
  app.sessionPhase = "active";
  app.exercise = "bicep_curl";

  const lm = createBaseLandmarks();
  lm[11] = { x: 0.40, y: 0.30, z: 0, visibility: 0.95 };
  lm[12] = { x: 0.60, y: 0.30, z: 0, visibility: 0.95 };
  lm[23] = { x: 0.42, y: 0.65, z: 0, visibility: 0.95 };
  lm[24] = { x: 0.58, y: 0.65, z: 0, visibility: 0.95 };

  // 1. Ready at bottom extension
  lm[13] = { x: 0.40, y: 0.48, z: 0, visibility: 0.95 };
  lm[15] = { x: 0.40, y: 0.66, z: 0, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1000);

  // 2. Curl up to top
  lm[15] = { x: 0.40, y: 0.33, z: -0.05, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1400);

  // 3. Lower halfway down (w.y = 0.48, elbow height)
  lm[15] = { x: 0.40, y: 0.48, z: -0.10, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1600);

  // 4. Reverse direction back up to top (without reaching bottom extension)
  lm[15] = { x: 0.40, y: 0.33, z: -0.05, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1800);

  // Still 0 reps at this point
  assert.equal(app.bicepCurlReps, 0, "No rep before full bottom extension");

  // 5. Finally lower all the way to bottom extension
  lm[15] = { x: 0.40, y: 0.66, z: 0, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 2200);

  assert.equal(app.bicepCurlReps, 1, "Exactly 1 rep counted after completing full range of motion");
});

test("Plank: perfect high plank on hands returns state: good", () => {
  const app = loadAppSandbox();
  const lm = createBaseLandmarks();

  // High plank: pushup top position on palms
  lm[11] = { x: 0.25, y: 0.40, z: 0, visibility: 0.95 }; // shoulder (0.40)
  lm[13] = { x: 0.25, y: 0.55, z: 0, visibility: 0.95 }; // elbow in air (0.55)
  lm[15] = { x: 0.25, y: 0.70, z: 0, visibility: 0.95 }; // hands on floor (0.70)
  lm[23] = { x: 0.50, y: 0.50, z: 0, visibility: 0.95 }; // hip (0.50)
  lm[25] = { x: 0.65, y: 0.55, z: 0, visibility: 0.95 }; // knee (0.55)
  lm[27] = { x: 0.80, y: 0.60, z: 0, visibility: 0.95 }; // feet on floor (0.60)

  const res = app.evaluatePlankPosture(lm);
  assert.equal(res.state, "good");
  assert.match(res.cue, /great line/i);
});

test("Plank: high plank with knees on floor triggers broken posture", () => {
  const app = loadAppSandbox();
  const lm = createBaseLandmarks();

  lm[11] = { x: 0.25, y: 0.40, z: 0, visibility: 0.95 }; // shoulder
  lm[13] = { x: 0.25, y: 0.55, z: 0, visibility: 0.95 }; // elbow in air
  lm[15] = { x: 0.25, y: 0.70, z: 0, visibility: 0.95 }; // hands on floor
  lm[23] = { x: 0.50, y: 0.55, z: 0, visibility: 0.95 }; // hip
  lm[25] = { x: 0.65, y: 0.70, z: 0, visibility: 0.95 }; // knee resting on floor at 0.70
  lm[27] = { x: 0.80, y: 0.60, z: 0, visibility: 0.95 }; // feet

  const res = app.evaluatePlankPosture(lm);
  assert.equal(res.state, "broken");
  assert.match(res.cue, /knees on floor/i);
});

test("Plank: exhausted user resting flat on floor with arms forward (e.y = 0.87) triggers broken posture", () => {
  const app = loadAppSandbox();
  const lm = createBaseLandmarks();

  // All body landmarks flat on floor
  lm[11] = { x: 0.25, y: 0.75, z: 0, visibility: 0.95 }; // shoulder flat on floor
  lm[13] = { x: 0.25, y: 0.87, z: 0, visibility: 0.95 }; // elbow resting forward
  lm[15] = { x: 0.35, y: 0.87, z: 0, visibility: 0.95 }; // wrist
  lm[23] = { x: 0.50, y: 0.75, z: 0, visibility: 0.95 }; // hip flat on floor
  lm[25] = { x: 0.65, y: 0.75, z: 0, visibility: 0.95 }; // knee flat on floor
  lm[27] = { x: 0.80, y: 0.75, z: 0, visibility: 0.95 }; // ankle flat on floor

  const res = app.evaluatePlankPosture(lm);
  assert.equal(res.state, "broken");
  assert.match(res.cue, /resting on floor/i);
});

test("Bicep Curl: front cheat curl with elbows swinging forward 70° in 3D does NOT register rep", () => {
  const app = loadAppSandbox();
  app.sessionPhase = "active";
  app.exercise = "bicep_curl";

  const lm = createBaseLandmarks();
  lm[11] = { x: 0.40, y: 0.30, z: 0, visibility: 0.95 }; // left shoulder
  lm[12] = { x: 0.60, y: 0.30, z: 0, visibility: 0.95 }; // right shoulder
  lm[23] = { x: 0.42, y: 0.65, z: 0, visibility: 0.95 }; // left hip
  lm[24] = { x: 0.58, y: 0.65, z: 0, visibility: 0.95 }; // right hip

  // 1. Ready at bottom extension
  lm[13] = { x: 0.40, y: 0.48, z: 0, visibility: 0.95 };
  lm[15] = { x: 0.40, y: 0.66, z: 0, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1000);

  // 2. Swings elbows forward 70° into room in 3D (upper arms raised forward)
  for (let t = 1100; t <= 1500; t += 100) {
    lm[13] = { x: 0.40, y: 0.35, z: -0.16, visibility: 0.95 };
    lm[15] = { x: 0.40, y: 0.32, z: -0.22, visibility: 0.95 };
    app.evaluateBicepCurlLive(lm, t);
  }

  // 3. Lowers back down
  for (let t = 1600; t <= 2000; t += 100) {
    lm[13] = { x: 0.40, y: 0.48, z: 0, visibility: 0.95 };
    lm[15] = { x: 0.40, y: 0.66, z: 0, visibility: 0.95 };
    app.evaluateBicepCurlLive(lm, t);
  }

  assert.equal(app.bicepCurlReps, 0, "Cheat curl with elbows swinging forward in 3D must NOT count as a rep");
});

test("Plank: exhausted user resting flat on floor with ankles slightly elevated triggers broken posture", () => {
  const app = loadAppSandbox();
  const lm = createBaseLandmarks();

  // Torso and thighs resting on mat, ankles slightly elevated (a.y = 0.70 vs body at 0.74-0.75)
  lm[11] = { x: 0.25, y: 0.74, z: 0, visibility: 0.95 }; // shoulder flat on floor
  lm[13] = { x: 0.25, y: 0.82, z: 0, visibility: 0.95 }; // elbow resting forward
  lm[15] = { x: 0.35, y: 0.82, z: 0, visibility: 0.95 }; // wrist
  lm[23] = { x: 0.50, y: 0.75, z: 0, visibility: 0.95 }; // hip flat on floor
  lm[25] = { x: 0.65, y: 0.75, z: 0, visibility: 0.95 }; // knee flat on floor
  lm[27] = { x: 0.80, y: 0.70, z: 0, visibility: 0.95 }; // ankle (feet relaxed with heels up)

  const res = app.evaluatePlankPosture(lm);
  assert.equal(res.state, "broken");
  assert.match(res.cue, /resting on floor/i);
});

test("Bicep Curl: straight-arm lateral raise in front view does NOT register rep", () => {
  const app = loadAppSandbox();
  app.sessionPhase = "active";
  app.exercise = "bicep_curl";

  const lm = createBaseLandmarks();
  lm[11] = { x: 0.40, y: 0.30, z: 0, visibility: 0.95 }; // left shoulder
  lm[12] = { x: 0.60, y: 0.30, z: 0, visibility: 0.95 }; // right shoulder
  lm[23] = { x: 0.42, y: 0.65, z: 0, visibility: 0.95 }; // left hip
  lm[24] = { x: 0.58, y: 0.65, z: 0, visibility: 0.95 }; // right hip

  // 1. Ready at sides
  lm[13] = { x: 0.40, y: 0.48, z: 0, visibility: 0.95 };
  lm[15] = { x: 0.40, y: 0.65, z: 0, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1000);

  // 2. Lateral raise out to sides (elbow and wrist raise laterally to shoulder level)
  for (let t = 1100; t <= 1600; t += 100) {
    lm[13] = { x: 0.22, y: 0.30, z: 0, visibility: 0.95 };
    lm[15] = { x: 0.05, y: 0.30, z: 0, visibility: 0.95 };
    app.evaluateBicepCurlLive(lm, t);
  }

  // 3. Lower back down to sides
  for (let t = 1700; t <= 2200; t += 100) {
    lm[13] = { x: 0.40, y: 0.48, z: 0, visibility: 0.95 };
    lm[15] = { x: 0.40, y: 0.65, z: 0, visibility: 0.95 };
    app.evaluateBicepCurlLive(lm, t);
  }

  assert.equal(app.bicepCurlReps, 0, "Straight arm lateral raise must not register a bicep curl rep");
});

test("Bicep Curl: bent-arm lateral raise (chicken-wing) does NOT register rep", () => {
  const app = loadAppSandbox();
  app.sessionPhase = "active";
  app.exercise = "bicep_curl";

  const lm = createBaseLandmarks();
  lm[11] = { x: 0.40, y: 0.30, z: 0, visibility: 0.95 };
  lm[12] = { x: 0.60, y: 0.30, z: 0, visibility: 0.95 };
  lm[23] = { x: 0.42, y: 0.65, z: 0, visibility: 0.95 };
  lm[24] = { x: 0.58, y: 0.65, z: 0, visibility: 0.95 };

  // Ready at sides
  lm[13] = { x: 0.40, y: 0.48, z: 0, visibility: 0.95 };
  lm[15] = { x: 0.40, y: 0.65, z: 0, visibility: 0.95 };
  app.evaluateBicepCurlLive(lm, 1000);

  // Bent-arm lateral raise: elbows flared high out to sides (upper arm angle ~90° to torso)
  for (let t = 1100; t <= 1600; t += 100) {
    lm[13] = { x: 0.22, y: 0.32, z: 0, visibility: 0.95 }; // elbow lifted laterally
    lm[15] = { x: 0.32, y: 0.32, z: 0, visibility: 0.95 }; // forearm flexed
    app.evaluateBicepCurlLive(lm, t);
  }

  // Lower back down
  for (let t = 1700; t <= 2200; t += 100) {
    lm[13] = { x: 0.40, y: 0.48, z: 0, visibility: 0.95 };
    lm[15] = { x: 0.40, y: 0.65, z: 0, visibility: 0.95 };
    app.evaluateBicepCurlLive(lm, t);
  }

  assert.equal(app.bicepCurlReps, 0, "Bent-arm lateral raise must not register a bicep curl rep");
});


