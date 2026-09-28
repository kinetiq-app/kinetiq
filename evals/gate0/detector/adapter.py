#!/usr/bin/env python3
"""
evals/gate0/detector/adapter.py

The `run_detector(keypoints_stream, exercise_id, config) -> DetectedClip` adapter
(EVAL_HARNESS_STAGE0_SPEC.md §5, deferred to Stage 1 by §12). Orchestrates the specialists in
VISION_ARCHITECTURE.md order:

  1. subject-lock (Stage 1)          -- who is the subject, per frame, or None/paused.
  2. rep-validity gate (Stage 3)     -- human-plausibility per frame; a locked-but-implausible
                                         frame (e.g. a bench a naive detector boxed as a person)
                                         contributes no signal, regardless of lock status.
  3. rep counter                     -- valley-detection FSM over the surviving angle signal
                                         (smoothing/hysteresis/tempo/graded-depth -- ROADMAP.md's
                                         Stage 3, built in Stage 1; see rep_counter.py, untouched
                                         here).
  4. deterministic form-flag rules,  -- detector/faults.py's per-frame rules are unchanged
     SUSTAINED across the rep's         "as-is"; detector/flag_hysteresis.py (Stage 3, SPRINT.md
     window (Stage 3)                   G2) is what changed: a per-frame flag only commits to
                                         the rep once it held true for a large enough fraction of
                                         the rep's evaluable frames, visibility-gated per flag.
                                         Rep-aggregate flags (shallow_pushup/shallow_lunge) are
                                         evaluated once against the rep's overall min angle, as
                                         in Stage 1 -- they have no per-frame landmark evidence to
                                         sustain (see flag_hysteresis.py's module docstring).
  5. deterministic safety veto (5b)  -- a rep the validity gate rejected can never be
                                         resurrected by anything downstream; there is nothing
                                         downstream of the validity gate that CAN override it,
                                         which is the veto property by construction, not an
                                         extra check bolted on. Flag hysteresis (step 4) only
                                         governs WHEN a flag first commits, never un-commits one.

run_detector itself never sees ground truth (no labels, no "expected subject") -- it can only
know what a real detector could know: the keypoints stream, the exercise, and config. Comparing
its chosen subject identity against a golden clip's labeled "expected" subject is a *grading*
step and deliberately lives outside this function (see score_subject_lock_against_expected
below) -- smuggling the answer key into the detector would make the eval meaningless.

Determinism: every step here is a pure function of its inputs (no randomness, no wall-clock, no
I/O) -- the same keypoints_stream always produces the same DetectedClip.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import exercise_lib
import gate_config
from detector.exercise_signals import get_bottom_angle_max, primary_angle, scoped_exercises
from detector.faults import shallow_lunge_present, shallow_pushup_present
from detector.flag_hysteresis import (
    EXERCISE_SUSTAINED_FLAG_IDS,
    FlagHysteresisConfig,
    FlagOutcome,
    evaluate_sustained_flag,
)
from detector.plausibility import PlausibilityConfig, is_plausible_human
from detector.rep_counter import RepCounterConfig, count_reps_with_state
from detector.subject_lock import track_subject

DETECTOR_VERSION = "kinetiq-v3-stage1-reference"


@dataclass(frozen=True)
class DetectorConfig:
    """Bundles every Stage-1 tunable in one place so run_detector's `config` parameter can
    override behaviour end-to-end (e.g. for a threshold-sensitivity sweep) without touching
    global config.py. Every field defaults from config.py via gate_config."""
    subject_selection_rule: str = gate_config.SUBJECT_SELECTION_RULE
    subject_lost_frames_threshold: int = gate_config.SUBJECT_LOST_FRAMES_THRESHOLD
    subject_reid_max_centroid_dist: float = gate_config.SUBJECT_REID_MAX_CENTROID_DIST
    plausibility: PlausibilityConfig = field(default_factory=PlausibilityConfig)
    rep_counter: RepCounterConfig = field(default_factory=RepCounterConfig)
    flag_hysteresis: FlagHysteresisConfig = field(default_factory=FlagHysteresisConfig)
    graded_depth_form_score_floor: float = gate_config.GRADED_DEPTH_FORM_SCORE_FLOOR
    form_score_penalty_per_flag: float = gate_config.FORM_SCORE_PENALTY_PER_FLAG


@dataclass
class DetectedRep:
    idx: int
    flags: List[str]
    form_score: float
    # Sustained flags (Stage 3) that couldn't be judged either way this rep -- too few
    # visibility-passing frames to say PRESENT or ABSENT with any confidence (SPRINT.md G2:
    # surfaced, never silently dropped). Empty for the common case of enough evidence either way.
    insufficient_evidence: List[str] = field(default_factory=list)


@dataclass
class DetectedClip:
    detected_reps: int
    reps: List[DetectedRep]
    frames_total: int
    subject_track_sequence: List[Optional[int]]  # per-frame locked track_id, or None
    coaching_cues: List[Dict[str, Any]]
    # Live/trailing rep-counter state as of the last frame (detector/rep_counter.py's
    # count_reps_with_state) -- a frozen-clip eval consumer has no use for "what's happening
    # right now" (the clip is already over), but a live session does; see prototype_api/.
    phase: str = "top"
    rep_in_progress: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected_reps": self.detected_reps,
            "reps": [
                {
                    "idx": r.idx,
                    "flags": r.flags,
                    "form_score": r.form_score,
                    "insufficient_evidence": r.insufficient_evidence,
                }
                for r in self.reps
            ],
            "coaching_cues": self.coaching_cues,
            "phase": self.phase,
            "rep_in_progress": self.rep_in_progress,
        }


def _rep_aggregate_flags(exercise_id: str, exercise_json: Dict[str, Any], rep_min_angle: float) -> List[str]:
    """shallow_pushup / shallow_lunge -- compared against the rep's overall smoothed minimum
    angle, not any single frame's landmarks. No hysteresis, no visibility gate: see
    flag_hysteresis.py's module docstring for why these two don't go through it."""
    if exercise_id == "pushup":
        return ["shallow_pushup"] if shallow_pushup_present(rep_min_angle, exercise_json) else []
    if exercise_id == "lunge":
        return ["shallow_lunge"] if shallow_lunge_present(rep_min_angle, exercise_json) else []
    if exercise_id in ("squat", "bicep_curl", "plank"):
        return []  # squat, bicep_curl, plank have no rep-aggregate flags
    raise KeyError(
        f"detector.faults is scoped to {scoped_exercises()}; {exercise_id!r} needs its own "
        f"fault evaluator added before Stage 1 can score its form"
    )


def _fault_phases(exercise_json: Dict[str, Any], fault_id: str) -> Tuple[str, ...]:
    """The phases (top/descending/bottom/ascending) this fault's rule declares itself relevant to
    -- read from the exercise library's own common_errors[].keypoint_signature.phase_detected_in
    (e.g. squat.json: shallow_depth is ["bottom"] only, but knee_cave is
    ["descending","ascending","bottom"]), never re-invented here. Defaults to every phase if the
    library doesn't declare one, so an undeclared fault is never silently narrowed."""
    for err in exercise_json.get("reference_keypoints", {}).get("common_errors", []):
        if err.get("error_id") == fault_id:
            phases = err.get("keypoint_signature", {}).get("phase_detected_in")
            if phases:
                return tuple(phases)
    return ("top", "descending", "bottom", "ascending")


def _is_bottom_phase_only(exercise_json: Dict[str, Any], fault_id: str) -> bool:
    phases = _fault_phases(exercise_json, fault_id)
    return "descending" not in phases and "ascending" not in phases


def _evaluate_rep_flags(
    exercise_id: str,
    exercise_json: Dict[str, Any],
    rep_min_angle: float,
    top_ref_deg: float,
    window: List[Tuple[int, float, Dict[str, Any], str]],
    severities: Dict[str, str],
    hysteresis_config: FlagHysteresisConfig,
) -> Tuple[List[str], List[str]]:
    """Returns (flags, insufficient_evidence) for one rep. flags: committed fault ids (sustained
    per-frame flags that cleared their severity's required fraction, plus any rep-aggregate
    flags). insufficient_evidence: sustained flags that couldn't be judged either way this rep --
    surfaced, never silently dropped (SPRINT.md G2).

    window: every (t_ms, angle, person, pose_model) in the rep's [start_ms, end_ms] span. A
    fault whose rule is declared bottom-phase-only (_is_bottom_phase_only) is evaluated only over
    the frames within the bottom `bottom_phase_fraction` of THIS REP'S OWN excursion range
    (top_ref_deg down to rep_min_angle) -- self-relative, not gated on the exercise's
    correct-depth threshold (config.py's FLAG_BOTTOM_PHASE_FRACTION explains why: a partial-depth
    rep still has a genuine bottom). Otherwise a genuinely clean, full-depth rep would spend much
    of its descent/ascent legitimately NOT yet at depth, which a whole-rep sustained-fraction
    check would misread as "shallow" (not a hypothetical -- exactly what happened on the
    synthetic golden fixture before this narrowing was added; see the test that pins it). Faults
    relevant across the whole rep (knee_cave, elbow_flare, hip_sag) use the full window unchanged.
    """
    thresholds = exercise_json.get("thresholds", {})
    full_frames = [(person, pose_model) for _, _, person, pose_model in window]
    excursion = top_ref_deg - rep_min_angle
    bottom_ceiling = rep_min_angle + excursion * hysteresis_config.bottom_phase_fraction
    bottom_frames = [
        (person, pose_model) for _, angle, person, pose_model in window if angle <= bottom_ceiling
    ]

    flags: List[str] = []
    insufficient: List[str] = []

    for fault_id in EXERCISE_SUSTAINED_FLAG_IDS.get(exercise_id, ()):
        severity = severities.get(fault_id)
        if severity is None:
            # A fault the exercise library doesn't declare a severity for shouldn't happen for a
            # scoped exercise (exercise_lib validates this at load) -- skip rather than guess,
            # rather than silently asserting a flag with no severity to gate its strictness.
            continue
        frames = bottom_frames if _is_bottom_phase_only(exercise_json, fault_id) else full_frames
        outcome = evaluate_sustained_flag(fault_id, frames, thresholds, severity, hysteresis_config)
        if outcome is FlagOutcome.PRESENT:
            flags.append(fault_id)
        elif outcome is FlagOutcome.INSUFFICIENT_EVIDENCE:
            insufficient.append(fault_id)

    flags.extend(_rep_aggregate_flags(exercise_id, exercise_json, rep_min_angle))
    return flags, insufficient


def _window_frames(
    frame_lookup: Dict[int, Tuple[Dict[str, Any], str]],
    angle_lookup: Dict[int, float],
    start_ms: int,
    end_ms: int,
) -> List[Tuple[int, float, Dict[str, Any], str]]:
    """Every (t_ms, angle, person, pose_model) in frame_lookup/angle_lookup (already restricted
    to locked+plausible+angle-computable frames) whose timestamp falls within the rep's
    [start_ms, end_ms] window -- what detector/flag_hysteresis.py evaluates a fault's sustained
    fraction over (via _evaluate_rep_flags). A timestamp with no entry (occlusion, implausible,
    or angle not computable that frame) simply contributes nothing, in either direction --
    exactly the "not evidence either way" behaviour flag_hysteresis.py's visibility gate already
    has, for the same reason."""
    return [
        (t, angle_lookup[t], *frame_lookup[t])
        for t in sorted(frame_lookup)
        if start_ms <= t <= end_ms
    ]


def _depth_form_score(
    top_ref_deg: float, min_angle_deg: float, bottom_max_deg: float, config: DetectorConfig
) -> float:
    """Graded depth (EVAL_STRATEGY.md case #5): a countable rep always scores somewhere in
    [floor, 10] based on how close it got to the exercise's own correct-bottom band, never 0 just
    for being shallow -- a fault-flag penalty is applied separately, on top of this."""
    span = top_ref_deg - bottom_max_deg
    if span <= 0:
        depth_fraction = 1.0
    else:
        depth_fraction = (top_ref_deg - min_angle_deg) / span
    depth_fraction = max(0.0, min(1.0, depth_fraction))
    floor = config.graded_depth_form_score_floor
    return floor + (10.0 - floor) * depth_fraction


def run_detector(
    keypoints_stream: Sequence[Dict[str, Any]],
    exercise_id: str,
    config: Optional[DetectorConfig] = None,
) -> DetectedClip:
    """keypoints_stream: parsed keypoints.jsonl frames, in ascending t_ms order (each a dict with
    t_ms/pose_model/people, per EVAL_HARNESS_STAGE0_SPEC.md §5). Deterministic: same input,
    same output."""
    cfg = config or DetectorConfig()
    frames = list(keypoints_stream)
    frames_total = len(frames)

    exercise_json = gate_config.load_exercise_library().get(exercise_id)
    if exercise_json is None:
        raise KeyError(f"{exercise_id!r} is not in the exercise library")

    locks = track_subject(
        frames,
        selection_rule=cfg.subject_selection_rule,
        lost_frames_threshold=cfg.subject_lost_frames_threshold,
        reid_max_centroid_dist=cfg.subject_reid_max_centroid_dist,
    )

    subject_track_sequence: List[Optional[int]] = []
    samples: List[tuple] = []  # (t_ms, angle_or_None)
    frame_lookup: Dict[int, tuple] = {}  # t_ms -> (person, pose_model)
    angle_lookup: Dict[int, float] = {}  # t_ms -> raw primary angle, same keys as frame_lookup

    for frame, lock in zip(frames, locks):
        subject_track_sequence.append(lock.track_id if lock.person is not None else None)

        t_ms = frame["t_ms"]
        pose_model = frame.get("pose_model")
        angle: Optional[float] = None
        if lock.person is not None and is_plausible_human(lock.person, pose_model, cfg.plausibility):
            angle = primary_angle(lock.person, pose_model, exercise_id)
            if angle is not None:
                frame_lookup[t_ms] = (lock.person, pose_model)
                angle_lookup[t_ms] = angle
        samples.append((t_ms, angle))

    counter_result = count_reps_with_state(samples, cfg.rep_counter)
    rep_events = counter_result.events
    bottom_max_deg = get_bottom_angle_max(exercise_id, exercise_json)
    severities = exercise_lib.load_fault_severities().get(exercise_id, {})

    reps: List[DetectedRep] = []

    for event in rep_events:
        window = _window_frames(frame_lookup, angle_lookup, event.start_ms, event.end_ms)
        flags, insufficient = _evaluate_rep_flags(
            exercise_id, exercise_json, event.min_angle_deg, event.top_ref_deg, window,
            severities, cfg.flag_hysteresis,
        )
        score = _depth_form_score(event.top_ref_deg, event.min_angle_deg, bottom_max_deg, cfg)
        score -= cfg.form_score_penalty_per_flag * len(flags)
        score = max(0.0, min(10.0, score))
        reps.append(DetectedRep(
            idx=event.idx, flags=flags, form_score=round(score, 1), insufficient_evidence=insufficient
        ))

    return DetectedClip(
        detected_reps=len(reps),
        reps=reps,
        frames_total=frames_total,
        subject_track_sequence=subject_track_sequence,
        # Coaching-cue generation is VISION_ARCHITECTURE.md Stage 6 (the coaching generator),
        # not Stage 1 -- always empty here rather than reusing the exercise library's flag_cues
        # text (which isn't guaranteed to fit the Stage-0 cue-length assertion and isn't this
        # detector's job to author or select).
        coaching_cues=[],
        phase=counter_result.phase,
        rep_in_progress=counter_result.rep_in_progress,
    )


def score_subject_lock_against_expected(
    detected: DetectedClip, expected_track_id: Optional[int]
) -> Dict[str, int]:
    """Harness-side grading only (EVAL_HARNESS_STAGE0_SPEC.md's detected.json subject_lock
    field): compares the detector's autonomously-chosen per-frame identity against the golden
    label's ground-truth subject_track_id. NOT part of run_detector -- a real detector has no
    "expected" identity to compare against, only the one it locked onto. If our
    SUBJECT_SELECTION_RULE picks the wrong person (e.g. a bystander with a bigger box), this is
    exactly where that shows up as a subject-lock failure."""
    frames_total = len(detected.subject_track_sequence)
    if expected_track_id is None:
        return {"frames_total": frames_total, "frames_on_expected_subject": frames_total}
    on_expected = sum(1 for t in detected.subject_track_sequence if t == expected_track_id)
    return {"frames_total": frames_total, "frames_on_expected_subject": on_expected}
