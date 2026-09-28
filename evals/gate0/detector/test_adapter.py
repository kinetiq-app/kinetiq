#!/usr/bin/env python3
"""Unit tests for detector/adapter.py -- the run_detector orchestration."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detector.adapter import (  # noqa: E402
    run_detector,
    score_subject_lock_against_expected,
)
from detector.keypoint_map import POSE_MODEL_LANDMARKS  # noqa: E402

MOVENET = "movenet_17"

FULL_VIS = 0.95
LOW_VIS = 0.05

# A fully-visible squat leg pose. TOP: knee angle 180 (straight). BOTTOM: knee angle 0 (an
# extreme, not biomechanically realistic value, but chosen deliberately so it triggers neither
# knee_cave (ankle.x == knee.x) nor shallow_depth (hip.y > knee.y) -- this test is about
# orchestration wiring, not fault-rule correctness (that's covered by test_faults.py).
_TOP_LEGS = {
    "left_hip": (0.45, 0.5), "left_knee": (0.45, 0.7), "left_ankle": (0.45, 0.9),
    "right_hip": (0.55, 0.5), "right_knee": (0.55, 0.7), "right_ankle": (0.55, 0.9),
}
_BOTTOM_LEGS = {
    "left_hip": (0.45, 0.75), "left_knee": (0.45, 0.7), "left_ankle": (0.45, 0.85),
    "right_hip": (0.55, 0.75), "right_knee": (0.55, 0.7), "right_ankle": (0.55, 0.85),
}


def make_person(track_id, overrides, pose_model=MOVENET, n=17, default_vis=FULL_VIS, vis=FULL_VIS):
    kp = [[0.5, 0.5, None, default_vis] for _ in range(n)]
    for name, (x, y) in overrides.items():
        idx = POSE_MODEL_LANDMARKS[pose_model][name]
        kp[idx] = [x, y, None, vis]
    return {"track_id": track_id, "kp": kp, "box": [0.3, 0.2, 0.4, 0.6]}


def frame(t_ms, people, pose_model=MOVENET):
    return {"t_ms": t_ms, "pose_model": pose_model, "people": people}


def _lerp_overrides(a, b, frac):
    return {
        name: (a[name][0] + (b[name][0] - a[name][0]) * frac,
               a[name][1] + (b[name][1] - a[name][1]) * frac)
        for name in a
    }


def make_squat_rep_frames(track_id, start_t_ms, step_ms=100, steps_down=5):
    """A realistic-shaped descent/hold/ascent across enough frames that the default 5-frame
    smoothing window doesn't wash out the excursion -- 2 * steps_down + 1 frames total."""
    frames = []
    t = start_t_ms
    for i in range(steps_down + 1):
        overrides = _lerp_overrides(_TOP_LEGS, _BOTTOM_LEGS, i / steps_down)
        frames.append(frame(t, [make_person(track_id, overrides)]))
        t += step_ms
    for i in range(1, steps_down + 1):
        overrides = _lerp_overrides(_BOTTOM_LEGS, _TOP_LEGS, i / steps_down)
        frames.append(frame(t, [make_person(track_id, overrides)]))
        t += step_ms
    return frames


class TestPhantomRejection(unittest.TestCase):
    def test_phantom_empty_yields_zero_reps(self):
        frames = [frame(t, []) for t in range(0, 1000, 200)]
        detected = run_detector(frames, "squat")
        self.assertEqual(detected.detected_reps, 0)
        self.assertTrue(all(t is None for t in detected.subject_track_sequence))

    def test_phantom_bench_like_blob_yields_zero_reps(self):
        # A "person" box every frame, but almost nothing visible -- fails the plausibility
        # check regardless of subject-lock (which happily locks onto it, since lock doesn't
        # look at plausibility at all -- see subject_lock.py's module docstring).
        bench = make_person(0, {}, vis=LOW_VIS)
        frames = [frame(t, [bench]) for t in range(0, 1000, 200)]
        detected = run_detector(frames, "squat")
        self.assertEqual(detected.detected_reps, 0)
        # subject-lock DID lock onto it -- the rejection is the plausibility gate's, not lock's.
        self.assertTrue(all(t == 0 for t in detected.subject_track_sequence))


class TestCleanRep(unittest.TestCase):
    def test_single_clean_squat_rep_is_detected(self):
        frames = make_squat_rep_frames(track_id=0, start_t_ms=0)
        detected = run_detector(frames, "squat")
        self.assertEqual(detected.detected_reps, 1)
        self.assertEqual(detected.reps[0].idx, 1)
        self.assertGreaterEqual(detected.reps[0].form_score, 0.0)
        self.assertLessEqual(detected.reps[0].form_score, 10.0)

    def test_two_sequential_clean_squat_reps_are_both_detected(self):
        frames = make_squat_rep_frames(track_id=0, start_t_ms=0)
        frames += make_squat_rep_frames(track_id=0, start_t_ms=frames[-1]["t_ms"] + 100)
        detected = run_detector(frames, "squat")
        self.assertEqual(detected.detected_reps, 2)
        self.assertEqual([r.idx for r in detected.reps], [1, 2])

    def test_frames_total_matches_input_length(self):
        top = make_person(0, _TOP_LEGS)
        frames = [frame(t, [top]) for t in range(0, 500, 100)]
        detected = run_detector(frames, "squat")
        self.assertEqual(detected.frames_total, len(frames))


class TestNearestValidFrameFallback(unittest.TestCase):
    def test_faults_still_evaluated_when_the_exact_minimum_frame_is_a_gap(self):
        # A brief implausible frame (e.g. motion blur) sits exactly at the deepest point of the
        # rep. The smoothed signal still reports its minimum there (filled in from neighbours by
        # the median filter), but there's no valid person record at that exact timestamp -- the
        # adapter must fall back to the nearest valid one rather than silently producing no rep
        # at all / no flags for it.
        frames = make_squat_rep_frames(track_id=0, start_t_ms=0)
        mid = len(frames) // 2
        blurry_t = frames[mid]["t_ms"]
        frames[mid] = frame(blurry_t, [make_person(0, {}, vis=LOW_VIS)])  # fails plausibility

        detected = run_detector(frames, "squat")
        self.assertEqual(detected.detected_reps, 1)
        self.assertGreater(detected.reps[0].form_score, 0.0)


class TestDeterminism(unittest.TestCase):
    def test_same_input_yields_identical_output(self):
        frames = make_squat_rep_frames(track_id=0, start_t_ms=0)
        first = run_detector(frames, "squat").to_dict()
        second = run_detector(frames, "squat").to_dict()
        self.assertEqual(first, second)
        self.assertEqual(first["detected_reps"], 1)


class TestUnscopedExercise(unittest.TestCase):
    def test_unknown_exercise_raises(self):
        with self.assertRaises(KeyError):
            run_detector([frame(0, [])], "deadlift")


class TestScoreSubjectLockAgainstExpected(unittest.TestCase):
    def test_matches_expected_track_id(self):
        top = make_person(0, _TOP_LEGS)
        frames = [frame(t, [top]) for t in range(0, 500, 100)]
        detected = run_detector(frames, "squat")
        stats = score_subject_lock_against_expected(detected, expected_track_id=0)
        self.assertEqual(stats["frames_on_expected_subject"], stats["frames_total"])

    def test_wrong_locked_identity_shows_up_as_a_failure(self):
        # detector locks onto track_id 5 (the only person), but the label says the expected user
        # was track_id 0 -- e.g. SUBJECT_SELECTION_RULE picked the wrong person.
        person = make_person(5, _TOP_LEGS)
        frames = [frame(t, [person]) for t in range(0, 500, 100)]
        detected = run_detector(frames, "squat")
        stats = score_subject_lock_against_expected(detected, expected_track_id=0)
        self.assertEqual(stats["frames_on_expected_subject"], 0)

    def test_single_person_clip_with_no_expected_id_is_trivially_full(self):
        top = make_person(0, _TOP_LEGS)
        frames = [frame(t, [top]) for t in range(0, 500, 100)]
        detected = run_detector(frames, "squat")
        stats = score_subject_lock_against_expected(detected, expected_track_id=None)
        self.assertEqual(stats["frames_on_expected_subject"], stats["frames_total"])


class TestBicepCurlRepCounts(unittest.TestCase):
    def test_bicep_curl_rep_counts_cleanly(self):
        # 1 rep: arm extended (0-200ms), curling up to shoulder (300-600ms), returning down (700-1000ms)
        frames = []
        # Arm positions
        base_body = {
            "left_hip": (0.45, 0.5), "left_knee": (0.45, 0.7), "left_ankle": (0.45, 0.9),
            "right_hip": (0.55, 0.5), "right_knee": (0.55, 0.7), "right_ankle": (0.55, 0.9),
            "left_shoulder": (0.4, 0.3), "left_elbow": (0.4, 0.55),
            "right_shoulder": (0.6, 0.3), "right_elbow": (0.6, 0.55), "right_wrist": (0.6, 0.75),
        }
        # Start at rest: left wrist at (0.4, 0.75)
        for t in range(0, 300, 50):
            p = make_person(0, {**base_body, "left_wrist": (0.4, 0.75)})
            frames.append(frame(t, [p]))
        # Ascend (curl up towards shoulder): wrist moves 0.75 -> 0.32
        for i, t in enumerate(range(300, 650, 50)):
            frac = (i + 1) / 7.0
            wy = 0.75 - frac * (0.75 - 0.32)
            p = make_person(0, {**base_body, "left_wrist": (0.4, wy)})
            frames.append(frame(t, [p]))
        # Descend (return to rest): wrist moves 0.32 -> 0.75
        for i, t in enumerate(range(650, 1100, 50)):
            frac = (i + 1) / 9.0
            wy = 0.32 + frac * (0.75 - 0.32)
            p = make_person(0, {**base_body, "left_wrist": (0.4, wy)})
            frames.append(frame(t, [p]))
        # Hold at rest
        for t in range(1100, 1350, 50):
            p = make_person(0, {**base_body, "left_wrist": (0.4, 0.75)})
            frames.append(frame(t, [p]))

        detected = run_detector(frames, "bicep_curl")
        self.assertEqual(detected.detected_reps, 1)
        self.assertGreater(detected.reps[0].form_score, 5.0)


if __name__ == "__main__":
    unittest.main()

