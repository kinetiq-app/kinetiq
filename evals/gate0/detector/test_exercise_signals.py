#!/usr/bin/env python3
"""Unit tests for detector/exercise_signals.py, including against the real exercises/*.json."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gate_config  # noqa: E402
from detector.exercise_signals import get_bottom_angle_max, primary_angle  # noqa: E402
from detector.keypoint_map import POSE_MODEL_LANDMARKS  # noqa: E402

MOVENET = "movenet_17"


def make_person(overrides, n=17, default_vis=0.9):
    kp = [[0.5, 0.5, None, default_vis] for _ in range(n)]
    for name, (x, y) in overrides.items():
        idx = POSE_MODEL_LANDMARKS[MOVENET][name]
        kp[idx] = [x, y, None, default_vis]
    return {"track_id": 0, "kp": kp, "box": [0.3, 0.2, 0.4, 0.6]}


class TestGetBottomAngleMax(unittest.TestCase):
    """Runs against the real exercise library -- confirms both JSON shapes (squat's nested
    key_angles vs pushup/lunge's flat thresholds) are read correctly."""

    def setUp(self):
        self.library = gate_config.load_exercise_library()

    def test_squat_reads_from_key_angles(self):
        self.assertEqual(get_bottom_angle_max("squat", self.library["squat"]), 100.0)

    def test_pushup_reads_from_thresholds(self):
        self.assertEqual(get_bottom_angle_max("pushup", self.library["pushup"]), 95.0)

    def test_lunge_reads_from_thresholds(self):
        self.assertEqual(get_bottom_angle_max("lunge", self.library["lunge"]), 100.0)

    def test_unscoped_exercise_raises_with_a_clear_message(self):
        with self.assertRaises(KeyError):
            get_bottom_angle_max("deadlift", {})


class TestPrimaryAngle(unittest.TestCase):
    def test_straight_leg_is_near_180_for_squat(self):
        person = make_person({
            "left_hip": (0.5, 0.4), "left_knee": (0.5, 0.6), "left_ankle": (0.5, 0.8),
            "right_hip": (0.55, 0.4), "right_knee": (0.55, 0.6), "right_ankle": (0.55, 0.8),
        })
        angle = primary_angle(person, MOVENET, "squat")
        self.assertAlmostEqual(angle, 180.0, places=1)

    def test_bent_knee_is_well_under_180_for_squat(self):
        person = make_person({
            "left_hip": (0.5, 0.4), "left_knee": (0.5, 0.6), "left_ankle": (0.6, 0.55),
            "right_hip": (0.55, 0.4), "right_knee": (0.55, 0.6), "right_ankle": (0.65, 0.55),
        })
        angle = primary_angle(person, MOVENET, "squat")
        self.assertLess(angle, 150.0)

    def test_pushup_uses_elbow_not_knee(self):
        person = make_person({
            "left_shoulder": (0.5, 0.3), "left_elbow": (0.5, 0.5), "left_wrist": (0.5, 0.7),
            "right_shoulder": (0.55, 0.3), "right_elbow": (0.55, 0.5), "right_wrist": (0.55, 0.7),
        })
        self.assertAlmostEqual(primary_angle(person, MOVENET, "pushup"), 180.0, places=1)

    def test_missing_landmarks_returns_none(self):
        person = {"track_id": 0, "kp": [], "box": [0, 0, 1, 1]}
        self.assertIsNone(primary_angle(person, MOVENET, "squat"))

    def test_bicep_curl_side_view(self):
        # Extended arm: near 180 deg
        extended = make_person({
            "left_shoulder": (0.5, 0.3), "left_elbow": (0.5, 0.5), "left_wrist": (0.5, 0.7),
        })
        self.assertGreater(primary_angle(extended, MOVENET, "bicep_curl"), 150.0)

        # Curled arm in side view (forearm angled up towards shoulder)
        curled = make_person({
            "left_shoulder": (0.5, 0.3), "left_elbow": (0.5, 0.5), "left_wrist": (0.62, 0.35),
        })
        self.assertLess(primary_angle(curled, MOVENET, "bicep_curl"), 70.0)

    def test_bicep_curl_front_view(self):
        # Extended arm in front view: fist below elbow
        extended = make_person({
            "left_shoulder": (0.4, 0.3), "left_elbow": (0.4, 0.55), "left_wrist": (0.4, 0.75),
            "right_shoulder": (0.6, 0.3), "right_elbow": (0.6, 0.55), "right_wrist": (0.6, 0.75),
        })
        self.assertGreater(primary_angle(extended, MOVENET, "bicep_curl"), 140.0)

        # Curled arm in front view: fist reaches shoulder level
        curled = make_person({
            "left_shoulder": (0.4, 0.3), "left_elbow": (0.4, 0.55), "left_wrist": (0.4, 0.32),
            "right_shoulder": (0.6, 0.3), "right_elbow": (0.6, 0.55), "right_wrist": (0.6, 0.75),
        })
        self.assertLess(primary_angle(curled, MOVENET, "bicep_curl"), 60.0)

    def test_unscoped_exercise_raises(self):
        person = make_person({})
        with self.assertRaises(KeyError):
            primary_angle(person, MOVENET, "deadlift")


if __name__ == "__main__":
    unittest.main()
