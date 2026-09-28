#!/usr/bin/env python3
"""
evals/gate0/detector/exercise_signals.py

Per-exercise mapping from "which joint angle drives this exercise's phase machine" to actual
keypoints, and "what counts as full depth" per the exercise's own library entry. This is
structural/algorithmic code (which anatomical landmarks matter for which exercise), not a
tunable numeric threshold -- the numbers themselves (bottom-angle bands, x/z thresholds for
faults) still come from exercises/*.json, never restated here (CLAUDE.md §3).

Scope: squat, pushup, lunge only (ROADMAP.md Stage 1 is scoped to the 3 already-instrumented
exercises; the 11 requested additions are Stage 5, gated one at a time). Extending this table to
a new exercise means adding one PrimaryJoint entry and a bottom-threshold lookup for it -- both
called out explicitly if this raises KeyError for an unscoped exercise, rather than silently
guessing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from detector.geometry import angle_deg
from detector.keypoint_map import get_point


@dataclass(frozen=True)
class PrimaryJoint:
    """The (proximal, vertex, distal) landmark-name triple whose angle at `vertex` drives this
    exercise's phase machine -- e.g. hip-knee-ankle for squat, so the knee angle is the signal."""
    left: Tuple[str, str, str]
    right: Tuple[str, str, str]


PRIMARY_JOINTS: Dict[str, PrimaryJoint] = {
    "squat": PrimaryJoint(
        left=("left_hip", "left_knee", "left_ankle"),
        right=("right_hip", "right_knee", "right_ankle"),
    ),
    "lunge": PrimaryJoint(
        left=("left_hip", "left_knee", "left_ankle"),
        right=("right_hip", "right_knee", "right_ankle"),
    ),
    "pushup": PrimaryJoint(
        left=("left_shoulder", "left_elbow", "left_wrist"),
        right=("right_shoulder", "right_elbow", "right_wrist"),
    ),
    "bicep_curl": PrimaryJoint(
        left=("left_shoulder", "left_elbow", "left_wrist"),
        right=("right_shoulder", "right_elbow", "right_wrist"),
    ),
    "plank": PrimaryJoint(
        left=("left_shoulder", "left_hip", "left_ankle"),
        right=("right_shoulder", "right_hip", "right_ankle"),
    ),
}

# Where each exercise's "correct bottom" angle band lives in its exercises/*.json entry -- the
# three files use two different shapes (CLAUDE.md's "one source of truth per value" holds; this
# is just reading two different existing shapes, not a new inconsistency introduced here).
_BOTTOM_ANGLE_MAX_LOOKUP = {
    # squat.json puts the bottom band under reference_keypoints.correct.key_angles.*_at_bottom.
    "squat": lambda ex: ex["reference_keypoints"]["correct"]["key_angles"][
        "left_knee_angle_at_bottom"
    ]["max"],
    # pushup.json / lunge.json put a single named bottom-max directly under thresholds.
    "pushup": lambda ex: ex["thresholds"]["depth_elbow_angle_max"],
    "lunge": lambda ex: ex["thresholds"]["front_knee_angle_bottom_max"],
    "bicep_curl": lambda ex: ex["reference_keypoints"]["correct"]["key_angles"]["elbow_angle_at_top"]["max"],
    "plank": lambda ex: ex["reference_keypoints"]["correct"]["key_angles"]["body_alignment_angle"]["max"],
}


def scoped_exercises() -> Tuple[str, ...]:
    return tuple(PRIMARY_JOINTS.keys())


def get_bottom_angle_max(exercise_id: str, exercise_json: Dict[str, Any]) -> float:
    """The angle (degrees) at/below which this exercise's primary joint counts as having
    reached full expected depth -- used only for graded-depth scoring, never for gating whether
    a rep counts at all (that's REP_MIN_EXCURSION_DEG in config.py)."""
    try:
        lookup = _BOTTOM_ANGLE_MAX_LOOKUP[exercise_id]
    except KeyError:
        raise KeyError(
            f"exercise_signals is scoped to {scoped_exercises()}; {exercise_id!r} needs its own "
            f"bottom-angle lookup added before Stage 1 can score it"
        ) from None
    return float(lookup(exercise_json))


def _arm_curl_angle(points: Sequence[Any]) -> Optional[float]:
    """Calculates elbow angle for bicep curl across both front and side camera views.
    points: [shoulder, elbow, wrist], each (x, y, z_or_none, vis).
    - If 3D z-depth is populated (e.g. BlazePose), computes true 3D joint angle.
    - If 2D (MoveNet or flat z), supports side view via 2D angle and front view
      via wrist-to-shoulder vertical displacement ratio.
    """
    s, e, w = points[0], points[1], points[2]
    u_len = math.hypot(s[0] - e[0], s[1] - e[1])
    # Reject collapsed/dummy keypoints (e.g. uninitialized landmarks where shoulder and elbow coincide)
    if u_len < 0.01:
        return None

    # 1. 3D joint angle if z is available
    has_3d = any(p[2] is not None and abs(p[2]) > 0.005 for p in (s, e, w))
    if has_3d:
        v1 = (s[0] - e[0], s[1] - e[1], (s[2] or 0.0) - (e[2] or 0.0))
        v2 = (w[0] - e[0], w[1] - e[1], (w[2] or 0.0) - (e[2] or 0.0))
        len1 = math.hypot(v1[0], v1[1], v1[2])
        len2 = math.hypot(v2[0], v2[1], v2[2])
        if len1 > 0 and len2 > 0:
            cos_theta = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (len1 * len2)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            return math.degrees(math.acos(cos_theta))

    # 2. 2D fallback: side view angle
    deg_2d = angle_deg((s[0], s[1]), (e[0], e[1]), (w[0], w[1]))

    # 3. Front view vertical excursion angle
    # Normalized wrist height relative to elbow (h: -1 extended down, +1 curled up at shoulder)
    h = (e[1] - w[1]) / u_len
    norm_h = max(-1.0, min(1.0, h / 0.85))
    front_deg = max(15.0, min(180.0, 95.0 - 85.0 * norm_h))

    if 10.0 < deg_2d < 170.0:
        return min(deg_2d, front_deg)
    return front_deg if abs(s[0] - w[0]) < 0.12 else deg_2d


def primary_angle(person: Dict[str, Any], pose_model: str, exercise_id: str) -> Optional[float]:
    """The exercise's primary joint angle for this person/frame, averaged over whichever side(s)
    have all three landmarks visible enough to compute an angle. For bicep_curl, tracks the
    active curling arm (min angle) to support both single-arm and double-arm curls."""
    joints = PRIMARY_JOINTS.get(exercise_id)
    if joints is None:
        raise KeyError(
            f"exercise_signals is scoped to {scoped_exercises()}; {exercise_id!r} needs a "
            f"PrimaryJoint entry added before Stage 1 can compute its phase signal"
        )
    angles = []
    for triple in (joints.left, joints.right):
        points = [get_point(person, pose_model, name) for name in triple]
        if any(p is None for p in points):
            continue
        if exercise_id == "bicep_curl":
            ang = _arm_curl_angle(points)
            if ang is not None:
                angles.append(ang)
        else:
            a, b, c = ((p[0], p[1]) for p in points)
            angles.append(angle_deg(a, b, c))
    if not angles:
        return None
    if exercise_id == "bicep_curl":
        return min(angles)
    return sum(angles) / len(angles)

