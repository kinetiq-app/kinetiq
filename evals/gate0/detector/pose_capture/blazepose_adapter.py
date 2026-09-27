#!/usr/bin/env python3
"""
evals/gate0/detector/pose_capture/blazepose_adapter.py

BlazePose-33 capture adapter (config.POSE_MODEL_CANDIDATES "blazepose_33"), via MediaPipe's
Pose Landmarker task. Landmark index -> name mapping lives in detector/keypoint_map.py, NOT
here -- this module only runs the model and emits its raw 33-point output in MediaPipe's
standard order; keypoint_map.py is what makes that order legible to the rest of the harness.

LIVE-TESTED (pose_capture/README.md): confirmed end-to-end against a real video on
mediapipe 1.0.1, using the `pose_landmarker_full` (float16) .task model
(https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task).
`PoseLandmarkerOptions` accepts `num_poses` (default 1, single-person) -- ADR-300 requires this
set >1 so subject-lock has more than one candidate to choose among; verified live that
`num_poses=2` runs without error and `pose_landmarks` (frame-normalized, see below) populate
x/y/z/visibility for every returned pose. The model .task file is resolved via
`KINETIQ_POSE_LANDMARKER_PATH` (env var) so a capture machine doesn't have to keep the file next
to the video; falls back to a `pose_landmarker.task` in the current working directory if unset.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterator

from detector.pose_capture.base import PoseCaptureAdapter, PoseRuntimeUnavailable

# ADR-300: natively multi-person via num_poses > 1 (no separate person_detector needed).
_NUM_POSES = 2


class BlazePoseAdapter(PoseCaptureAdapter):
    pose_model_name = "blazepose_33"

    def is_available(self) -> bool:
        try:
            import cv2  # noqa: F401
            import mediapipe  # noqa: F401
        except ImportError:
            return False
        model_path = os.environ.get("KINETIQ_POSE_LANDMARKER_PATH", "pose_landmarker.task")
        return Path(model_path).is_file()

    def install_hint(self) -> str:
        return (
            "pip install mediapipe opencv-python; download a Pose Landmarker .task model "
            "(https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker#models) and "
            "point KINETIQ_POSE_LANDMARKER_PATH at it (or leave it next to the cwd as "
            "pose_landmarker.task); then run this adapter for real."
        )

    def infer_frames(self, video_path: Path) -> Iterator[Dict[str, Any]]:
        if not self.is_available():
            raise PoseRuntimeUnavailable(f"{self.pose_model_name}: {self.install_hint()}")

        import cv2
        import mediapipe as mp

        model_path = os.environ.get("KINETIQ_POSE_LANDMARKER_PATH", "pose_landmarker.task")
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=_NUM_POSES,
        )
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        try:
            with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
                frame_idx = 0
                while True:
                    ok, image = cap.read()
                    if not ok:
                        break
                    t_ms = int(frame_idx * 1000 / fps)
                    # cv2 reads BGR; mp.Image declared SRGB needs an explicit conversion or every
                    # frame is fed to the model with red/blue channels swapped (caught live this
                    # session -- the original sketch passed raw BGR `image` here).
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = landmarker.detect_for_video(mp_image, t_ms)

                    people = []
                    # `pose_landmarks` (not `pose_world_landmarks`): frame-normalized [0, 1]
                    # x/y matching EVAL_HARNESS_STAGE0_SPEC.md's schema and subject_lock.py's
                    # "most_central" rule (distance to frame centre (0.5, 0.5)) -- caught live
                    # this session that the original sketch used `pose_world_landmarks` (real-
                    # world meters, hip-centred, can be negative/unbounded), which would silently
                    # break subject-lock's centroid math. `pose_landmarks` still carries a z (§2:
                    # "3D where available"), just relative rather than metric.
                    for track_id, pose in enumerate(result.pose_landmarks or []):
                        kp = [[lm.x, lm.y, lm.z, lm.visibility] for lm in pose]
                        xs = [lm.x for lm in pose]
                        ys = [lm.y for lm in pose]
                        box = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                        people.append({"track_id": track_id, "kp": kp, "box": box})

                    yield {"t_ms": t_ms, "pose_model": self.pose_model_name, "people": people}
                    frame_idx += 1
        finally:
            cap.release()
