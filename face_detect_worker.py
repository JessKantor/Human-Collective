#!/usr/bin/env python3
"""
Isolated face-detection worker.

Runs MediaPipe in a separate process so a native crash (segfault) cannot
take down the main database builder. Prints JSON to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Avoid matplotlib cache issues when mediapipe pulls it in as a dependency.
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_PATH = Path("face_landmarker.task")

# Landmarks needed by build_database.classify_face_shape
LANDMARK_INDICES = (
    10, 152, 234, 454, 67, 297, 172, 397,
)


def prepare_image_for_detection(image_bgr: Any) -> Any:
    height, width = image_bgr.shape[:2]
    min_dim = min(height, width)
    if min_dim >= 400:
        return image_bgr
    scale = 400 / min_dim
    return cv2.resize(
        image_bgr,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_CUBIC,
    )


def detect(image_path: Path) -> dict[str, Any]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return {"ok": False, "reason": "unreadable_image"}

    image_bgr = prepare_image_for_detection(image_bgr)
    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    options = vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH),
            delegate=mp_tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=5,
        min_face_detection_confidence=0.35,
        min_face_presence_confidence=0.35,
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return {"ok": False, "reason": "no_face"}

    if len(result.face_landmarks) == 1:
        chosen = result.face_landmarks[0]
    else:

        def face_area(landmarks: list[Any]) -> float:
            xs = [lm.x * width for lm in landmarks]
            ys = [lm.y * height for lm in landmarks]
            return (max(xs) - min(xs)) * (max(ys) - min(ys))

        chosen = max(result.face_landmarks, key=face_area)

    landmarks = {
        str(index): {"x": chosen[index].x, "y": chosen[index].y, "z": chosen[index].z}
        for index in LANDMARK_INDICES
    }
    return {
        "ok": True,
        "width": width,
        "height": height,
        "landmarks": landmarks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a JPEG/PNG image file")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(json.dumps({"ok": False, "reason": "missing_image"}))
        return 1
    if not MODEL_PATH.exists():
        print(json.dumps({"ok": False, "reason": "missing_model"}))
        return 1

    payload = detect(image_path)
    print(json.dumps(payload))
    return 0 if payload.get("ok") or payload.get("reason") == "no_face" else 1


if __name__ == "__main__":
    raise SystemExit(main())
