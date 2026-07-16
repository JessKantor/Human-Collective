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

import face_validation as fv
from face_validation import (
    face_area_from_landmarks,
    validate_face_landmarks,
)

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


def choose_dominant_face(
    faces: list[list[Any]], width: int, height: int
) -> tuple[list[Any], int, float]:
    """
    Pick the largest face and compute its share of total detected face area.
    """
    if len(faces) == 1:
        return faces[0], 1, 1.0

    areas = [face_area_from_landmarks(face, width, height) for face in faces]
    total_area = sum(areas) or 1.0
    best_index = max(range(len(faces)), key=lambda i: areas[i])
    dominant_ratio = areas[best_index] / total_area
    return faces[best_index], len(faces), dominant_ratio


def detect(image_path: Path, *, artwork: bool = False) -> dict[str, Any]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return {"ok": False, "reason": "unreadable_image"}

    image_bgr = prepare_image_for_detection(image_bgr)
    height, width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    det_conf = (
        fv.ARTWORK_MIN_FACE_DETECTION_CONFIDENCE
        if artwork
        else fv.MIN_FACE_DETECTION_CONFIDENCE
    )
    pres_conf = (
        fv.ARTWORK_MIN_FACE_PRESENCE_CONFIDENCE
        if artwork
        else fv.MIN_FACE_PRESENCE_CONFIDENCE
    )

    options = vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH),
            delegate=mp_tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=5,
        min_face_detection_confidence=det_conf,
        min_face_presence_confidence=pres_conf,
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return {"ok": False, "reason": "no_face"}

    chosen, face_count, dominant_ratio = choose_dominant_face(
        result.face_landmarks, width, height
    )

    ok, rejection_reason, validation = validate_face_landmarks(
        chosen,
        width,
        height,
        face_count=face_count,
        dominant_face_ratio=dominant_ratio,
        artwork=artwork,
    )

    landmarks = {
        str(index): {"x": chosen[index].x, "y": chosen[index].y, "z": chosen[index].z}
        for index in LANDMARK_INDICES
    }

    payload: dict[str, Any] = {
        "ok": ok,
        "width": width,
        "height": height,
        "landmarks": landmarks,
        "validation": validation,
    }
    if not ok:
        payload["reason"] = rejection_reason
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a JPEG/PNG image file")
    parser.add_argument(
        "--artwork",
        action="store_true",
        help="Use relaxed validation thresholds for museum artwork portraits",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(json.dumps({"ok": False, "reason": "missing_image"}))
        return 1
    if not MODEL_PATH.exists():
        print(json.dumps({"ok": False, "reason": "missing_model"}))
        return 1

    payload = detect(image_path, artwork=args.artwork)
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
