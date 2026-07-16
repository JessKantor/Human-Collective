"""
Face-shape geometry and scoring classifier.

Uses the same four landmark metrics as build_database.py but scores all five
wikiHow-style shapes (oval, round, square, heart, oblong) instead of a strict
if/else chain that collapses most portraits into oval.
"""

from __future__ import annotations

import math
from typing import Any

# Landmark indices — MediaPipe Face Mesh
LANDMARK_FOREHEAD_TOP = 10
LANDMARK_CHIN = 152
LANDMARK_LEFT_CHEEK = 234
LANDMARK_RIGHT_CHEEK = 454
LANDMARK_LEFT_TEMPLE = 67
LANDMARK_RIGHT_TEMPLE = 297
LANDMARK_LEFT_JAW = 172
LANDMARK_RIGHT_JAW = 397

FACE_SHAPES = ("oval", "round", "square", "heart", "oblong")

# Ideal metric centroids tuned to cosmetology / wikiHow face-shape diagrams.
# Metrics: length_to_width, jaw_to_forehead, cheek_to_jaw, jaw_angle_deg
_SHAPE_IDEALS: dict[str, dict[str, float]] = {
    "round": {"length_to_width": 1.02, "jaw_to_forehead": 0.96, "cheek_to_jaw": 1.00, "jaw_angle_deg": 135.0},
    "oval": {"length_to_width": 1.28, "jaw_to_forehead": 0.88, "cheek_to_jaw": 1.06, "jaw_angle_deg": 128.0},
    "square": {"length_to_width": 1.12, "jaw_to_forehead": 0.97, "cheek_to_jaw": 1.00, "jaw_angle_deg": 112.0},
    "heart": {"length_to_width": 1.22, "jaw_to_forehead": 0.74, "cheek_to_jaw": 1.14, "jaw_angle_deg": 124.0},
    "oblong": {"length_to_width": 1.58, "jaw_to_forehead": 0.94, "cheek_to_jaw": 1.02, "jaw_angle_deg": 130.0},
}

# How far each metric may deviate before its partial score hits zero.
_SHAPE_TOLERANCES: dict[str, float] = {
    "length_to_width": 0.22,
    "jaw_to_forehead": 0.14,
    "cheek_to_jaw": 0.12,
    "jaw_angle_deg": 18.0,
}


def _landmark_distance(
    landmarks: list[Any], idx_a: int, idx_b: int, width: int, height: int
) -> float:
    a = landmarks[idx_a]
    b = landmarks[idx_b]
    ax, ay = a.x * width, a.y * height
    bx, by = b.x * width, b.y * height
    return math.hypot(ax - bx, ay - by)


def _angle_at_vertex(
    landmarks: list[Any],
    vertex_idx: int,
    arm_a_idx: int,
    arm_b_idx: int,
    width: int,
    height: int,
) -> float:
    vertex = landmarks[vertex_idx]
    arm_a = landmarks[arm_a_idx]
    arm_b = landmarks[arm_b_idx]

    vx, vy = vertex.x * width, vertex.y * height
    ax, ay = arm_a.x * width, arm_a.y * height
    bx, by = arm_b.x * width, arm_b.y * height

    vec_a = (ax - vx, ay - vy)
    vec_b = (bx - vx, by - vy)

    mag_a = math.hypot(*vec_a)
    mag_b = math.hypot(*vec_b)
    if mag_a == 0 or mag_b == 0:
        return 0.0

    cos_angle = (vec_a[0] * vec_b[0] + vec_a[1] * vec_b[1]) / (mag_a * mag_b)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def compute_face_shape_metrics(
    landmarks: list[Any], image_width: int, image_height: int
) -> dict[str, float]:
    """Extract the four geometric ratios used for shape classification."""
    face_height = _landmark_distance(
        landmarks, LANDMARK_FOREHEAD_TOP, LANDMARK_CHIN, image_width, image_height
    )
    cheek_width = _landmark_distance(
        landmarks, LANDMARK_LEFT_CHEEK, LANDMARK_RIGHT_CHEEK, image_width, image_height
    )
    forehead_width = _landmark_distance(
        landmarks, LANDMARK_LEFT_TEMPLE, LANDMARK_RIGHT_TEMPLE, image_width, image_height
    )
    jaw_width = _landmark_distance(
        landmarks, LANDMARK_LEFT_JAW, LANDMARK_RIGHT_JAW, image_width, image_height
    )

    if cheek_width == 0 or forehead_width == 0 or jaw_width == 0:
        return {
            "length_to_width": 1.28,
            "jaw_to_forehead": 0.88,
            "cheek_to_jaw": 1.06,
            "jaw_angle_deg": 128.0,
        }

    left_jaw_angle = _angle_at_vertex(
        landmarks,
        LANDMARK_LEFT_JAW,
        LANDMARK_CHIN,
        LANDMARK_LEFT_CHEEK,
        image_width,
        image_height,
    )
    right_jaw_angle = _angle_at_vertex(
        landmarks,
        LANDMARK_RIGHT_JAW,
        LANDMARK_CHIN,
        LANDMARK_RIGHT_CHEEK,
        image_width,
        image_height,
    )

    return {
        "length_to_width": face_height / cheek_width,
        "jaw_to_forehead": jaw_width / forehead_width,
        "cheek_to_jaw": cheek_width / jaw_width,
        "jaw_angle_deg": (left_jaw_angle + right_jaw_angle) / 2.0,
    }


def _metric_closeness(actual: float, ideal: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(actual - ideal) / tolerance)


def score_face_shapes(metrics: dict[str, float]) -> dict[str, float]:
    """Score each face shape 0–1 based on proximity to ideal metric profiles."""
    raw_scores: dict[str, float] = {}
    for shape in FACE_SHAPES:
        ideals = _SHAPE_IDEALS[shape]
        parts = [
            _metric_closeness(metrics[key], ideals[key], _SHAPE_TOLERANCES[key])
            for key in _SHAPE_TOLERANCES
        ]
        raw_scores[shape] = sum(parts) / len(parts)

    total = sum(raw_scores.values())
    if total <= 0:
        return {shape: 1.0 / len(FACE_SHAPES) for shape in FACE_SHAPES}
    return {shape: score / total for shape, score in raw_scores.items()}


def classify_face_shape_detailed(
    landmarks: list[Any], image_width: int, image_height: int
) -> tuple[str, float, dict[str, float], dict[str, float]]:
    """
    Return (best_shape, confidence, shape_scores, face_metrics).

    confidence is the winning shape's normalized score (0–1).
    """
    metrics = compute_face_shape_metrics(landmarks, image_width, image_height)
    scores = score_face_shapes(metrics)
    best_shape = max(scores, key=scores.get)
    return best_shape, scores[best_shape], scores, metrics


def classify_face_shape(
    landmarks: list[Any], image_width: int, image_height: int
) -> str:
    """Backward-compatible wrapper returning only the best shape label."""
    shape, _, _, _ = classify_face_shape_detailed(landmarks, image_width, image_height)
    return shape
