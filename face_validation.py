"""
Shared face-quality validation for artwork scanning and live visitor photos.

Phase 1 gates: metadata blocklist, face count/dominance, size, frontality, aspect.
"""

from __future__ import annotations

import math
from typing import Any

# MediaPipe Face Mesh landmark indices (same as build_database.py)
LANDMARK_NOSE_TIP = 1
LANDMARK_FOREHEAD_TOP = 10
LANDMARK_CHIN = 152
LANDMARK_LEFT_CHEEK = 234
LANDMARK_RIGHT_CHEEK = 454

# Detection options (used by face_detect_worker.py)
MIN_FACE_DETECTION_CONFIDENCE = 0.6
MIN_FACE_PRESENCE_CONFIDENCE = 0.6
ARTWORK_MIN_FACE_DETECTION_CONFIDENCE = 0.35
ARTWORK_MIN_FACE_PRESENCE_CONFIDENCE = 0.35

# Quality gates applied after a face is detected
MIN_FACE_AREA_RATIO = 0.10       # live visitor photos
ARTWORK_MIN_FACE_AREA_RATIO = 0.05  # museum portraits (figure + background)
MIN_DOMINANT_FACE_RATIO = 0.60   # if multiple faces, largest must be >= 60% of total face area
MAX_ABS_YAW_DEG = 25.0           # reject strong profiles
ARTWORK_MAX_ABS_YAW_DEG = 35.0   # painted portraits are often slightly turned
MIN_FACE_ASPECT = 0.75           # bbox height / width
MAX_FACE_ASPECT = 2.0

# Met object text fields scanned for obvious non-portrait subjects
METADATA_BLOCKLIST = (
    "animal",
    "animals",
    "bead",
    "bird",
    "birds",
    "book of the dead",
    "bowl",
    "carpet",
    "cattle",
    "cat ",
    " cats",
    "codex",
    "dog",
    "dish ",
    "drum",
    "ewer",
    "fish",
    "fragment",
    "goblet",
    "horse",
    "horses",
    "landscape",
    "lion",
    "lions",
    "manuscript",
    "mask",
    "masks",
    "mirror bearer",
    "mirror-bearer",
    "monkey",
    "papyrus",
    "pendant",
    "plate",
    "prayer book",
    "scroll",
    "still life",
    "textile",
    "tile",
    "tusk",
    "vase",
    "vessel",
)

# Image URLs that are conservation/technical views, not visitor-facing portraits
IMAGE_URL_BLOCKLIST = (
    "irr",
    "irrdetail",
    "xray",
    "x-ray",
    "reflectography",
    "nocap",
    "_strm",
    "technical",
)

METADATA_FIELDS = (
    "title",
    "objectName",
    "classification",
    "medium",
    "tags",
    "objectWikipediaTitles",
)


class _LandmarkPoint:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value).lower()
    return str(value).lower()


def metadata_exclusion_reason(obj: dict[str, Any]) -> str | None:
    """
    Return a short reason if Met object metadata suggests a non-human portrait.
    """
    combined = " ".join(
        _normalize_text(obj.get(field)) for field in METADATA_FIELDS
    )
    if not combined.strip():
        return None

    for term in METADATA_BLOCKLIST:
        if term in combined:
            return f"metadata:{term.strip()}"
    return None


def image_url_exclusion_reason(image_url: str) -> str | None:
    """Reject conservation/X-ray/study images that are not display portraits."""
    lowered = (image_url or "").lower()
    if not lowered:
        return None
    for term in IMAGE_URL_BLOCKLIST:
        if term in lowered:
            return f"image_url:{term}"
    return None


def artwork_exclusion_reason(obj: dict[str, Any], image_url: str = "") -> str | None:
    """Combined metadata + image URL exclusion check for saved artworks."""
    meta_reason = metadata_exclusion_reason(obj)
    if meta_reason:
        return meta_reason
    url = image_url or obj.get("imageUrl") or obj.get("primaryImage") or ""
    return image_url_exclusion_reason(url)


def face_bbox_metrics(
    landmarks: list[Any], image_width: int, image_height: int
) -> dict[str, float]:
    """Bounding-box metrics for a single face mesh."""
    xs = [lm.x * image_width for lm in landmarks]
    ys = [lm.y * image_height for lm in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    area = width * height
    image_area = max(image_width * image_height, 1)
    return {
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": area,
        "face_area_ratio": area / image_area,
        "face_aspect_ratio": height / width,
    }


def estimate_yaw_deg(
    landmarks: list[Any], image_width: int, image_height: int
) -> float:
    """
    Rough head-yaw estimate from nose position relative to cheek midpoint.

    0° = frontal; positive/negative = turned left/right.
    """
    nose = landmarks[LANDMARK_NOSE_TIP]
    left = landmarks[LANDMARK_LEFT_CHEEK]
    right = landmarks[LANDMARK_RIGHT_CHEEK]
    center_x = (left.x + right.x) / 2.0
    half_width = abs(right.x - left.x) / 2.0
    if half_width <= 1e-6:
        return 0.0
    offset = (nose.x - center_x) / half_width
    offset = max(-1.0, min(1.0, offset))
    return offset * 45.0


def validate_face_landmarks(
    landmarks: list[Any],
    image_width: int,
    image_height: int,
    *,
    face_count: int = 1,
    dominant_face_ratio: float = 1.0,
    artwork: bool = False,
) -> tuple[bool, str | None, dict[str, float]]:
    """
    Apply Phase 1 quality gates to one chosen face.

    Returns (ok, rejection_reason, metrics_dict).
    """
    min_area = ARTWORK_MIN_FACE_AREA_RATIO if artwork else MIN_FACE_AREA_RATIO
    max_yaw = ARTWORK_MAX_ABS_YAW_DEG if artwork else MAX_ABS_YAW_DEG

    metrics = face_bbox_metrics(landmarks, image_width, image_height)
    metrics["face_count"] = float(face_count)
    metrics["dominant_face_ratio"] = dominant_face_ratio
    metrics["yaw_deg"] = estimate_yaw_deg(landmarks, image_width, image_height)

    if face_count > 1 and dominant_face_ratio < MIN_DOMINANT_FACE_RATIO:
        return False, "multiple_faces", metrics

    if metrics["face_area_ratio"] < min_area:
        return False, "face_too_small", metrics

    if abs(metrics["yaw_deg"]) > max_yaw:
        return False, "profile_pose", metrics

    aspect = metrics["face_aspect_ratio"]
    if aspect < MIN_FACE_ASPECT or aspect > MAX_FACE_ASPECT:
        return False, "bad_face_aspect", metrics

    return True, None, metrics


def landmarks_from_index_map(
    landmark_map: dict[str, dict[str, float]], indices: tuple[int, ...]
) -> list[_LandmarkPoint]:
    """Build a sparse landmark list large enough for geometry helpers."""
    points: list[_LandmarkPoint] = [_LandmarkPoint(0.0, 0.0, 0.0)] * 500
    for index in indices:
        key = str(index)
        if key in landmark_map:
            coords = landmark_map[key]
            points[index] = _LandmarkPoint(coords["x"], coords["y"], coords.get("z", 0.0))
    return points


def face_area_from_landmarks(
    landmarks: list[Any], image_width: int, image_height: int
) -> float:
    return face_bbox_metrics(landmarks, image_width, image_height)["bbox_area"]
