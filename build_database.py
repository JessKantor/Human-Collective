#!/usr/bin/env python3
"""
Build art_database.json from Metropolitan Museum of Art portrait artworks.

Queries the Met's public API, downloads images, detects faces with MediaPipe
Face Landmarker (Tasks API), classifies face shape from landmark geometry, and
saves matching works.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
import subprocess
import tempfile

from face_validation import artwork_exclusion_reason
from face_shape import classify_face_shape, classify_face_shape_detailed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MET_API_BASE = "https://collectionapi.metmuseum.org/public/collection/v1"
OUTPUT_FILE = Path("art_database.json")
CANDIDATES_FILE = Path("build_candidates.json")
PROGRESS_FILE = Path("build_progress.json")
COMPLETE_MARKER = Path("build_complete.json")

# departmentId -> display name (for logging and saved records)
DEPARTMENTS: dict[int, str] = {
    11: "European Paintings",
    6: "Asian Art",
    14: "Islamic Art",
    10: "Egyptian Art",
    13: "Greek and Roman Art",
    5: "Arts of Africa, Oceania, and the Americas",
}

# Broad portrait-related terms — many works are not tagged "portrait" in metadata.
SEARCH_QUERIES = ("portrait", "face", "figure", "woman", "man", "bust", "head")

# The Met search endpoint has no offset/page params; each call returns one ID list.
# We run multiple strategies per term to maximize coverage.
SEARCH_STRATEGIES: tuple[dict[str, str], ...] = (
    {},  # full-text search across all object fields
    {"title": "true"},  # title field only
    {"tags": "true"},  # subject keyword tags only
)

# Medium-filtered supplemental searches surface objects missed by general queries.
SUPPLEMENTAL_MEDIUM_SEARCHES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Paintings", ("portrait", "figure", "woman", "man", "head")),
    ("Sculpture", ("portrait", "bust", "head", "figure")),
)

# If a search reports more total hits than returned IDs, re-query in date windows.
DATE_RANGE_WINDOWS: tuple[tuple[int, int], ...] = (
    (-3000, -1000),
    (-1000, -1),
    (0, 999),
    (1000, 1399),
    (1400, 1599),
    (1600, 1799),
    (1800, 1899),
    (1900, 2025),
)

# Polite delay between Met API calls (seconds)
API_DELAY = 0.25
REQUEST_TIMEOUT = 30

# MediaPipe Face Landmarker model (Tasks API requires a bundled .task file)
FACE_LANDMARKER_MODEL_PATH = Path("face_landmarker.task")
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_DETECT_WORKER = Path(__file__).resolve().parent / "face_detect_worker.py"
FACE_DETECT_TIMEOUT = 120

# Re-export landmark constants for any external tooling that imported them here.
from face_shape import (  # noqa: E402
    LANDMARK_CHIN,
    LANDMARK_FOREHEAD_TOP,
    LANDMARK_LEFT_CHEEK,
    LANDMARK_LEFT_JAW,
    LANDMARK_LEFT_TEMPLE,
    LANDMARK_RIGHT_CHEEK,
    LANDMARK_RIGHT_JAW,
    LANDMARK_RIGHT_TEMPLE,
)


# ---------------------------------------------------------------------------
# Met API helpers
# ---------------------------------------------------------------------------


def met_get(
    url: str, session: requests.Session, params: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """GET a Met API endpoint; return parsed JSON or None on failure."""
    for attempt in range(3):
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code in {403, 429, 500, 502, 503, 504}:
                wait = API_DELAY * (attempt + 2)
                print(f"  [warn] HTTP {response.status_code}, retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            time.sleep(API_DELAY)
            return response.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            if attempt < 2:
                time.sleep(API_DELAY * (attempt + 2))
                continue
            print(f"  [warn] API request failed: {url} ({exc})")
            return None
    return None


def _search_params(
    dept_id: int,
    query: str,
    extra: dict[str, str] | None = None,
    medium: str | None = None,
    date_begin: int | None = None,
    date_end: int | None = None,
) -> dict[str, str]:
    params: dict[str, str] = {
        "departmentId": str(dept_id),
        "q": query,
        "hasImages": "true",
    }
    if extra:
        params.update(extra)
    if medium:
        params["medium"] = medium
    if date_begin is not None and date_end is not None:
        params["dateBegin"] = str(date_begin)
        params["dateEnd"] = str(date_end)
    return params


def _ingest_search_results(
    data: dict[str, Any],
    seen: set[int],
    ordered_ids: list[int],
) -> tuple[int, int, int]:
    """Add new object IDs from a search response.

    Returns (total_reported, returned_count, newly_added_count).
    """
    object_ids = data.get("objectIDs") or []
    total = data.get("total", 0)
    new_count = 0
    for obj_id in object_ids:
        if obj_id not in seen:
            seen.add(obj_id)
            ordered_ids.append(obj_id)
            new_count += 1
    return total, len(object_ids), new_count


def _run_search(
    session: requests.Session,
    dept_id: int,
    query: str,
    extra: dict[str, str] | None = None,
    medium: str | None = None,
    date_begin: int | None = None,
    date_end: int | None = None,
) -> dict[str, Any] | None:
    return met_get(
        f"{MET_API_BASE}/search",
        session,
        _search_params(dept_id, query, extra, medium, date_begin, date_end),
    )


def _describe_search(
    query: str,
    extra: dict[str, str] | None = None,
    medium: str | None = None,
    date_begin: int | None = None,
    date_end: int | None = None,
) -> str:
    parts = [f"q={query!r}"]
    if medium:
        parts.append(f"medium={medium!r}")
    if extra:
        for key, value in extra.items():
            parts.append(f"{key}={value!r}")
    if date_begin is not None and date_end is not None:
        parts.append(f"dates={date_begin}..{date_end}")
    return ", ".join(parts)


def collect_object_ids(
    session: requests.Session,
) -> tuple[list[int], dict[str, int], list[str]]:
    """
    Search all departments with many query variants and return deduplicated IDs.

    The Met API does not support pagination (no offset/limit). Each search returns
    a single objectIDs array. When ``total`` exceeds ``len(objectIDs)``, the
    response is truncated; we detect that and run supplemental date-window searches.
    """
    seen: set[int] = set()
    ordered_ids: list[int] = []
    dept_candidate_counts: Counter[str] = Counter()
    audit_notes: list[str] = []

    print(f"Searching {len(DEPARTMENTS)} departments: {', '.join(DEPARTMENTS.values())}\n")

    for dept_id, dept_name in DEPARTMENTS.items():
        dept_before = len(ordered_ids)
        truncated_queries: set[str] = set()
        print(f"=== {dept_name} (departmentId={dept_id}) ===")

        def ingest(
            data: dict[str, Any] | None,
            label: str,
            query: str | None = None,
            track_truncation: bool = False,
        ) -> None:
            if not data:
                return
            total, returned, new_count = _ingest_search_results(data, seen, ordered_ids)
            print(
                f"  {label}: total={total}, returned={returned}, "
                f"new={new_count}, pool={len(ordered_ids)}"
            )
            if total > returned:
                note = (
                    f"TRUNCATED: {dept_name} [{label}] reported {total} hits "
                    f"but only returned {returned} IDs"
                )
                audit_notes.append(note)
                print(f"  [warn] {note}")
                if track_truncation and query:
                    truncated_queries.add(query)

        for query in SEARCH_QUERIES:
            for strategy in SEARCH_STRATEGIES:
                label = _describe_search(query, strategy)
                ingest(
                    _run_search(session, dept_id, query, strategy),
                    label,
                    query=query,
                    track_truncation=not strategy,
                )

        for medium, medium_queries in SUPPLEMENTAL_MEDIUM_SEARCHES:
            for query in medium_queries:
                label = _describe_search(query, medium=medium)
                ingest(_run_search(session, dept_id, query, medium=medium), label)

        for query in sorted(truncated_queries):
            print(f"  Recovering truncated q={query!r} via date windows...")
            for date_begin, date_end in DATE_RANGE_WINDOWS:
                label = _describe_search(query, date_begin=date_begin, date_end=date_end)
                ingest(
                    _run_search(
                        session, dept_id, query, date_begin=date_begin, date_end=date_end
                    ),
                    label,
                )

        dept_added = len(ordered_ids) - dept_before
        dept_candidate_counts[dept_name] = dept_added
        print(f"  Department subtotal: {dept_added} new candidates\n")

    # Map each object ID back to its department via a lightweight ownership hint.
    # (Exact per-ID department is resolved later when object metadata is fetched.)
    print(f"Collected {len(ordered_ids)} unique candidate object IDs across all searches.\n")
    if audit_notes:
        print("Search audit notes:")
        for note in audit_notes:
            print(f"  - {note}")
        print()

    return ordered_ids, dict(dept_candidate_counts), audit_notes


def fetch_object_details(
    object_id: int, session: requests.Session
) -> dict[str, Any] | None:
    """Fetch full object metadata from the Met API."""
    url = f"{MET_API_BASE}/objects/{object_id}"
    return met_get(url, session)


def download_image(image_url: str, session: requests.Session) -> np.ndarray | None:
    """Download an image and return it as a BGR numpy array for OpenCV/MediaPipe."""
    try:
        response = session.get(image_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        image_array = np.frombuffer(response.content, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image
    except (requests.RequestException, cv2.error) as exc:
        print(f"  [warn] Image download failed: {image_url} ({exc})")
        return None


# ---------------------------------------------------------------------------
# Face detection (isolated subprocess — survives MediaPipe segfaults)
# ---------------------------------------------------------------------------


class _LandmarkPoint:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


def landmarks_from_worker_payload(payload: dict[str, Any]) -> list[_LandmarkPoint]:
    """Rehydrate landmark points returned by face_detect_worker.py."""
    landmarks: list[_LandmarkPoint] = [_LandmarkPoint(0.0, 0.0, 0.0)] * 500
    for index_str, coords in payload["landmarks"].items():
        landmarks[int(index_str)] = _LandmarkPoint(
            coords["x"], coords["y"], coords["z"]
        )
    return landmarks


def ensure_face_landmarker_model() -> Path:
    """Download the Face Landmarker model if it is not already present locally."""
    if FACE_LANDMARKER_MODEL_PATH.exists():
        return FACE_LANDMARKER_MODEL_PATH

    print(f"Downloading Face Landmarker model to {FACE_LANDMARKER_MODEL_PATH}...")
    response = requests.get(FACE_LANDMARKER_MODEL_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    FACE_LANDMARKER_MODEL_PATH.write_bytes(response.content)
    print("Model download complete.\n")
    return FACE_LANDMARKER_MODEL_PATH


def image_urls_to_try(obj: dict[str, Any]) -> list[str]:
    """
    Build an ordered list of image URLs to attempt for face detection.

    We try the full-resolution primary image first, then the web thumbnail, then
    any additional views — many portraits are only detectable in a larger crop.
    """
    urls: list[str] = []
    for key in ("primaryImage", "primaryImageSmall"):
        url = (obj.get(key) or "").strip()
        if url and url not in urls:
            urls.append(url)
    for url in obj.get("additionalImages") or []:
        cleaned = (url or "").strip()
        if cleaned and cleaned not in urls:
            urls.append(cleaned)
    return urls


def detect_face_in_image_file(image_path: Path, *, artwork: bool = False) -> dict[str, Any] | None:
    """
    Run face detection in a child process.

    If MediaPipe crashes with a segfault, only the worker dies — the builder
    keeps running and simply treats the image as "no face detected".
    """
    ensure_face_landmarker_model()
    try:
        cmd = [sys.executable, str(FACE_DETECT_WORKER), "--image", str(image_path)]
        if artwork:
            cmd.append("--artwork")
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FACE_DETECT_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print("  [warn] Face detection timed out")
        return None

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        if not stdout:
            if stderr:
                print(f"  [warn] Face worker exited {completed.returncode}: {stderr[:200]}")
            return None

    stdout = (completed.stdout or "").strip()
    if not stdout:
        return None

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        print("  [warn] Face worker returned invalid JSON")
        return None

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        if stderr and payload.get("reason") not in {
            "no_face",
            "unreadable_image",
            "profile_pose",
            "face_too_small",
            "multiple_faces",
            "bad_face_aspect",
        }:
            print(f"  [warn] Face worker exited {completed.returncode}: {stderr[:200]}")

    return payload


def detect_face_in_object(
    obj: dict[str, Any],
    session: requests.Session,
) -> tuple[list[Any] | None, int, int, str, str | None]:
    """
    Try each available image URL until a valid face is detected.

    Returns (landmarks, width, height, image_url_used, rejection_reason).
    rejection_reason is None when a valid face is found.
    """
    last_reason = "no_face"
    for image_url in image_urls_to_try(obj):
        image = download_image(image_url, session)
        if image is None:
            continue

        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
                temp_path = handle.name
                cv2.imwrite(temp_path, image)

            payload = detect_face_in_image_file(Path(temp_path), artwork=True)
            if payload is None:
                continue
            if payload.get("ok"):
                return (
                    landmarks_from_worker_payload(payload),
                    int(payload["width"]),
                    int(payload["height"]),
                    image_url,
                    None,
                )
            last_reason = payload.get("reason") or last_reason
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    return None, 0, 0, "", last_reason


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------


def build_record(
    obj: dict[str, Any],
    face_shape: str,
    *,
    shape_confidence: float | None = None,
    face_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Map Met API object fields to our database schema."""
    record: dict[str, Any] = {
        "id": obj["objectID"],
        "title": obj.get("title") or "",
        "artist": obj.get("artistDisplayName") or "",
        "date": obj.get("objectDate") or "",
        "culture": obj.get("culture") or "",
        "department": obj.get("department") or "",
        "imageUrl": obj.get("primaryImageSmall") or "",
        "metUrl": obj.get("objectURL") or "",
        "faceShape": face_shape,
    }
    if shape_confidence is not None:
        record["shapeConfidence"] = round(shape_confidence, 3)
    if face_metrics is not None:
        record["faceMetrics"] = {
            key: round(value, 3) for key, value in face_metrics.items()
        }
    return record


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------


def load_existing_results() -> list[dict[str, Any]]:
    """Load previously saved artworks so a restarted run can continue."""
    if not OUTPUT_FILE.exists():
        return []
    try:
        data = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[warn] Could not read {OUTPUT_FILE}: {exc}")
        return []


def load_saved_ids(results: list[dict[str, Any]]) -> set[int]:
    return {int(record["id"]) for record in results if record.get("id") is not None}


def load_processed_ids() -> set[int]:
    if not PROGRESS_FILE.exists():
        return set()
    try:
        data = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        return {int(obj_id) for obj_id in data.get("processed_ids", [])}
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        print(f"[warn] Could not read {PROGRESS_FILE}: {exc}")
        return set()


def save_progress(processed_ids: set[int]) -> None:
    PROGRESS_FILE.write_text(
        json.dumps(
            {
                "processed_ids": sorted(processed_ids),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def save_results(results: list[dict[str, Any]]) -> None:
    OUTPUT_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_cached_candidates() -> tuple[list[int], dict[str, int], list[str]] | None:
    if not CANDIDATES_FILE.exists():
        return None
    try:
        data = json.loads(CANDIDATES_FILE.read_text(encoding="utf-8"))
        object_ids = [int(obj_id) for obj_id in data.get("object_ids", [])]
        if not object_ids:
            return None
        return (
            object_ids,
            data.get("dept_candidate_counts", {}),
            data.get("audit_notes", []),
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        print(f"[warn] Could not read {CANDIDATES_FILE}: {exc}")
        return None


def save_cached_candidates(
    object_ids: list[int],
    dept_candidate_counts: dict[str, int],
    audit_notes: list[str],
) -> None:
    CANDIDATES_FILE.write_text(
        json.dumps(
            {
                "object_ids": object_ids,
                "dept_candidate_counts": dept_candidate_counts,
                "audit_notes": audit_notes,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def mark_build_complete(
    results: list[dict[str, Any]],
    processed: int,
    saved_by_department: Counter[str],
) -> None:
    COMPLETE_MARKER.write_text(
        json.dumps(
            {
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "saved_artworks": len(results),
                "objects_examined": processed,
                "saved_by_department": dict(saved_by_department),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def clear_checkpoints() -> None:
    for path in (CANDIDATES_FILE, PROGRESS_FILE, COMPLETE_MARKER):
        if path.exists():
            path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build art_database.json from Met portraits.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint files (default when checkpoints exist).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete checkpoints and start a brand-new build.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    has_checkpoints = any(
        path.exists() for path in (OUTPUT_FILE, CANDIDATES_FILE, PROGRESS_FILE)
    )
    resume = args.resume or (has_checkpoints and not args.fresh)

    print("=" * 60)
    print("Human Collective — Met Museum art database builder")
    print("=" * 60)

    if args.fresh:
        clear_checkpoints()
        if OUTPUT_FILE.exists():
            OUTPUT_FILE.unlink()
        print("Starting fresh — removed prior checkpoints.\n")
    elif resume:
        print("Resume mode — will skip finished candidates and keep existing saves.\n")

    session = requests.Session()
    session.headers.update({"User-Agent": "HumanCollectiveMVP/1.0 (educational project)"})

    cached = load_cached_candidates() if resume else None
    if cached:
        object_ids, dept_candidate_counts, audit_notes = cached
        print(
            f"Loaded {len(object_ids)} cached candidates from {CANDIDATES_FILE.name} "
            "(skipping Met search).\n"
        )
    else:
        object_ids, dept_candidate_counts, audit_notes = collect_object_ids(session)
        if not object_ids:
            print("No object IDs found. Exiting.")
            return 1
        save_cached_candidates(object_ids, dept_candidate_counts, audit_notes)

    results = load_existing_results() if resume else []
    saved_ids = load_saved_ids(results)
    processed_ids = load_processed_ids() if resume else set()
    processed_ids.update(saved_ids)

    remaining_ids = [obj_id for obj_id in object_ids if obj_id not in processed_ids]
    saved_by_department = Counter(record.get("department") or "Unknown" for record in results)
    skipped_by_department: Counter[str] = Counter()
    skipped_no_domain = 0
    skipped_no_image = 0
    skipped_no_face = 0
    skipped_metadata = 0
    skipped_face_quality: Counter[str] = Counter()
    api_errors = 0
    processed_this_run = 0

    print(
        f"Processing {len(remaining_ids)} remaining candidates "
        f"({len(processed_ids)} already done, {len(results)} saved so far)...\n"
    )

    if not remaining_ids:
        print("Nothing left to process.")
    else:
        for index, object_id in enumerate(remaining_ids, start=1):
            processed_this_run += 1
            print(
                f"[{index}/{len(remaining_ids)}] Object {object_id}...",
                end=" ",
                flush=True,
            )

            try:
                obj = fetch_object_details(object_id, session)
                if not obj:
                    api_errors += 1
                    print("skipped (API error)")
                    continue

                department = obj.get("department") or "Unknown"

                if not obj.get("isPublicDomain"):
                    skipped_no_domain += 1
                    skipped_by_department[department] += 1
                    print("skipped (not public domain)")
                    continue

                if not image_urls_to_try(obj):
                    skipped_no_image += 1
                    skipped_by_department[department] += 1
                    print("skipped (no image)")
                    continue

                metadata_reason = artwork_exclusion_reason(obj)
                if metadata_reason:
                    skipped_metadata += 1
                    skipped_by_department[department] += 1
                    print(f"skipped ({metadata_reason})")
                    continue

                landmarks, width, height, image_url_used, face_reason = detect_face_in_object(
                    obj, session
                )
                if landmarks is None:
                    if face_reason == "no_face":
                        skipped_no_face += 1
                    else:
                        skipped_face_quality[face_reason or "face_quality"] += 1
                    skipped_by_department[department] += 1
                    print(f"skipped ({face_reason or 'no face detected'})")
                    continue

                url_reason = artwork_exclusion_reason(obj, image_url_used)
                if url_reason:
                    skipped_metadata += 1
                    skipped_by_department[department] += 1
                    print(f"skipped ({url_reason})")
                    continue

                face_shape, shape_confidence, _, face_metrics = classify_face_shape_detailed(
                    landmarks, width, height
                )
                record = build_record(
                    obj,
                    face_shape,
                    shape_confidence=shape_confidence,
                    face_metrics=face_metrics,
                )
                if image_url_used and image_url_used != record["imageUrl"]:
                    record["imageUrl"] = image_url_used
                results.append(record)
                saved_ids.add(object_id)
                saved_by_department[department] += 1
                save_results(results)

                title_preview = (
                    (record["title"][:50] + "…")
                    if len(record["title"]) > 50
                    else record["title"]
                )
                print(f"saved as {face_shape!r} — {title_preview!r} ({len(results)} saved)")
            finally:
                processed_ids.add(object_id)
                save_progress(processed_ids)

    mark_build_complete(results, len(processed_ids), saved_by_department)

    print("\n" + "=" * 60)
    print("Done.")
    print(f"  Saved artworks:        {len(results)}")
    print(f"  Objects examined:      {len(processed_ids)}")
    print(f"  Examined this run:     {processed_this_run}")
    print(f"  Skipped (not domain):  {skipped_no_domain}")
    print(f"  Skipped (no image):    {skipped_no_image}")
    print(f"  Skipped (no face):     {skipped_no_face}")
    print(f"  Skipped (metadata):    {skipped_metadata}")
    if skipped_face_quality:
        print("  Skipped (face quality):")
        for reason, count in skipped_face_quality.most_common():
            print(f"    {reason:<22} {count:>5}")
    print(f"  API errors:            {api_errors}")
    print(f"  Output file:           {OUTPUT_FILE.resolve()}")

    print("\nSearch candidates added per department:")
    for dept_name in DEPARTMENTS.values():
        print(f"  {dept_name:<45} {dept_candidate_counts.get(dept_name, 0):>5}")

    print("\nSaved artworks per department:")
    for dept_name in DEPARTMENTS.values():
        print(f"  {dept_name:<45} {saved_by_department.get(dept_name, 0):>5}")
    for dept_name, count in sorted(saved_by_department.items()):
        if dept_name not in DEPARTMENTS.values():
            print(f"  {dept_name:<45} {count:>5}")

    if skipped_by_department:
        print("\nSkipped (all reasons) per department:")
        for dept_name in DEPARTMENTS.values():
            print(f"  {dept_name:<45} {skipped_by_department.get(dept_name, 0):>5}")

    if audit_notes:
        print("\nTruncated searches detected during collection:")
        for note in audit_notes:
            print(f"  - {note}")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
