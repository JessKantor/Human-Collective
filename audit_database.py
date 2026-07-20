#!/usr/bin/env python3
"""
Audit and clean art_database.json using Phase 1 + Phase 2 rules.

Re-downloads each artwork image, applies metadata and face-quality gates,
reclassifies face shape with the scoring classifier, and writes:

  - art_database_clean.json  — kept records with updated labels/metrics
  - audit_report.json        — full audit log with rejects and summary stats

Usage:
  python3 audit_database.py
  python3 audit_database.py --limit 10          # smoke test
  python3 audit_database.py --metadata-only     # skip image re-scan
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import requests

from build_database import (
    MET_API_BASE,
    detect_face_in_image_file,
    download_image,
    landmarks_from_worker_payload,
    met_get,
)
from face_shape import classify_face_shape_detailed
from face_validation import artwork_exclusion_reason, metadata_exclusion_reason

INPUT_FILE = Path("art_database.json")
OUTPUT_CLEAN = Path("art_database_clean.json")
OUTPUT_REPORT = Path("audit_report.json")
PROGRESS_FILE = Path("audit_progress.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit art_database.json and produce a cleaned, reclassified database."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FILE,
        help=f"Source database (default: {INPUT_FILE})",
    )
    parser.add_argument(
        "--output-clean",
        type=Path,
        default=OUTPUT_CLEAN,
        help=f"Cleaned database output (default: {OUTPUT_CLEAN})",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=OUTPUT_REPORT,
        help=f"Audit report output (default: {OUTPUT_REPORT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N records (0 = all)",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Apply metadata blocklist only; do not re-download images",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from audit_progress.json if present",
    )
    parser.add_argument(
        "--met-delay",
        type=float,
        default=0.25,
        help="Delay between Met API calls when fetching full image URLs (seconds)",
    )
    return parser.parse_args()


def record_as_metadata_obj(record: dict[str, Any]) -> dict[str, Any]:
    """Map a saved database record to the metadata shape used by Phase 1."""
    return {
        "title": record.get("title") or "",
        "objectName": record.get("objectName") or "",
        "classification": record.get("classification") or "",
        "medium": record.get("medium") or "",
        "tags": record.get("tags") or "",
    }


def updated_record(
    record: dict[str, Any],
    face_shape: str,
    shape_confidence: float,
    face_metrics: dict[str, float],
    validation: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Return a copy of the record with refreshed classification fields."""
    clean = {key: value for key, value in record.items() if key != "faceMetrics"}
    clean["faceShape"] = face_shape
    clean["shapeConfidence"] = round(shape_confidence, 3)
    clean["faceMetrics"] = {key: round(value, 3) for key, value in face_metrics.items()}
    if validation is not None:
        clean["faceValidation"] = {
            key: round(value, 3) if isinstance(value, float) else value
            for key, value in validation.items()
        }
    return clean


def rejection_entry(record: dict[str, Any], reason: str, **extra: Any) -> dict[str, Any]:
    entry = {
        "id": record.get("id"),
        "title": record.get("title") or "",
        "department": record.get("department") or "",
        "imageUrl": record.get("imageUrl") or "",
        "oldFaceShape": record.get("faceShape"),
        "reason": reason,
    }
    entry.update(extra)
    return entry


def load_progress(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[int]]:
    if not path.exists():
        return [], [], set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        kept = data.get("kept") or []
        rejected = data.get("rejected") or []
        done_ids = {int(item["id"]) for item in kept + rejected if item.get("id") is not None}
        return kept, rejected, done_ids
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return [], [], set()


def save_progress(path: Path, kept: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps(
            {
                "kept": kept,
                "rejected": rejected,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def image_urls_for_record(record: dict[str, Any], session: requests.Session) -> list[str]:
    """
    Build an ordered URL list: Met API full images first, then saved imageUrl.
    """
    urls: list[str] = []
    record_id = record.get("id")
    if record_id is not None:
        obj = met_get(f"{MET_API_BASE}/objects/{record_id}", session)
        if obj:
            for key in ("primaryImage", "primaryImageSmall"):
                url = (obj.get(key) or "").strip()
                if url and url not in urls:
                    urls.append(url)
            for url in obj.get("additionalImages") or []:
                cleaned = (url or "").strip()
                if cleaned and cleaned not in urls:
                    urls.append(cleaned)

    saved = (record.get("imageUrl") or "").strip()
    if saved and saved not in urls:
        urls.append(saved)
    return urls


def audit_record_image(
    record: dict[str, Any], session: requests.Session
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
    """
    Re-scan one artwork image, trying full-resolution Met URLs when available.

    Returns (updated_record_or_none, rejection_reason, extra_details).
    """
    urls = image_urls_for_record(record, session)
    if not urls:
        return None, "no_image_url", {}

    last_reason = "no_face"
    last_extra: dict[str, Any] = {}

    for image_url in urls:
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
            if not payload.get("ok"):
                last_reason = payload.get("reason") or last_reason
                last_extra = {"validation": payload.get("validation"), "imageUrl": image_url}
                continue

            landmarks = landmarks_from_worker_payload(payload)
            width = int(payload["width"])
            height = int(payload["height"])
            face_shape, confidence, _, metrics = classify_face_shape_detailed(
                landmarks, width, height
            )
            updated = updated_record(
                record,
                face_shape,
                confidence,
                metrics,
                validation=payload.get("validation"),
            )
            if image_url != record.get("imageUrl"):
                updated["imageUrl"] = image_url
            return updated, None, {}
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    return None, last_reason, last_extra


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        return 1

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        print("Input database must be a JSON array.")
        return 1

    if args.limit > 0:
        records = records[: args.limit]

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    done_ids: set[int] = set()
    reclassified = 0
    shape_changes: list[dict[str, Any]] = []

    if args.resume:
        kept, rejected, done_ids = load_progress(PROGRESS_FILE)
        if done_ids:
            print(f"Resuming — {len(done_ids)} records already audited.\n")
    elif PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

    session = requests.Session()
    session.headers.update({"User-Agent": "HumanCollectiveMVP/1.0 (audit script)"})

    print("=" * 60)
    print("Human Collective — database audit")
    print("=" * 60)
    print(f"Input:          {args.input} ({len(records)} records)")
    print(f"Metadata only:  {args.metadata_only}")
    print()

    for index, record in enumerate(records, start=1):
        record_id = record.get("id")
        if record_id in done_ids:
            continue

        title = (record.get("title") or "")[:55]
        old_shape = record.get("faceShape")
        print(f"[{index}/{len(records)}] {record_id} — {title!r}...", end=" ", flush=True)

        metadata_reason = artwork_exclusion_reason(record_as_metadata_obj(record), record.get("imageUrl") or "")
        if metadata_reason:
            rejected.append(rejection_entry(record, metadata_reason))
            print(f"REJECT ({metadata_reason})")
            save_progress(PROGRESS_FILE, kept, rejected)
            continue

        if args.metadata_only:
            kept.append(dict(record))
            print("KEEP (metadata-only pass)")
            save_progress(PROGRESS_FILE, kept, rejected)
            continue

        updated, reason, extra = audit_record_image(record, session)
        if args.met_delay and record.get("id") is not None:
            time.sleep(args.met_delay)
        if updated is None:
            rejected.append(rejection_entry(record, reason or "unknown", **extra))
            print(f"REJECT ({reason})")
            save_progress(PROGRESS_FILE, kept, rejected)
            continue

        new_shape = updated["faceShape"]
        if old_shape and old_shape != new_shape:
            reclassified += 1
            shape_changes.append(
                {
                    "id": record_id,
                    "title": record.get("title") or "",
                    "oldFaceShape": old_shape,
                    "newFaceShape": new_shape,
                    "shapeConfidence": updated.get("shapeConfidence"),
                }
            )

        kept.append(updated)
        print(f"KEEP → {new_shape} ({updated.get('shapeConfidence', 0):.0%})")
        save_progress(PROGRESS_FILE, kept, rejected)

    processed_ids = {r["id"] for r in kept} | {r["id"] for r in rejected}
    original_shapes = Counter(
        r.get("faceShape") for r in records if r.get("id") in processed_ids
    )
    new_shapes = Counter(r["faceShape"] for r in kept)
    rejection_reasons = Counter(r["reason"] for r in rejected)

    summary = {
        "total_input": len(records),
        "kept": len(kept),
        "rejected": len(rejected),
        "metadata_only": args.metadata_only,
        "reclassified": reclassified,
        "original_shape_counts": dict(original_shapes.most_common()),
        "new_shape_counts": dict(new_shapes.most_common()),
        "rejection_reasons": dict(rejection_reasons.most_common()),
    }

    args.output_clean.write_text(
        json.dumps(kept, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = {
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_file": str(args.input),
        "output_clean": str(args.output_clean),
        "summary": summary,
        "shape_changes": shape_changes,
        "rejected": rejected,
    }
    args.output_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

    print("\n" + "=" * 60)
    print("Audit complete.")
    print(f"  Kept:         {summary['kept']}")
    print(f"  Rejected:     {summary['rejected']}")
    print(f"  Reclassified: {summary['reclassified']}")
    print(f"  Clean DB:     {args.output_clean.resolve()}")
    print(f"  Report:       {args.output_report.resolve()}")
    print("\nOriginal shapes:")
    for shape, count in original_shapes.most_common():
        print(f"  {shape:<10} {count:>5}")
    print("\nNew shapes (kept records):")
    for shape, count in new_shapes.most_common():
        print(f"  {shape:<10} {count:>5}")
    if rejection_reasons:
        print("\nRejection reasons:")
        for reason, count in rejection_reasons.most_common():
            print(f"  {reason:<24} {count:>5}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
