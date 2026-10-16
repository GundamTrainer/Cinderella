from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from ai.vision import BoundingBox

DATASET_SCHEMA_VERSION = "1.0"
SPLITS = {"train", "validation", "test"}
VISIBILITY_VALUES = {"visible", "partially_occluded", "mostly_occluded"}
ANNOTATION_SOURCES = {"human", "synthetic"}
VERIFICATION_STATES = {"verified", "unverified"}


class DatasetValidationError(ValueError):
    """Raised when an object-detection dataset manifest is invalid."""


def validate_dataset_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise DatasetValidationError(
            f"schema_version must be {DATASET_SCHEMA_VERSION!r}."
        )
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise DatasetValidationError("samples must be a list.")
    sample_ids: set[str] = set()
    session_splits: dict[str, str] = {}
    for index, sample in enumerate(samples):
        _validate_sample(sample, index, sample_ids, session_splits)


def _validate_sample(
    sample: Any,
    index: int,
    sample_ids: set[str],
    session_splits: dict[str, str],
) -> None:
    prefix = f"samples[{index}]"
    if not isinstance(sample, dict):
        raise DatasetValidationError(f"{prefix} must be an object.")
    sample_id = _nonempty_string(sample.get("sample_id"), f"{prefix}.sample_id")
    if sample_id in sample_ids:
        raise DatasetValidationError(f"Duplicate sample_id: {sample_id}.")
    sample_ids.add(sample_id)
    image_ref = _nonempty_string(sample.get("image_ref"), f"{prefix}.image_ref")
    path = PurePosixPath(image_ref)
    if path.is_absolute() or ".." in path.parts or "\\" in image_ref:
        raise DatasetValidationError(f"{prefix}.image_ref must be a safe relative POSIX path.")
    width = _positive_int(sample.get("image_width"), f"{prefix}.image_width")
    height = _positive_int(sample.get("image_height"), f"{prefix}.image_height")
    session_id = _nonempty_string(sample.get("session_id"), f"{prefix}.session_id")
    split = sample.get("split")
    if split not in SPLITS:
        raise DatasetValidationError(f"{prefix}.split must be one of {sorted(SPLITS)}.")
    previous_split = session_splits.setdefault(session_id, split)
    if previous_split != split:
        raise DatasetValidationError(
            f"Session {session_id!r} appears in multiple dataset splits."
        )
    ownership = sample.get("consent_or_ownership")
    if not isinstance(ownership, dict) or not ownership.get("statement"):
        raise DatasetValidationError(f"{prefix}.consent_or_ownership.statement is required.")
    annotations = sample.get("annotations")
    if not isinstance(annotations, list):
        raise DatasetValidationError(f"{prefix}.annotations must be a list.")
    for annotation_index, annotation in enumerate(annotations):
        _validate_annotation(annotation, prefix, annotation_index, width, height)


def _validate_annotation(
    annotation: Any, prefix: str, index: int, width: int, height: int
) -> None:
    annotation_prefix = f"{prefix}.annotations[{index}]"
    if not isinstance(annotation, dict):
        raise DatasetValidationError(f"{annotation_prefix} must be an object.")
    _nonempty_string(annotation.get("object_class"), f"{annotation_prefix}.object_class")
    box = annotation.get("bbox")
    if not isinstance(box, dict):
        raise DatasetValidationError(f"{annotation_prefix}.bbox must be an object.")
    try:
        BoundingBox(
            float(box["x_min"]) / width,
            float(box["y_min"]) / height,
            float(box["x_max"]) / width,
            float(box["y_max"]) / height,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetValidationError(f"{annotation_prefix}.bbox is invalid: {exc}") from exc
    if annotation.get("visibility") not in VISIBILITY_VALUES:
        raise DatasetValidationError(f"{annotation_prefix}.visibility is invalid.")
    occlusion = annotation.get("occlusion_level")
    if isinstance(occlusion, bool) or not isinstance(occlusion, (int, float)) or not 0 <= occlusion <= 1:
        raise DatasetValidationError(
            f"{annotation_prefix}.occlusion_level must be in [0, 1]."
        )
    if annotation.get("annotation_source") not in ANNOTATION_SOURCES:
        raise DatasetValidationError(f"{annotation_prefix}.annotation_source is invalid.")
    if annotation.get("annotator_verification_state") not in VERIFICATION_STATES:
        raise DatasetValidationError(
            f"{annotation_prefix}.annotator_verification_state is invalid."
        )


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{field} must be a non-empty string.")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DatasetValidationError(f"{field} must be a positive integer.")
    return value


def example_synthetic_metadata() -> dict[str, Any]:
    """Return non-media sample metadata suitable for docs and schema tests."""
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "samples": [
            {
                "sample_id": "synthetic-session-001-frame-0001",
                "session_id": "synthetic-session-001",
                "image_ref": "synthetic/session-001/frame-0001.png",
                "image_width": 640,
                "image_height": 480,
                "split": "train",
                "consent_or_ownership": {
                    "statement": "synthetic placeholder; no image file is included"
                },
                "annotations": [
                    {
                        "object_class": "synthetic_box",
                        "bbox": {
                            "x_min": 100,
                            "y_min": 80,
                            "x_max": 300,
                            "y_max": 280,
                        },
                        "visibility": "visible",
                        "occlusion_level": 0.0,
                        "annotation_source": "synthetic",
                        "annotator_verification_state": "verified",
                    }
                ],
            }
        ],
    }
