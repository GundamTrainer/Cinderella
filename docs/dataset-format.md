# Object-detection dataset manifest

`datasets/object_detection/schema.py` validates metadata manifests. Validation does not load image files or create a dataset. The synthetic example is metadata only and includes no media.

## Top-level fields

- `schema_version`: currently `"1.0"`.
- `samples`: list of sample metadata records.

## Sample fields

- `sample_id`: unique, non-empty ID.
- `image_ref`: safe relative POSIX path; absolute paths, parent traversal, and backslashes are rejected.
- `image_width`, `image_height`: positive integers.
- `session_id`: non-empty capture/session identifier.
- `split`: `train`, `validation`, or `test`; samples from one session cannot cross splits.
- `consent_or_ownership.statement`: required non-empty statement.
- `annotations`: list of object annotations.

Each annotation contains a non-empty `object_class`, pixel-coordinate `bbox` with `x_min`, `y_min`, `x_max`, `y_max`, `visibility`, `occlusion_level` in `[0, 1]`, `annotation_source` (`human` or `synthetic`), and `annotator_verification_state` (`verified` or `unverified`). Bounding boxes are validated through the shared normalized-box type.

This schema does not define consent sufficiency, retention, de-identification, or license review. Those controls must be decided before any real camera-derived media is collected or stored. See [privacy.md](privacy.md).
