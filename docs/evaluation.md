# Evaluation

## Current helper

`ai/evaluation.py` provides `evaluate_detections` for one frame. It greedily matches predictions to remaining ground-truth boxes by class and IoU, sorted by prediction confidence. The result includes true positives, false positives, false negatives, precision, recall, and mean matched IoU. The default IoU threshold is `0.5`.

`evaluate_detection_benchmark` aggregates those counts over caller-supplied `DetectionBenchmarkSample` records and reports micro-averaged precision, recall, and matched IoU. Sample IDs must be unique, and empty benchmark inputs are rejected. This is an offline metric helper: callers must generate predictions using an explicitly chosen model/configuration and supply fixed annotations. It does not load images, fetch data, train a model, or run an automatic hardware check.

The helper validates that the IoU threshold is in `[0, 1]`. Empty denominators produce `0.0` for precision or recall.

## Limitations

- It is not a dataset runner, benchmark harness, or training/evaluation pipeline.
- It does not measure temporal stability, latency, memory, fairness, calibration, or speech/language quality.
- It uses greedy single-frame matching; it does not calculate average precision across confidence thresholds.
- Ground-truth annotations and model predictions must already use the same class IDs and normalized coordinate convention.
- The live pipeline's box validation, confidence filtering, temporal stabilizer, and short-lived world state are data-quality and presentation safeguards, not evidence that the underlying detector is more accurate.
- Current unit tests use synthetic detector records and do not constitute a real-camera accuracy benchmark.

Do not report a model as improved based only on this helper or on stabilized UI output. Compare model versions on a fixed, documented, session-separated held-out dataset.
