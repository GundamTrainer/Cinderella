# Evaluation

## Current helper

`ai/evaluation.py` provides `evaluate_detections` for one frame. It greedily matches predictions to remaining ground-truth boxes by class and IoU, sorted by prediction confidence. The result includes true positives, false positives, false negatives, precision, recall, and mean matched IoU. The default IoU threshold is `0.5`.

The helper validates that the IoU threshold is in `[0, 1]`. Empty denominators produce `0.0` for precision or recall.

## Limitations

- It is not a dataset runner, benchmark harness, or training/evaluation pipeline.
- It does not measure temporal stability, latency, memory, fairness, calibration, or speech/language quality.
- It uses greedy single-frame matching; it does not calculate average precision across confidence thresholds.
- Ground-truth annotations and model predictions must already use the same class IDs and normalized coordinate convention.

Do not report a model as improved based only on this helper or on stabilized UI output. Compare model versions on a fixed, documented, session-separated held-out dataset.
