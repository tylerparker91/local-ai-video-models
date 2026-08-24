# Local AI Video Studio model delivery

This repository contains only the reproducible export and release infrastructure for the Android application's versioned model package. Large ONNX models are published exclusively as GitHub Release assets and are never committed to Git history.

The workflow is fail-closed: it downloads pinned upstream checkpoints, exports the four Android-specific FP16 ONNX graphs, and publishes only when every byte count and SHA-256 matches `expected-models.json`.

DreamShaper is not rebuilt or re-hosted here because the application uses its proven commit-pinned upstream download.
