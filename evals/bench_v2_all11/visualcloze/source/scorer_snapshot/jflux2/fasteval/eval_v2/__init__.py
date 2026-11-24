"""eval v2 (2026-10-06): the bench_v2 evaluator the user specified after the post-processing review.

"we should accept RGB image output paired with the actual ground truth data, post-process the RGB output to the
actual metric-calculated format, then calculate against the ground truth in standard format" (the user, 2026-10-06),
with CLASS-AGNOSTIC metrics ("for each same class segments, regard as binary segmentation") and the decisions of
2026-10-06: switch the metrics of record, standard normalisers, normals charged 180 deg where the prediction is invalid,
GT sidecars, normals / inpainting evaluated as their fields do. Design: docs/review_postproc_2026-10-06/REPORT.md.

Modules (versioned; never edited in place once results depend on them):
  postprocess_v1  model RGB -> the metric's native format (label maps, points, inverse depth, unit normals, ...)
  gt_v1           ground truth in standard format, from the target tiles and the bench_v2_gt_v1 sidecars
  metrics_v1      the standard metrics, class- and identity-agnostic where the task has labels
"""
