"""Lossless published-palette decoding, otherwise frozen v1 post-processing.

The renderer and episode coder publish a global 1024-colour DISTINCT_PALETTE.
When EVERY prediction pixel is exactly a code in that public format, decode its
RGB code losslessly: one label per colour, black = 0, without morphology, area
thresholds or Lab merges. Thin/small/disconnected objects are representable and
must survive a codec roundtrip. Geometry can still be incorrect: shifted,
merged or scrambled palette predictions receive their ordinary native scores.

This format test takes only the prediction bytes and the global public codebook;
it never uses a target, GT count, dataset, input image or episode vocabulary.
For ANY off-code pixel, use postprocess_v1.seg_label_map unchanged. This retains
its established continuous-RGB/noisy-output behaviour. The image-wide format
switch is discontinuous: success on exact palette rasters does not establish
robust decoding of noisy thin structures or arbitrary recolouring.

Point and dense-field helpers are exact v1 aliases. Keypoint rounding, overwritten
or merged discs and dense-field rendering limits are not reconstructed here.
The previous blend candidate and its unsuccessful real-RGB controls are saved
in docs/eval_v2_postprocess_validation_2026-10-08/decoder_candidates/.
"""
from __future__ import annotations

import numpy as np

from tools import bench_episode_v1 as _codebook
from . import postprocess_v1 as _v1

BLACK_MAX = _v1.BLACK_MAX
SEG_DELTA_E = _v1.SEG_DELTA_E
SEG_CORE_WIDTH = _v1.SEG_CORE_WIDTH
SEG_CORE_MIN_PX = _v1.SEG_CORE_MIN_PX
SEG_MIN_FRAGMENT_PX = _v1.SEG_MIN_FRAGMENT_PX
POINT_MIN_BLOB_PX = _v1.POINT_MIN_BLOB_PX
EIGHT = _v1.EIGHT
rgb_to_lab = _v1.rgb_to_lab
is_black = _v1.is_black
relabel_first_appearance = _v1.relabel_first_appearance
point_blobs = _v1.point_blobs
depth_grey = _v1.depth_grey
decode_normals_rgb = _v1.decode_normals_rgb
normals_unit = _v1.normals_unit
luma01 = _v1.luma01
inpaint_composite = _v1.inpaint_composite

SEG_PROTOCOL = 'exact_public_palette_else_frozen_v1'
CODEBOOK_SOURCE = 'tools/bench_episode_v1.py'


def _rgb(pred_u8: np.ndarray) -> np.ndarray:
    a = np.asarray(pred_u8)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError('seg_label_map requires an HWC RGB uint8 prediction')
    return a


def is_exact_palette(pred_u8: np.ndarray) -> bool:
    """Whether every RGB code is in the PUBLIC global codebook, prediction only."""
    positions, _ = _codebook.predecode(_rgb(pred_u8))
    return bool((positions >= 0).all())


def seg_label_map(pred_u8: np.ndarray, *, delta_e: float = SEG_DELTA_E,
                  core_width: int = SEG_CORE_WIDTH, core_min_px: int = SEG_CORE_MIN_PX,
                  min_fragment_px: int = SEG_MIN_FRAGMENT_PX) -> np.ndarray:
    """Exact public palette -> lossless labels; otherwise delegate v1 unchanged.

    All 1024 global codes are eligible, irrespective of which GT labels or demo
    colours exist. No nearest-code snapping or tolerance is used. Semantic
    classes and occluded instances retain one label across disconnected pieces;
    identical-colour instances remain indistinguishable. GT semantic/source void
    masks and instance background penalties remain the metric's responsibility.
    """
    a = _rgb(pred_u8)
    if not np.isfinite(delta_e) or delta_e < 0:
        raise ValueError('delta_e must be finite and non-negative')
    if (not isinstance(core_width, (int, np.integer)) or core_width < 1 or
            not isinstance(core_min_px, (int, np.integer)) or core_min_px < 1 or
            not isinstance(min_fragment_px, (int, np.integer)) or min_fragment_px < 0):
        raise ValueError('core dimensions must be positive integers and fragment size non-negative')
    positions, _ = _codebook.predecode(a)
    if bool((positions >= 0).all()):
        return relabel_first_appearance(positions)
    return _v1.seg_label_map(a, delta_e=delta_e, core_width=core_width,
                             core_min_px=core_min_px, min_fragment_px=min_fragment_px)
