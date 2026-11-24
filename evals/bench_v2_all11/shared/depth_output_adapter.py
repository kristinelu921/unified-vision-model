"""Versioned native-depth decoder for saved Painter/VICT RGB predictions.

Released depth inference maps output intensity linearly to metric depth.
The benchmark's depth metric takes inverse depth. This conversion needs no
target, demonstration endpoint, fitted parameter, or per-query decision.
It does not change the frozen benchmark metric or decode other model families.
"""
import numpy as np

CONTRACT = 'bench_v2_painter_vict_native_depth_decode_v1'
MODELS = frozenset({'painter', 'vict'})


def decode_depth_prediction(pred_u8, *, model):
    if model not in MODELS:
        raise ValueError(f'No established native-depth decoder for {model}')
    pred = np.asarray(pred_u8)
    if pred.dtype != np.uint8 or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError('Expected the original uint8 H×W×3 prediction, before any decoding')
    invalid = (pred == 0).all(axis=-1)
    depth = pred.astype(np.float64).mean(axis=-1) / 255.
    inverse = np.zeros_like(depth)
    np.divide(1., depth, out=inverse, where=~invalid)
    # A fixed positive unit factor is absorbed by the existing scale fit.
    # Preserve exact-black invalid pixels rather than filling or dropping them.
    return inverse, invalid
