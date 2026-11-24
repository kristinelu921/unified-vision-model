"""Explicit IMProv edge postprocessing experiment; not a native-decoder fix.

Apply one fixed formula to every saved edge prediction, before the unchanged
99-threshold BSDS scorer. No targets, fitted parameters or extra inputs.
"""
import numpy as np

CONTRACT = 'bench_v2_improv_edge_sqrt_contrast_v1'


def improv_edge_confidence(rgb_u8):
    a = np.asarray(rgb_u8)
    if a.dtype != np.uint8 or a.shape != (256, 256, 3):
        raise ValueError('Expected the original uint8 [256, 256, 3] prediction')
    a = a.astype(np.float64) / 255.0
    luma = .299*a[..., 0] + .587*a[..., 1] + .114*a[..., 2]
    return np.sqrt(np.clip(luma, 0., 1.))
