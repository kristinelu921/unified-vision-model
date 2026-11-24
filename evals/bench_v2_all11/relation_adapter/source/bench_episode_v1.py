"""Pure NumPy packed-episode selection and v4 color coding.

The active bench path codes palette labels in demo-first, row-major first
appearance order. All other tasks use CID 0, preserving their uint8 targets.
These are static copies of the original palette construction and selection
rules, with bounded integer position reuse in place of repeated float decoding.
Only RGB position maps are retained; every AE inference remains fresh.
"""
from __future__ import annotations

from collections import OrderedDict
import zlib

import numpy as np

FAMILY_GRAY = ("metric_depth", "relative_depth", "saliency_detection", "edge_detection", "shadow_detection")
FAMILY_NORMALS = ("surface_normals",)
FAMILY_PALETTE = ("semantic_segmentation", "instance_segmentation", "panoptic_segmentation",
                  "human_pose_2d", "face_landmarks", "object_keypoints")
FAMILY_FIXED = ("colorization", "denoising", "deblurring", "inpainting", "super_resolution")
_FAMILY_OF = {**{t: "gray" for t in FAMILY_GRAY}, **{t: "normals" for t in FAMILY_NORMALS},
              **{t: "palette" for t in FAMILY_PALETTE}, **{t: "fixed" for t in FAMILY_FIXED}}
PALETTE_MIN_PREFIX_DEFAULT = 16
_STATE = {"prefix": PALETTE_MIN_PREFIX_DEFAULT}


def _srgb_to_lab(rgb_u8):
    # Original rgb_supervision_v4.py:307, preserving float64 construction.
    c = rgb_u8.astype(np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return np.stack([116.0*f[:, 1]-16.0, 500.0*(f[:, 0]-f[:, 1]), 200.0*(f[:, 1]-f[:, 2])], axis=1)


def _build_distinct_palette(size=1024, levels=16):
    # Original rgb_supervision_v4.py:317, including tie order and black at 0.
    g = np.linspace(0, 255, levels).round().astype(np.uint8)
    cand = np.array([[r, gg, b] for r in g for gg in g for b in g], np.uint8)
    cand = cand[np.any(cand > 40, axis=1)]
    lab = _srgb_to_lab(cand)
    black = _srgb_to_lab(np.zeros((1, 3), np.uint8))[0]
    d = np.linalg.norm(lab-black, axis=1)
    chosen = [int(np.argmax(d))]
    mind = np.minimum(np.linalg.norm(lab-lab[chosen[0]], axis=1), d)
    while len(chosen) < size-1:
        nxt = int(np.argmax(mind)); chosen.append(nxt)
        mind = np.minimum(mind, np.linalg.norm(lab-lab[nxt], axis=1))
    return np.concatenate([np.zeros((1, 3), np.uint8), cand[chosen]], axis=0)


DISTINCT_PALETTE = _build_distinct_palette()
DISTINCT_PALETTE.flags.writeable = False
_DP_F32 = DISTINCT_PALETTE.astype(np.float32)/255.0
_DP_PACKED = (DISTINCT_PALETTE[:, 0].astype(np.int64)*65536
              + DISTINCT_PALETTE[:, 1].astype(np.int64)*256 + DISTINCT_PALETTE[:, 2].astype(np.int64))
_DP_SORTED_KEYS = np.sort(_DP_PACKED)
_DP_SORTED_POS = np.argsort(_DP_PACKED).astype(np.int64)


def family_of(task):
    try:
        return _FAMILY_OF[str(task)]
    except KeyError:
        raise ValueError(f"no convention family for task {task!r}; known: {sorted(_FAMILY_OF)}") from None


def convention_pools(task):
    fam = family_of(task)
    ids = {"gray": tuple(range(14)), "normals": tuple(range(48)), "palette": tuple(range(4096)), "fixed": (0,)}[fam]
    ev = {"gray": (12, 13), "normals": (1, 8), "palette": (0,), "fixed": ()}[fam]
    return tuple(i for i in ids if i not in ev), ev


def _packed(m8):
    m8 = np.asarray(m8, np.int64)
    return m8[..., 0]*65536 + m8[..., 1]*256 + m8[..., 2]


def _posmap(packed):
    p = np.asarray(packed, np.int64)
    i = np.clip(np.searchsorted(_DP_SORTED_KEYS, p), 0, len(_DP_SORTED_KEYS)-1)
    return np.where(_DP_SORTED_KEYS[i] == p, _DP_SORTED_POS[i], -1).astype(np.int64)


def predecode(tile):
    a = np.asarray(tile)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError("palette predecode requires uint8 HWC RGB")
    pos = _posmap(_packed(a)).astype(np.int16)
    positive = pos.reshape(-1)
    positive = positive[positive > 0]
    if positive.size:
        _, first = np.unique(positive, return_index=True)
        labels = positive[np.sort(first)]
    else:
        labels = np.empty(0, np.int16)
    pos.flags.writeable = False; labels.flags.writeable = False
    return pos, labels


def recolor(tiles, decoded, cid=0, prefix=PALETTE_MIN_PREFIX_DEFAULT):
    """Original v4 palette mapping applied to integer positions, preserving RGB."""
    order, seen = [], set()
    for _, labels in decoded:
        for label in labels:
            label = int(label)
            if label not in seen:
                seen.add(label); order.append(label)
    lut = np.arange(len(DISTINCT_PALETTE), dtype=np.int16)
    if order:
        if int(cid) == 0:
            assigned = np.arange(1, len(order)+1)
        else:
            p = min(max(len(order), int(prefix)), len(DISTINCT_PALETTE)-1)
            assigned = np.random.default_rng([int(cid), 0xD15C0]).permutation(np.arange(1, p+1))[:len(order)]
        lut[np.asarray(order)] = assigned
    return [np.where(pos[..., None] >= 0, DISTINCT_PALETTE[lut[np.clip(pos, 0, None)]], tile).astype(np.uint8)
            for tile, (pos, _) in zip(tiles, decoded)]


class PaletteCoder:
    """Bounded derived RGB positions for one immutable task array.

    At 256px, the default maximum of 128 position maps uses 16 MiB plus
    small first-appearance lists. Eviction affects only CPU coding cost.
    No image encoding, model state or learned value is stored here.
    """
    def __init__(self, masks, max_rows=128):
        if type(max_rows) is not int or max_rows < 1:
            raise ValueError("palette position limit must be positive")
        self.masks, self.max_rows = masks, max_rows
        self._cache = OrderedDict()
        self.hits = self.misses = 0

    def _row(self, row):
        row = int(row)
        if self.masks.flags.writeable:
            raise ValueError("palette position reuse requires immutable target arrays")
        if row in self._cache:
            self.hits += 1
            self._cache.move_to_end(row)
            return self._cache[row]
        self.misses += 1
        decoded = predecode(self.masks[row])
        self._cache[row] = decoded
        while len(self._cache) > self.max_rows:
            self._cache.popitem(last=False)
        return decoded

    def code_rows(self, rows, cid=0):
        rows = list(rows)
        return recolor([self.masks[int(i)] for i in rows], [self._row(i) for i in rows], cid)

    def stats(self):
        return {"retained_rows": len(self._cache), "max_rows": self.max_rows,
                "position_bytes": sum(pos.nbytes + labels.nbytes for pos, labels in self._cache.values()),
                "hits": self.hits, "misses": self.misses, "ae_values_retained": 0}


def task_seed(task):
    return zlib.crc32(task.encode("utf-8")) & 0xFFFF


def draw_demo_indices(n_records, query, p, rng, candidates=None):
    pool = range(n_records) if candidates is None else candidates
    others = np.array([i for i in pool if i != query])
    if others.size < p:
        others = np.array([i for i in range(n_records) if i != query])
    return rng.choice(others, size=p, replace=False)


def to_chw01(hwc_u8):
    return np.transpose(np.asarray(hwc_u8, np.float32)/255.0, (2, 0, 1))


def to_hwc_u8(chw01):
    return np.rint(np.clip(np.transpose(np.asarray(chw01, np.float32), (1, 2, 0)), 0, 1)*255).astype(np.uint8)


def code_episode(tiles_u8, task, cid=0):
    tiles = [np.asarray(t, np.uint8) for t in tiles_u8]
    if not tiles:
        return tiles
    try:
        fam = family_of(task)
    except (KeyError, ValueError):
        fam = "unknown"
    if fam == "palette":
        return recolor(tiles, [predecode(t) for t in tiles], cid, _STATE["prefix"])
    if int(cid) == 0 or fam == "fixed":
        return tiles
    # These helpers retain the existing gray/normals variants for callers.
    if fam == "gray":
        channels = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                             [1, 1, 0], [1, 0, 1], [0, 1, 1]], np.float32)
        return [np.rint(((1.0-to_float(t)) if int(cid)%2 else to_float(t))*channels[int(cid)//2]*255).astype(np.uint8)
                for t in tiles]
    if fam == "normals":
        perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
        out = []
        for tile in tiles:
            a = to_float(tile)[..., list(perms[int(cid)//8])]
            for c in range(3):
                if (int(cid)%8 >> c) & 1:
                    a[..., c] = 1.0-a[..., c]
            out.append(np.rint(a*255).astype(np.uint8))
        return out
    raise ValueError(f"unsupported nonzero convention for {task!r}")


def to_float(tile):
    return np.asarray(tile, np.float32)/255.0


def build_conditions(imgs, msks, task, q, p, rng, *, cid, which, candidates=None, palette_coder=None):
    """The accelerator path's original real-demo condition, with exact coding."""
    if tuple(which) != ("real",):
        raise ValueError("packed accelerator episodes require real paired demonstrations")
    di = draw_demo_indices(len(imgs), q, p, rng, candidates)
    rows = [int(i) for i in di] + [int(q)]
    if palette_coder is not None:
        if palette_coder.masks is not msks or family_of(task) != "palette":
            raise ValueError("palette coder belongs to another target array or task")
        ep = palette_coder.code_rows(rows, cid)
    else:
        ep = code_episode([msks[j] for j in rows], task, cid)
    real = [(to_chw01(imgs[j]), to_chw01(t)) for j, t in zip(di, ep[:-1])]
    return {"real": real}, di, ep[-1]
