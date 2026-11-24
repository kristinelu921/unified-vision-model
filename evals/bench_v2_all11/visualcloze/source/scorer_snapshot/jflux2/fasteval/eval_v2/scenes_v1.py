"""eval v2 scene groups, v1: the unit within which bench_v2 images are not independent (for scene-grouped intervals).

Same video / sequence, scanner position, room or source photograph -> one group; datasets of independent photographs
get one group per record. Keys from pack metadata only (docs/review_postproc_2026-10-06/cross_cutting/scene_groups_v1
.json: 12 of 26 test packs are grouped; CamVid has 2 sequences and is reported per sequence instead)."""
from __future__ import annotations


def scene_key(rec: dict) -> str:
    rid = rec["record_id"]; sm = rec.get("source_metadata") or {}
    src = rid.split(":", 1)[0]
    if src == "gopro":                                   # gopro:<split>:GOPR0384_11_00:000001 -> the named sequence
        return f"gopro:{sm.get('sequence') or rid.split(':')[2]}"
    if src == "realblur_j":                              # realblur_j:test:scene002:10
        return f"realblur_j:{sm.get('scene') or rid.split(':')[2]}"
    if src == "sidd":                                    # sidd:validation:02_05 -> source image 02 (32 blocks each)
        return f"sidd:{sm.get('scene_instance') or rid.split(':')[2].split('_')[0]}"
    if src == "nyu_depth_v2" and sm.get("scene"):        # the named room (video) of the labelled frame
        return f"nyu:{sm['scene']}"
    if src == "diode":                                   # diode:00019_00183_indoors_000_010 -> scene 00019, scan 00183
        parts = rid.split(":", 1)[1].split("_")
        return f"diode:{parts[0]}_{parts[1]}"
    if src == "camvid":                                  # camvid:test:0001TP_008550 -> sequence 0001TP
        return f"camvid:{sm.get('sequence') or rid.split(':')[2].split('_')[0]}"
    if src == "flic":                                    # one movie = one group (the FLIC authors' own caution)
        return f"flic:{sm.get('movie') or rid}"
    return rid
