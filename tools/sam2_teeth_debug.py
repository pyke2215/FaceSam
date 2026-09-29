#!/usr/bin/env python3
"""Visualise MediaPipe-derived SAM 2 prompts and the resulting teeth mask."""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faceslim.models import ensure_model  # noqa: E402
from faceslim.pipeline import FaceWarpEngine, face_components  # noqa: E402
from faceslim.teeth_sam import (  # noqa: E402
    TeethSamSegmenter, mouth_inner_polygon, mouth_open_ratio,
)


def draw_overview(frame_rgb, lms, result):
    vis = frame_rgb.copy()
    cv2.polylines(vis, [mouth_inner_polygon(lms)], True, (255, 255, 0), 1, cv2.LINE_AA)
    if result is not None:
        x1, y1, x2, y2 = result.bounds
        cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 2)
    return vis


def draw_closeup(frame_rgb, lms, result, height):
    """Enlarged mouth crop with mask, inner-lip polygon, box, and prompt points."""
    x1, y1, x2, y2 = result.bounds
    s = height / float(y2 - y1)
    crop = cv2.resize(frame_rgb[y1:y2, x1:x2], (int(round((x2 - x1) * s)), height),
                      interpolation=cv2.INTER_CUBIC).astype(np.float32)
    m = cv2.resize(result.mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)[:, :, None]
    crop = crop * (1 - 0.5 * m) + np.array([60, 140, 255], np.float32) * 0.5 * m
    vis = np.clip(crop, 0, 255).astype(np.uint8)
    off = np.array([x1, y1], np.float32)
    to_view = lambda p: tuple(np.round((np.asarray(p, np.float32) - off) * s).astype(int))
    poly = np.array([to_view(p) for p in mouth_inner_polygon(lms)], np.int32)
    cv2.polylines(vis, [poly], True, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.rectangle(vis, to_view(result.box[:2]), to_view(result.box[2:]), (255, 200, 0), 1)
    for p, label in zip(result.points, result.labels):
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        cv2.circle(vis, to_view(p), 6, color, -1, cv2.LINE_AA)
        cv2.circle(vis, to_view(p), 6, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("-o", "--output", help="Output PNG (default: <image>_sam2_teeth.png)")
    parser.add_argument("--no-closeup", action="store_true", help="Skip the enlarged mouth panel")
    args = parser.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"Cannot read {args.image}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if not ensure_model():
        raise SystemExit("MediaPipe face landmarker unavailable")

    engine = FaceWarpEngine("image", max_faces=1)
    faces = engine.detect(rgb)
    if not faces:
        raise SystemExit("No face detected")
    lms, _conf, _bs = face_components(faces[0])

    segmenter = TeethSamSegmenter()
    started = time.perf_counter()
    result = segmenter.segment(rgb, lms)
    elapsed = (time.perf_counter() - started) * 1000.0
    print(f"mouth open ratio: {mouth_open_ratio(lms):.3f}")
    if result is None:
        print("mouth closed -> SAM skipped, no teeth mask")
    else:
        print(f"SAM 2 on {segmenter.device}: {elapsed:.0f} ms | prompts: "
              f"{int((result.labels == 1).sum())} positive, {int((result.labels == 0).sum())} negative | "
              f"mask pixels: {int(result.mask.sum())}")

    vis = draw_overview(rgb, lms, result)
    if result is not None and not args.no_closeup:
        vis = np.hstack([vis, draw_closeup(rgb, lms, result, vis.shape[0])])
    out = args.output or str(Path(args.image).with_name(Path(args.image).stem + "_sam2_teeth.png"))
    cv2.imwrite(out, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    engine.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
