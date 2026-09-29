#!/usr/bin/env python3
"""SAM 2 teeth segmentation prompted by MediaPipe face mesh landmarks."""

import contextlib
import os
import threading
import time

import cv2
import numpy as np

from .models import SAM2_MODEL, SAM2_PATH, ensure_sam2_model

# Closed inner-lip polygon: upper inner lip 78 -> 308, then lower inner lip 324 -> 95.
MOUTH_INNER_POLYGON = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                       324, 318, 402, 317, 14, 87, 178, 88, 95]
MIN_MOUTH_OPEN_RATIO = 0.07
WIDE_OPEN_RATIO = 0.30


def mouth_open_ratio(lms):
    mouth_w = float(np.linalg.norm(lms[308] - lms[78]))
    if mouth_w < 1.0:
        return 0.0
    return float(np.linalg.norm(lms[14] - lms[13])) / mouth_w


def mouth_inner_polygon(lms):
    return np.round(lms[MOUTH_INNER_POLYGON]).astype(np.int32)


def teeth_sam_prompts(lms, frame_rgb, min_open=MIN_MOUTH_OPEN_RATIO):
    """Build SAM prompts for the visible teeth.

    Returns (points Nx2 float32, labels N int32, box [x1, y1, x2, y2] float32)
    in frame pixel coordinates, or None when the mouth is closed.
    """
    if len(lms) < 468:
        return None
    ratio = mouth_open_ratio(lms)
    if ratio < min_open:
        return None

    def lerp(a, b, t):
        return lms[a] + (lms[b] - lms[a]) * t

    h, w = frame_rgb.shape[:2]
    poly = mouth_inner_polygon(lms)
    x1, y1 = poly.min(axis=0)
    x2, y2 = poly.max(axis=0)
    box = np.array([max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)], np.float32)

    gap = float(np.linalg.norm(lms[14] - lms[13]))
    region = np.zeros((h, w), np.uint8)
    cv2.fillPoly(region, [poly], 1)
    k = max(3, int(gap * 0.15)) | 1
    core = cv2.erode(region, np.ones((k, k), np.uint8)) > 0
    inside = region > 0
    lum = cv2.GaussianBlur(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)[:, :, 0], (5, 5), 0)

    pos, neg = [], []
    if inside.any():
        lum_floor = float(np.percentile(lum[inside], 40))
        for p in (lerp(13, 14, 0.25), lerp(82, 87, 0.25), lerp(312, 317, 0.25)):
            x, y = int(round(p[0])), int(round(p[1]))
            if 0 <= x < w and 0 <= y < h and inside[y, x] and lum[y, x] >= lum_floor:
                pos.append(p)
        search = core if core.any() else inside
        by, bx = np.unravel_index(np.argmax(np.where(search, lum, 0)), lum.shape)
        pos.append(np.array([bx, by], np.float64))
        if ratio > WIDE_OPEN_RATIO:
            dy, dx = np.unravel_index(np.argmin(np.where(search, lum, 255)), lum.shape)
            neg.append(np.array([dx, dy], np.float64))
    else:
        pos.append(lerp(13, 14, 0.25))

    neg += [lerp(0, 13, 0.5), lerp(17, 14, 0.5), lms[61], lms[291],
            lerp(2, 0, 0.5), lerp(17, 152, 0.4)]

    points = np.array(pos + neg, np.float32)
    labels = np.array([1] * len(pos) + [0] * len(neg), np.int32)
    return points, labels, box


def mouth_crop_bounds(lms, h, w, scale=1.8):
    """Square crop around the mouth so SAM sees the teeth at high resolution."""
    center = (lms[61] + lms[291] + lms[0] + lms[17]) / 4.0
    half = max(16.0, float(np.linalg.norm(lms[291] - lms[61])) * scale / 2.0)
    x1 = int(max(0, center[0] - half)); y1 = int(max(0, center[1] - half))
    x2 = int(min(w, center[0] + half)); y2 = int(min(h, center[1] + half))
    return x1, y1, x2, y2


class TeethSamResult:
    __slots__ = ("bounds", "mask", "points", "labels", "box", "anchor")

    def __init__(self, bounds, mask, points, labels, box, anchor):
        self.bounds = bounds      # (x1, y1, x2, y2) crop in frame coords
        self.mask = mask          # float32 crop-sized mask in [0, 1]
        self.points = points      # prompt points in frame coords
        self.labels = labels
        self.box = box
        self.anchor = anchor      # mouth center when the mask was computed


def prepare_mouth_crop(frame_rgb, lms):
    """Copy the mouth crop for SAM. Returns (crop, local_lms, offset) or None if the mouth is closed."""
    if len(lms) < 468 or mouth_open_ratio(lms) < MIN_MOUTH_OPEN_RATIO:
        return None
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = mouth_crop_bounds(lms, h, w)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    offset = np.array([x1, y1], np.float64)
    return np.ascontiguousarray(frame_rgb[y1:y2, x1:x2]).copy(), lms - offset, offset


class TeethSamSegmenter:
    """SAM 2 image predictor specialised for teeth. Thread-safe; use shared_segmenter()."""

    def __init__(self, device=None):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if not ensure_sam2_model():
            raise RuntimeError(f"{SAM2_MODEL['label']} checkpoint unavailable")
        self._torch = torch
        self._lock = threading.Lock()
        self.warmed = False
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._autocast_dtype = None
        if self.device == "cpu":
            # Leave half the cores for landmarks, parsing, and warp on the preview thread.
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        if self.device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
            # Native bf16/TF32 only from Ampere (RTX 30xx) on; older GPUs emulate bf16 slowly.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self._autocast_dtype = torch.bfloat16
        model = build_sam2(SAM2_MODEL["config"], SAM2_PATH, device=self.device)
        self.predictor = SAM2ImagePredictor(model)
        precision = "bf16" if self._autocast_dtype is not None else "fp32"
        print(f"  SAM 2 teeth segmenter loaded ({SAM2_MODEL['label']}, {self.device}, {precision})")

    def _autocast(self):
        if self._autocast_dtype is None:
            return contextlib.nullcontext()
        return self._torch.autocast("cuda", dtype=self._autocast_dtype)

    def warmup(self, runs=1, size=384):
        """Run SAM on a dummy crop; returns the mean latency in ms of the timed runs."""
        rng = np.random.default_rng(0)
        crop = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        box = np.array([size * 0.25, size * 0.4, size * 0.75, size * 0.6], np.float32)
        timings = []
        for _ in range(max(0, runs) + 1):
            started = time.perf_counter()
            with self._lock, self._torch.inference_mode(), self._autocast():
                self.predictor.set_image(crop)
                self.predictor.predict(box=box, multimask_output=False)
            if self.device == "cuda":
                self._torch.cuda.synchronize()
            timings.append((time.perf_counter() - started) * 1000.0)
        self.warmed = True
        return float(np.mean(timings[1:])) if len(timings) > 1 else timings[0]

    def segment(self, frame_rgb, lms):
        """Return TeethSamResult, or None when the mouth is closed."""
        prepared = prepare_mouth_crop(frame_rgb, lms)
        return None if prepared is None else self.segment_prepared(*prepared)

    def segment_prepared(self, crop, local_lms, offset):
        prompts = teeth_sam_prompts(local_lms, crop)
        if prompts is None:
            return None
        points, labels, box = prompts
        ch, cw = crop.shape[:2]
        keep = ((points[:, 0] >= 0) & (points[:, 0] < cw) & (points[:, 1] >= 0) & (points[:, 1] < ch)) | (labels == 1)
        points, labels = points[keep], labels[keep]

        with self._lock, self._torch.inference_mode(), self._autocast():
            self.predictor.set_image(crop)
            masks, _scores, _ = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=box,
                multimask_output=False,
            )

        region = np.zeros((ch, cw), np.uint8)
        cv2.fillPoly(region, [mouth_inner_polygon(local_lms)], 1)
        grow = max(3, int(np.linalg.norm(local_lms[308] - local_lms[78]) * 0.04)) | 1
        region = cv2.dilate(region, np.ones((grow, grow), np.uint8))
        mask = ((masks[0] > 0.5) & (region > 0)).astype(np.float32)
        x1, y1 = int(offset[0]), int(offset[1])
        off32 = offset.astype(np.float32)
        anchor = (local_lms[13] + local_lms[14]) / 2.0 + offset
        return TeethSamResult((x1, y1, x1 + cw, y1 + ch), mask, points + off32, labels,
                              box + np.tile(off32, 2), anchor)


_shared_lock = threading.Lock()
_shared_segmenter = None


def shared_segmenter():
    """One SAM 2 model per process, reused across engines (webcam restarts, video loops)."""
    global _shared_segmenter
    with _shared_lock:
        if _shared_segmenter is None:
            _shared_segmenter = TeethSamSegmenter()
        return _shared_segmenter


class AsyncTeethSam:
    """Runs SAM 2 on a worker thread so live preview never waits for it.

    Callers submit the newest frame every tick; older pending frames are dropped,
    so the worker always segments the most recent mouth crop.
    """

    def __init__(self, factory=shared_segmenter):
        self._factory = factory
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._pending = {}
        self._results = {}
        self._stopped = False
        self.failed = False
        self.last_ms = None
        self._thread = threading.Thread(target=self._run, name="faceslim-teeth-sam", daemon=True)
        self._thread.start()

    def submit(self, face_idx, frame_rgb, lms):
        prepared = prepare_mouth_crop(frame_rgb, lms)
        with self._lock:
            if prepared is None:
                self._pending.pop(face_idx, None)
                self._results[face_idx] = None
                return
            self._pending[face_idx] = prepared
        self._wake.set()

    def latest(self, face_idx):
        """Return (has_result, TeethSamResult or None)."""
        with self._lock:
            return face_idx in self._results, self._results.get(face_idx)

    def close(self):
        self._stopped = True
        self._wake.set()

    def _run(self):
        try:
            segmenter = self._factory()
            if not segmenter.warmed:
                segmenter.warmup(runs=0)
        except Exception as e:
            self.failed = True
            print(f"  SAM 2 teeth segmenter unavailable: {e} (using face parsing mask)")
            return
        while not self._stopped:
            self._wake.wait(0.5)
            self._wake.clear()
            while not self._stopped:
                with self._lock:
                    if not self._pending:
                        break
                    face_idx, prepared = self._pending.popitem()
                started = time.perf_counter()
                try:
                    result = segmenter.segment_prepared(*prepared)
                except Exception as e:
                    print(f"SAM 2 teeth segmentation error: {e}")
                    continue
                self.last_ms = (time.perf_counter() - started) * 1000.0
                with self._lock:
                    self._results[face_idx] = result


def paste_result_into_roi(result, roi_bounds, roi_shape, lms=None, feather=5):
    """Map a TeethSamResult onto a face ROI, shifting by mouth motion when lms are given."""
    mask = np.zeros(roi_shape[:2], np.float32)
    if result is None:
        return mask
    dx = dy = 0
    if lms is not None:
        shift = (lms[13] + lms[14]) / 2.0 - result.anchor
        dx, dy = int(round(shift[0])), int(round(shift[1]))
    cx1, cy1 = result.bounds[:2]
    rx1, ry1 = roi_bounds[0], roi_bounds[1]
    tx1, ty1 = cx1 + dx - rx1, cy1 + dy - ry1
    th, tw = result.mask.shape
    sx1, sy1 = max(0, -tx1), max(0, -ty1)
    ex1, ey1 = max(0, tx1), max(0, ty1)
    ex2 = min(roi_shape[1], tx1 + tw); ey2 = min(roi_shape[0], ty1 + th)
    if ex2 > ex1 and ey2 > ey1:
        mask[ey1:ey2, ex1:ex2] = result.mask[sy1:sy1 + (ey2 - ey1), sx1:sx1 + (ex2 - ex1)]
    if feather > 0 and mask.any():
        k = max(3, feather) | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask
