#!/usr/bin/env python3
"""Report whether this machine can run FaceSlim's full pipeline, GPU warp, and SAM 2 teeth mask."""

import argparse
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def line(ok, label, detail=""):
    mark = "OK  " if ok else ("WARN" if ok is None else "FAIL")
    print(f"  [{mark}] {label}" + (f": {detail}" if detail else ""))
    return ok


def check_python():
    v = sys.version_info
    ok = (3, 10) <= (v.major, v.minor) <= (3, 12)
    line(ok, "Python", f"{platform.python_version()} ({sys.executable})"
         + ("" if ok else " - use Python 3.10-3.12, 3.11 is the tested version"))
    return ok


def check_core_packages():
    import faceslim  # noqa: F401  (loads torch before PyQt5, exactly like the app)
    ok = True
    for module, label in (("PyQt5.QtCore", "PyQt5"), ("cv2", "OpenCV"), ("mediapipe", "MediaPipe"),
                          ("numpy", "NumPy"), ("scipy", "SciPy"), ("onnxruntime", "ONNX Runtime")):
        try:
            mod = __import__(module, fromlist=["_"])
            version = getattr(mod, "__version__", "") or getattr(mod, "PYQT_VERSION_STR", "")
            line(True, label, version)
        except Exception as e:
            ok = line(False, label, str(e))
    try:
        import onnxruntime as ort
        line(True, "ONNX providers", ", ".join(ort.get_available_providers()))
    except Exception:
        pass
    return ok


def check_torch():
    try:
        import torch
    except Exception as e:
        line(False, "PyTorch", f"not installed ({e}); GPU warp and SAM 2 unavailable")
        return False, False
    cuda = torch.cuda.is_available()
    line(True, "PyTorch", f"{torch.__version__} (CUDA build: {torch.version.cuda or 'none'})")
    if not cuda:
        line(None, "CUDA GPU", "not available - FaceSlim and SAM 2 run on CPU")
        return True, False
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    try:
        x = torch.randn(256, 256, device="cuda")
        (x @ x).sum().item()
        precision = "bf16" if cap[0] >= 8 else "fp32"
        line(True, "CUDA GPU", f"{name}, compute {cap[0]}.{cap[1]}, SAM 2 precision {precision}")
        return True, True
    except Exception as e:
        line(False, "CUDA GPU", f"{name} detected but kernels fail ({e}); reinstall torch for this GPU")
        return True, False


def check_models():
    from faceslim.models import model_inventory
    required = {"face_landmarker", "bisenet_resnet18", "sam2.1_hiera_tiny"}
    ok = True
    for item in model_inventory():
        if item["key"] not in required:
            continue
        verified = item["verified"]
        ok &= bool(line(verified, f"Model {item['label']}", item["status"]))
    return ok


def check_sam2(benchmark):
    try:
        import sam2
        line(True, "sam2 package", os.path.dirname(sam2.__file__))
    except Exception as e:
        line(False, "sam2 package", f"{e} - run tools/setup_sam2.py")
        return False
    if not benchmark:
        return True
    try:
        from faceslim.teeth_sam import shared_segmenter
        seg = shared_segmenter()
        ms = seg.warmup(runs=5)
        fps_note = "live preview updates the mask about every frame" if ms < 50 else \
            "live preview mask lags behind; still usable, runs on a background thread"
        line(True, "SAM 2 benchmark", f"{ms:.0f} ms per mouth crop on {seg.device} ({fps_note})")
        return True
    except Exception as e:
        line(False, "SAM 2 load", str(e))
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-benchmark", action="store_true", help="Skip loading SAM 2 and timing it.")
    args = parser.parse_args()

    print("FaceSlim environment check")
    results = [check_python(), check_core_packages()]
    has_torch, _has_cuda = check_torch()
    results.append(has_torch)
    results.append(check_models())
    if has_torch:
        results.append(check_sam2(not args.no_benchmark))
    ok = all(r is not False for r in results)
    print("Ready." if ok else "Some checks failed; see docs/SETUP_WINDOWS_GPU.md (Troubleshooting).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
