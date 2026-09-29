"""FaceSlim package."""

try:
    # torch must load before PyQt5 on Windows; see FaceSlim_v1.py.
    import torch  # noqa: F401
except Exception:
    pass

from .runtime import VERSION
