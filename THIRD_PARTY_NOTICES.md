# Third-party notices

FaceSlim v1.29.1 keeps its own source under the MIT license in [LICENSE](LICENSE). That permission does not replace the licenses of bundled dependencies.

## Windows binary status

The v1.29.1 executable is built and tested locally but is not published. A redistribution review found that [PyQt5 is licensed under GPLv3 or a commercial agreement](https://riverbankcomputing.com/software/pyqt/), while [pyvirtualcam 0.15.0 declares GPLv2](https://pypi.org/project/pyvirtualcam/0.15.0/). The pyvirtualcam maintainer also explains the inherited GPL code in [the upstream licensing discussion](https://github.com/letmaik/pyvirtualcam/issues/4).

No commercial PyQt license or explicit permission resolving this combination has been established for this release. The release does not assume that GPLv2 means GPLv2-or-later. Binary publication is paused until that point is clarified or a separately reviewed packaging change removes the conflict. No existing feature has been removed from the source.

The source archive includes the application and build instructions. Python 3.11 is the tested setup. A local build is not evidence that a combined binary may be redistributed under MIT alone.

## Main dependencies

| Package | Role | Upstream terms |
| --- | --- | --- |
| PyQt5 5.15.11 | Desktop interface | GPLv3 or commercial |
| PyQt5-Qt5 | Qt runtime | LGPLv3 package; bundled component notices also apply |
| pyvirtualcam 0.15.0 | Optional virtual-camera output | GPLv2 declaration |
| MediaPipe 0.10.35 | Face landmarks | Apache-2.0 |
| OpenCV contrib 4.11.0.86 | Image and video processing | Apache-2.0; wheel includes third-party notices |
| Pillow 12.3.0 | Image formats and metadata | MIT-CMU |
| NumPy 1.26.4 and SciPy 1.13.1 | Numerical processing | BSD-3-Clause; wheel notices also apply |
| ONNX Runtime 1.18.1 | Optional model inference | MIT |
| Protobuf 6.33.5 | Runtime support | BSD-3-Clause |
| SAM 2 (facebookresearch/sam2), PyTorch | Optional teeth segmentation, installed by `tools/setup_sam2.py` | Apache-2.0; PyTorch BSD-3-Clause |

This table summarizes the main direct dependencies, not every file in a packaged executable. Preserve each dependency's complete license and bundled notices when distributing it. Downloaded model weights have separate licenses listed in the [model inventory](README.md#models).
