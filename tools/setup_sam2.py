#!/usr/bin/env python3
"""Clone, install, and verify SAM 2 for FaceSlim's teeth segmentation."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAM2_DIR = ROOT / "third_party" / "sam2"
SAM2_REPO = "https://github.com/facebookresearch/sam2.git"
SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"


def run(cmd, env=None):
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], check=True, cwd=ROOT, env=env)


def clone_sam2(commit=SAM2_COMMIT):
    if not (SAM2_DIR / ".git").exists():
        SAM2_DIR.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", SAM2_REPO, SAM2_DIR])
    run(["git", "-C", SAM2_DIR, "fetch", "--depth", "1", "origin", commit])
    run(["git", "-C", SAM2_DIR, "checkout", "--detach", commit])


def install_sam2(python):
    env = os.environ.copy()
    # The optional CUDA extension only speeds up mask hole-filling, which FaceSlim does not use.
    env["SAM2_BUILD_CUDA"] = "0"
    env["SAM2_BUILD_ALLOW_ERRORS"] = "1"
    run([python, "-m", "pip", "install", "--no-build-isolation", "-e", SAM2_DIR], env=env)


def verify(python):
    code = (
        "import torch\n"
        "from faceslim.models import ensure_sam2_model\n"
        "assert ensure_sam2_model(), 'checkpoint download failed'\n"
        "from faceslim.teeth_sam import TeethSamSegmenter\n"
        "seg = TeethSamSegmenter()\n"
        "print('SAM 2 ready on', seg.device, '| torch', torch.__version__, "
        "'| CUDA', torch.cuda.is_available())\n"
    )
    run([python, "-c", code])


def main():
    parser = argparse.ArgumentParser(description="Set up SAM 2 teeth segmentation for FaceSlim.")
    parser.add_argument("--python", default=sys.executable, help="Interpreter to install SAM 2 into.")
    parser.add_argument("--commit", default=SAM2_COMMIT, help="SAM 2 git commit to check out.")
    parser.add_argument("--skip-clone", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    if not args.skip_clone:
        clone_sam2(args.commit)
    if not args.skip_install:
        install_sam2(args.python)
    if not args.skip_verify:
        verify(args.python)
    print("SAM 2 setup complete. Select 'SAM 2 (landmark prompts)' under Teeth Mask, "
          "or pass --teeth-segmenter sam2 on the CLI.")


if __name__ == "__main__":
    main()
