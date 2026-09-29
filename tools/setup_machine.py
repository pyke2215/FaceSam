#!/usr/bin/env python3
"""One-command FaceSlim setup: .venv, requirements, CUDA PyTorch, SAM 2, models, and a health check."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import bootstrap_dev  # noqa: E402
import setup_sam2  # noqa: E402

ROOT = bootstrap_dev.ROOT
TORCH_INDEX = "https://download.pytorch.org/whl/{}"
# Tried in order until a build runs a kernel on the local GPU.
AUTO_CUDA_BUILDS = ("cu128", "cu126", "cu130")

GPU_PROBE = (
    "import torch\n"
    "assert torch.cuda.is_available(), 'CUDA not available'\n"
    "x = torch.randn(128, 128, device='cuda')\n"
    "(x @ x).sum().item()\n"
    "print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))\n"
)


def run(cmd, check=True):
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    return subprocess.run([str(part) for part in cmd], cwd=ROOT, check=check).returncode == 0


def has_nvidia_gpu():
    if not shutil.which("nvidia-smi"):
        return False
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    if out.returncode != 0 or not out.stdout.strip():
        return False
    print("NVIDIA GPU: " + out.stdout.strip().replace("\n", "; "))
    return True


def gpu_probe(python):
    return subprocess.run([str(python), "-c", GPU_PROBE], cwd=ROOT).returncode == 0


def torch_build(python):
    code = "import torch; print(torch.version.cuda or 'cpu')"
    out = subprocess.run([str(python), "-c", code], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def install_torch(python, cuda):
    if cuda == "cpu" or (cuda == "auto" and not has_nvidia_gpu()):
        print("Installing CPU PyTorch (no usable NVIDIA GPU requested/detected).")
        run([python, "-m", "pip", "install", "torch", "torchvision",
             "--index-url", TORCH_INDEX.format("cpu")])
        return "cpu"

    current = torch_build(python)
    if current and current != "cpu" and gpu_probe(python):
        print(f"Existing CUDA PyTorch works (CUDA {current}); keeping it.")
        return current

    builds = AUTO_CUDA_BUILDS if cuda == "auto" else (cuda,)
    for build in builds:
        print(f"Trying PyTorch {build} ...")
        if current:
            run([python, "-m", "pip", "uninstall", "-y", "torch", "torchvision"], check=False)
        if not run([python, "-m", "pip", "install", "torch", "torchvision",
                    "--index-url", TORCH_INDEX.format(build)], check=False):
            current = torch_build(python)
            continue
        current = torch_build(python)
        if gpu_probe(python):
            print(f"PyTorch {build} runs on this GPU.")
            return build
        print(f"PyTorch {build} installed but cannot run kernels on this GPU.")
    raise SystemExit("No CUDA PyTorch build worked. Update the NVIDIA driver, or rerun with --cuda cpu.")


def download_models(python):
    code = (
        "from faceslim.models import ensure_model, ensure_parsing_model, ensure_sam2_model\n"
        "ok = ensure_model() and ensure_parsing_model()\n"
        "ok = ensure_sam2_model() and ok\n"
        "raise SystemExit(0 if ok else 1)\n"
    )
    run([python, "-c", code])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", default="auto",
                        help="PyTorch build: auto (default), cu128, cu126, cu130, or cpu.")
    parser.add_argument("--skip-sam2", action="store_true", help="Do not install SAM 2.")
    parser.add_argument("--skip-models", action="store_true", help="Do not pre-download models.")
    args = parser.parse_args()

    if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
        raise SystemExit(
            f"Python {sys.version.split()[0]} is not supported for the pinned requirements. "
            "Install Python 3.11 and run: py -3.11 tools\\setup_machine.py")

    python = bootstrap_dev.ensure_venv()
    run([python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    bootstrap_dev.install_requirements(python)
    build = install_torch(python, args.cuda)
    if not args.skip_sam2:
        setup_sam2.clone_sam2()
        setup_sam2.install_sam2(python)
    if not args.skip_models:
        download_models(python)
    run([python, TOOLS / "doctor.py"] + (["--no-benchmark"] if args.skip_sam2 else []), check=False)
    print(f"\nSetup finished (PyTorch: {build}). Start FaceSlim with run.bat "
          f"or: {python} FaceSlim_v1.py")


if __name__ == "__main__":
    main()
