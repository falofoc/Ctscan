"""Anatomical segmentation backends.

The default backend is TotalSegmentator (Wasserthal et al., Radiol Artif Intell 2023;
nnU-Net), an openly licensed, externally validated CT model. Its weights are used
unmodified. Inference runs in a child process: a crash or out-of-memory in the model
cannot take the service down, and all model memory is returned when it exits.
The biomarker code only needs a `Segmentation`, so another backend can be swapped in.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import nibabel as nib
import numpy as np

from .labels import NAME_TO_ID
from .types import Segmentation, Study

_LOCK = threading.Lock()  # one inference at a time: RAM is finite


class SegmentationError(RuntimeError):
    pass


def class_names() -> dict[str, int]:
    return dict(NAME_TO_ID)


def available() -> bool:
    import importlib.util

    return importlib.util.find_spec("totalsegmentator") is not None


def default_mode() -> tuple[bool, str]:
    """(fast, device). CPU defaults to the 3 mm model; GPU to the full 1.5 mm model."""
    device = os.environ.get("WIQAYA_DEVICE")
    if device is None:
        import torch

        device = "gpu" if torch.cuda.is_available() else "cpu"
    env = os.environ.get("WIQAYA_FAST")
    fast = (env == "1") if env is not None else (device == "cpu")
    return fast, device


def segment_ct(study: Study, fast: bool | None = None, device: str | None = None, timeout_s: int = 3600) -> Segmentation:
    d_fast, d_device = default_mode()
    fast = d_fast if fast is None else fast
    device = device or d_device
    with tempfile.TemporaryDirectory(prefix="wiqaya_seg_") as tmp, _LOCK:
        src, dst = Path(tmp) / "ct.nii.gz", Path(tmp) / "seg.nii.gz"
        nib.save(nib.Nifti1Image(study.pixels.astype(np.int16), study.affine), str(src))
        cmd = [sys.executable, "-m", "wiqaya.segment", str(src), str(dst), "1" if fast else "0", device]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if proc.returncode != 0 or not dst.exists():
            raise SegmentationError(f"segmentation failed: {proc.stderr[-800:]}")
        labels = np.asarray(nib.load(str(dst)).dataobj).astype(np.uint8)
    if labels.shape != study.pixels.shape:
        raise SegmentationError("segmentation grid does not match the input volume")
    return Segmentation(labels, class_names(), source="totalsegmentator" + ("_fast" if fast else ""))


def from_label_volume(labels: np.ndarray, source: str = "external") -> Segmentation:
    return Segmentation(labels.astype(np.uint8), class_names(), source=source)


def _disable_telemetry() -> None:
    """TotalSegmentator posts usage statistics to its authors by default.

    Nothing may leave a hospital network implicitly, so the opt-out is enforced in its
    config file before the library is imported, on every run.
    """
    import json

    home = Path(os.environ.get("TOTALSEG_HOME_DIR") or (Path("/tmp") if str(Path.home()) == "/" else Path.home()) / ".totalsegmentator")
    home.mkdir(parents=True, exist_ok=True)
    cfg_file = home / "config.json"
    cfg = {"totalseg_id": "local", "prediction_counter": 0}
    if cfg_file.exists():
        try:
            cfg = json.loads(cfg_file.read_text())
        except ValueError:
            pass
    if cfg.get("send_usage_stats") is not False:
        cfg["send_usage_stats"] = False
        cfg_file.write_text(json.dumps(cfg, indent=4))


def _worker(src: str, dst: str, fast: str, device: str) -> None:
    _disable_telemetry()
    from totalsegmentator.python_api import totalsegmentator

    out = totalsegmentator(nib.load(src), None, ml=True, fast=fast == "1", device=device, quiet=True, skip_saving=True)
    nib.save(out, dst)


if __name__ == "__main__":
    _worker(*sys.argv[1:5])
