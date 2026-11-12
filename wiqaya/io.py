"""Modality-agnostic ingestion: DICOM (folder / zip / single file), NIfTI, plain images.

Everything volumetric is returned in canonical RAS+ orientation so that downstream
geometry (anterior/posterior, superior/inferior) never depends on the scanner.
Problems found while reading are recorded as warning codes rather than silently fixed.
"""
from __future__ import annotations

import hashlib
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

from .types import Study

NIFTI_SUFFIXES = (".nii", ".nii.gz")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
VOLUME_MODALITIES = {"CT", "MR", "PT"}
PROJECTION_MODALITIES = {"CR", "DX", "MG", "XA", "RF", "US", "NM", "OT", "SC", "IO", "PX"}


class IngestError(Exception):
    """The input could not be turned into an analysable study."""


def pseudonym(value: str | None) -> str | None:
    """One-way pseudonymous identifier; raw patient identifiers are never stored."""
    if not value:
        return None
    return hashlib.sha256(str(value).encode()).hexdigest()[:12]


def load_any(path: str | Path) -> Study:
    path = Path(path)
    if not path.exists():
        raise IngestError(f"path not found: {path}")
    if path.is_dir():
        return _load_dicom_files([p for p in sorted(path.rglob("*")) if p.is_file()])
    name = path.name.lower()
    if name.endswith(NIFTI_SUFFIXES):
        return load_nifti(path)
    if name.endswith(IMAGE_SUFFIXES):
        return _load_plain_image(path)
    if zipfile.is_zipfile(path):
        with tempfile.TemporaryDirectory() as tmp:
            _safe_extract(path, Path(tmp))
            files = [p for p in sorted(Path(tmp).rglob("*")) if p.is_file()]
            nifti = [p for p in files if p.name.lower().endswith(NIFTI_SUFFIXES)]
            if nifti and not any(_looks_like_dicom(p) for p in files[:50]):
                return load_nifti(nifti[0])
            return _load_dicom_files(files)
    return _load_dicom_files([path])


def _safe_extract(zip_path: Path, dest: Path, max_bytes: int = 8 * 1024**3) -> None:
    """Extract refusing path traversal and decompression bombs."""
    total = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            target = (dest / info.filename).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise IngestError("archive contains an unsafe path")
            total += info.file_size
            if total > max_bytes:
                raise IngestError("archive is too large when extracted")
        z.extractall(dest)


def _looks_like_dicom(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            head = f.read(132)
        return head[128:132] == b"DICM"
    except OSError:
        return False


# ----------------------------------------------------------------------------- NIfTI


def load_nifti(path: str | Path, modality: str | None = None) -> Study:
    try:
        img = nib.load(str(path))
    except Exception as e:  # nibabel raises several unrelated types
        raise IngestError(f"cannot read NIfTI: {e}") from e
    if len(img.shape) == 4 and img.shape[3] == 1:
        img = nib.Nifti1Image(np.asanyarray(img.dataobj)[..., 0], img.affine)
    if len(img.shape) != 3:
        raise IngestError(f"expected a 3D volume, got shape {img.shape}")
    img = nib.as_closest_canonical(img)
    data = np.asarray(img.get_fdata(dtype=np.float32))
    spacing = tuple(float(s) for s in nib.affines.voxel_sizes(img.affine))
    warnings: list[str] = []
    # NIfTI carries no modality tag: infer CT from a Hounsfield-like range.
    lo, hi = np.percentile(data[::4, ::4, ::2], [0.5, 99.9])
    if modality is None:
        modality = "CT" if (lo < -500 and hi > 150) else "UNKNOWN"
        warnings.append("modality_inferred")
    kind = {"CT": "ct", "MR": "mr"}.get(modality, "other")
    if kind == "ct":
        data = np.clip(np.rint(data), -32768, 32767).astype(np.int16)
    meta = {"modality": modality, "source_format": "nifti", "n_slices": int(data.shape[2])}
    warnings.append("no_acquisition_metadata")
    return Study(kind, data, spacing, np.array(img.affine), meta, warnings)


# ----------------------------------------------------------------------------- DICOM


def _load_dicom_files(files: list[Path]) -> Study:
    import pydicom

    headers = []
    for p in files:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=False)
        except Exception:
            continue  # non-DICOM files (DICOMDIR text, thumbnails...) are simply skipped
        if "SOPClassUID" in ds and "Rows" in ds:
            headers.append((p, ds))
    if not headers:
        raise IngestError("no readable DICOM images were found")

    series = defaultdict(list)
    for p, ds in headers:
        series[str(ds.get("SeriesInstanceUID", "none"))].append((p, ds))

    best, best_score = None, -1.0
    for uid, items in series.items():
        score = _series_score(items)
        if score > best_score:
            best, best_score = items, score
    assert best is not None
    ds0 = best[0][1]
    modality = str(ds0.get("Modality", "OT")).upper()
    n_frames = int(ds0.get("NumberOfFrames", 1) or 1)

    if modality in VOLUME_MODALITIES and (len(best) >= 8 or n_frames >= 8):
        study = _load_enhanced_volume(best[0][0]) if n_frames >= 8 else _load_volume_series(best)
    else:
        study = _load_projection(best[0][0])
    study.meta["n_series_in_upload"] = len(series)
    return study


def _series_score(items) -> float:
    ds = items[0][1]
    image_type = [str(v).upper() for v in ds.get("ImageType", [])]
    n = len(items) * int(ds.get("NumberOfFrames", 1) or 1)
    score = float(n)
    if any(t in image_type for t in ("LOCALIZER", "SCOUT", "DOSE_INFO", "SCREEN SAVE")):
        score *= 0.01
    if any(t in image_type for t in ("MIP", "MINIP", "REFORMATTED", "MPR", "VRT")):
        score *= 0.2
    iop = ds.get("ImageOrientationPatient")
    if iop is not None and len(iop) == 6:
        normal = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
        if abs(normal[2]) < 0.8:  # not axial → reformats are a poor basis for measurement
            score *= 0.3
    try:
        if float(ds.get("SliceThickness", 1.0)) > 5.0:
            score *= 0.5
    except (TypeError, ValueError):
        pass
    return score


def _common_meta(ds, modality: str) -> dict:
    def num(tag):
        try:
            v = ds.get(tag)
            return float(v) if v not in (None, "") else None
        except (TypeError, ValueError):
            return None

    age = None
    raw_age = str(ds.get("PatientAge", "") or "")
    if len(raw_age) == 4 and raw_age[:3].isdigit() and raw_age[3] == "Y":
        age = float(raw_age[:3])
    sex = str(ds.get("PatientSex", "") or "").upper()
    height = num("PatientSize")  # metres
    return {
        "modality": modality,
        "source_format": "dicom",
        "patient_pseudo_id": pseudonym(ds.get("PatientID")),
        "study_uid_hash": pseudonym(ds.get("StudyInstanceUID")),
        "study_date": str(ds.get("StudyDate", "") or "") or None,
        "study_description": str(ds.get("StudyDescription", "") or "") or None,
        "series_description": str(ds.get("SeriesDescription", "") or "") or None,
        "body_part": str(ds.get("BodyPartExamined", "") or "") or None,
        "manufacturer": str(ds.get("Manufacturer", "") or "") or None,
        "model": str(ds.get("ManufacturerModelName", "") or "") or None,
        "kvp": num("KVP"),
        "slice_thickness": num("SliceThickness"),
        "kernel": _kernel(ds),
        "contrast_agent_tag": str(ds.get("ContrastBolusAgent", "") or "") or None,
        "view_position": str(ds.get("ViewPosition", "") or "") or None,
        "age": age,
        "sex": sex if sex in ("M", "F") else None,
        "height_m": height if height and 1.0 < height < 2.3 else None,
        "weight_kg": num("PatientWeight"),
    }


def _kernel(ds) -> str | None:
    k = ds.get("ConvolutionKernel")
    if k is None:
        return None
    if isinstance(k, (list, tuple)) or hasattr(k, "__iter__") and not isinstance(k, str):
        k = "/".join(str(x) for x in k)
    return str(k) or None


def _rescale(ds, arr: np.ndarray) -> np.ndarray:
    slope = float(ds.get("RescaleSlope", 1) or 1)
    intercept = float(ds.get("RescaleIntercept", 0) or 0)
    out = arr.astype(np.float32) * slope + intercept
    return out


def _load_volume_series(items) -> Study:
    import pydicom

    warnings: list[str] = []
    ds0 = items[0][1]
    modality = str(ds0.get("Modality")).upper()
    usable = [(p, ds) for p, ds in items
              if ds.get("ImagePositionPatient") is not None and ds.get("ImageOrientationPatient") is not None]
    if len(usable) < 8:
        raise IngestError("series lacks slice position information")
    iop = np.array(usable[0][1].ImageOrientationPatient, dtype=float)
    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)
    shape0 = (int(usable[0][1].Rows), int(usable[0][1].Columns))

    keyed = {}
    for p, ds in usable:
        if (int(ds.Rows), int(ds.Columns)) != shape0:
            continue
        if not np.allclose(np.array(ds.ImageOrientationPatient, float), iop, atol=1e-3):
            warnings.append("mixed_orientation_slices_dropped")
            continue
        pos = round(float(np.dot(np.array(ds.ImagePositionPatient, float), normal)), 3)
        if pos in keyed:
            warnings.append("duplicate_slice_positions")
            continue
        keyed[pos] = (p, ds)
    positions = sorted(keyed)
    if len(positions) < 8:
        raise IngestError("too few consistent slices in series")

    steps = np.diff(positions)
    step = float(np.median(steps))
    if step <= 0:
        raise IngestError("invalid slice spacing")
    if np.any(np.abs(steps - step) > max(0.05 * step, 0.05)):
        # Missing or irregular slices change every volume and distance we compute.
        warnings.append("nonuniform_slice_spacing")

    slices = []
    for pos in positions:
        p, hdr = keyed[pos]
        ds = pydicom.dcmread(str(p))
        try:
            px = ds.pixel_array
        except Exception as e:
            raise IngestError(f"cannot decode pixel data ({ds.file_meta.TransferSyntaxUID.name}): {e}") from e
        slices.append(_rescale(ds, px).T)  # → [column(i), row(j)]
    vol = np.stack(slices, axis=-1)

    ps = [float(v) for v in ds0.PixelSpacing]  # [between rows, between columns]
    first_ipp = np.array(keyed[positions[0]][1].ImagePositionPatient, dtype=float)
    lps = np.eye(4)
    lps[:3, 0] = row_dir * ps[1]
    lps[:3, 1] = col_dir * ps[0]
    lps[:3, 2] = normal * step
    lps[:3, 3] = first_ipp
    ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps

    if abs(normal[2]) < 0.995:
        warnings.append("gantry_tilt_or_oblique")
    img = nib.as_closest_canonical(nib.Nifti1Image(vol, ras))
    data = np.asarray(img.dataobj)
    if modality == "CT":
        data = np.clip(np.rint(data), -32768, 32767).astype(np.int16)
    else:
        data = data.astype(np.float32)
    meta = _common_meta(ds0, modality)
    meta["n_slices"] = int(data.shape[2])
    thick = meta.get("slice_thickness")
    if thick and step > thick * 1.05:
        warnings.append("slice_gap")
    spacing = tuple(float(s) for s in nib.affines.voxel_sizes(img.affine))
    return Study({"CT": "ct", "MR": "mr"}.get(modality, "other"), data, spacing,
                 np.array(img.affine), meta, sorted(set(warnings)))


def _load_enhanced_volume(path: Path) -> Study:
    """Enhanced (multi-frame) CT/MR object: geometry lives in per-frame functional groups."""
    import pydicom

    ds = pydicom.dcmread(str(path))
    modality = str(ds.get("Modality")).upper()
    try:
        shared = ds.SharedFunctionalGroupsSequence[0]
        frames = ds.PerFrameFunctionalGroupsSequence

        def group(i, name):
            fg = frames[i]
            return (fg if name in fg else shared)[name][0]

        iop = np.array(group(0, "PlaneOrientationSequence").ImageOrientationPatient, float)
        ps = [float(v) for v in group(0, "PixelMeasuresSequence").PixelSpacing]
        ipps = [np.array(group(i, "PlanePositionSequence").ImagePositionPatient, float)
                for i in range(len(frames))]
    except Exception as e:
        raise IngestError(f"unsupported multi-frame DICOM layout: {e}") from e
    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)
    px = ds.pixel_array.astype(np.float32)
    order = np.argsort([float(np.dot(p, normal)) for p in ipps])
    pos = np.array([float(np.dot(ipps[i], normal)) for i in order])
    steps = np.diff(pos)
    step = float(np.median(steps))
    warnings = []
    if step <= 0:
        raise IngestError("multi-frame object is not a single spatial stack")
    if np.any(np.abs(steps - step) > max(0.05 * step, 0.05)):
        warnings.append("nonuniform_slice_spacing")
    out = []
    for i in order:
        try:
            tr = group(int(i), "PixelValueTransformationSequence")
            slope, inter = float(tr.RescaleSlope), float(tr.RescaleIntercept)
        except Exception:
            slope, inter = float(ds.get("RescaleSlope", 1) or 1), float(ds.get("RescaleIntercept", 0) or 0)
        out.append((px[int(i)] * slope + inter).T)
    vol = np.stack(out, axis=-1)
    lps = np.eye(4)
    lps[:3, 0], lps[:3, 1], lps[:3, 2], lps[:3, 3] = row_dir * ps[1], col_dir * ps[0], normal * step, ipps[int(order[0])]
    img = nib.as_closest_canonical(nib.Nifti1Image(vol, np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps))
    data = np.asarray(img.dataobj)
    data = np.clip(np.rint(data), -32768, 32767).astype(np.int16) if modality == "CT" else data.astype(np.float32)
    meta = _common_meta(ds, modality)
    meta["n_slices"] = int(data.shape[2])
    spacing = tuple(float(s) for s in nib.affines.voxel_sizes(img.affine))
    return Study({"CT": "ct", "MR": "mr"}.get(modality, "other"), data, spacing, np.array(img.affine), meta, warnings)


def _load_projection(path: Path) -> Study:
    import pydicom
    from pydicom.pixels import apply_voi_lut

    ds = pydicom.dcmread(str(path))
    modality = str(ds.get("Modality", "OT")).upper()
    try:
        arr = ds.pixel_array
    except Exception as e:
        raise IngestError(f"cannot decode pixel data: {e}") from e
    warnings: list[str] = []
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        arr = arr[arr.shape[0] // 2]  # multi-frame (cine/US): take the middle frame
        warnings.append("multiframe_middle_frame_used")
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    else:
        try:
            arr = apply_voi_lut(arr, ds)
        except Exception:
            pass
    arr = arr.astype(np.float32)
    if str(ds.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr  # make bone/dense structures bright, as models expect
    ps = ds.get("ImagerPixelSpacing") or ds.get("PixelSpacing")
    spacing = tuple(float(v) for v in ps) if ps else (float("nan"), float("nan"))
    meta = _common_meta(ds, modality)
    kind = "xray" if modality in ("CR", "DX") else "image2d"
    return Study(kind, arr, spacing, None, meta, warnings)


def _load_plain_image(path: Path) -> Study:
    from PIL import Image

    try:
        im = Image.open(path)
        arr = np.asarray(im.convert("F"), dtype=np.float32)
    except Exception as e:
        raise IngestError(f"cannot read image: {e}") from e
    meta = {"modality": "UNKNOWN", "source_format": path.suffix.lstrip(".").lower()}
    # A consumer image has no calibration, no view tag and possibly burnt-in annotations.
    return Study("image2d", arr, (float("nan"), float("nan")), None, meta,
                 ["no_acquisition_metadata", "uncalibrated_image"])
