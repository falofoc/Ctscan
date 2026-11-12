"""Digital reference phantom with analytically known ground truth.

Used by the test-suite (and `python -m wiqaya selftest`) to prove that each biomarker
algorithm recovers a known value. It is a geometry/HU test object, not a patient model.
"""
from __future__ import annotations

import numpy as np

from .labels import NAME_TO_ID
from .types import Segmentation, Study

SPACING = (1.0, 1.0, 2.5)
SHAPE = (256, 256, 160)
VERTEBRAE = {"L5": 30, "L4": 60, "L3": 90, "L2": 120, "L1": 150, "T12": 180, "T11": 210,
             "T10": 240, "T9": 270, "T8": 300, "T7": 330}  # body start z (mm); 24 mm body + 6 mm disc


def build(l1_hu: float = 95.0, liver_hu: float = 35.0, spleen_hu: float = 50.0, blood_hu: float = 40.0,
          aneurysm_mm: float = 36.0, noise_sd: float = 8.0, seed: int = 0) -> tuple[Study, Segmentation, dict]:
    sx, sy, sz = SPACING
    x = (np.arange(SHAPE[0]) - SHAPE[0] / 2 + 0.5)[:, None, None] * sx
    y = (np.arange(SHAPE[1]) - SHAPE[1] / 2 + 0.5)[None, :, None] * sy
    z = (np.arange(SHAPE[2]) + 0.5)[None, None, :] * sz
    hu = np.full(SHAPE, -1000.0, dtype=np.float32)
    lab = np.zeros(SHAPE, dtype=np.uint8)
    truth: dict = {}

    def ell(a, b):
        return (x / a) ** 2 + (y / b) ** 2 <= 1.0

    def put(mask, value, name=None):
        mask = np.broadcast_to(mask, SHAPE)
        hu[mask] = value
        if name is not None:
            lab[mask] = NAME_TO_ID[name]
        return mask

    body = ell(115, 90)
    put(body, -100)                                   # fat
    put(body & ~ell(113, 88), 40)                     # 2 mm skin
    wall = ell(105, 80) & ~ell(95, 70)
    put(wall, 50)                                     # abdominal wall muscle ring

    def ellipsoid(c, r):
        return ((x - c[0]) / r[0]) ** 2 + ((y - c[1]) / r[1]) ** 2 + ((z - c[2]) / r[2]) ** 2 <= 1.0

    put(ellipsoid((-48, 5, 190), (38, 40, 45)), liver_hu, "liver")
    put(ellipsoid((62, -15, 195), (20, 22, 32)), spleen_hu, "spleen")
    lung_l = put(ellipsoid((55, 0, 310), (42, 60, 70)), -860, "lung_upper_lobe_left")
    lung_r = put(ellipsoid((-55, 0, 310), (42, 60, 70)), -860, "lung_upper_lobe_right")
    emph = ellipsoid((52, 10, 320), (15, 20, 25))
    put(emph, -985, "lung_upper_lobe_left")
    put(ellipsoid((0, 20, 290), (22, 24, 30)), blood_hu, "heart")
    for side, sgn in (("left", 1), ("right", -1)):
        put(((x - sgn * 32) ** 2 + (y + 45) ** 2 <= 14 ** 2) & (z > 40) & (z < 180), 50, f"iliopsoas_{side}")
        put(((x - sgn * 28) / 22) ** 2 + ((y + 86) / 9) ** 2 <= 1.0, 50, f"autochthon_{side}")

    # Descending/abdominal aorta: a true oblique cylinder (axis tilted in x) so that the
    # axial cross-section is an ellipse and only a correct method returns the real diameter.
    u = np.array([0.25, 0.0, 1.0]) / np.hypot(0.25, 1.0)
    p0 = np.array([-12.0, -22.0, 20.0])
    dx, dy, dz = x - p0[0], y - p0[1], z - p0[2]
    t = dx * u[0] + dy * u[1] + dz * u[2]
    perp2 = dx ** 2 + dy ** 2 + dz ** 2 - t ** 2
    zrange = (z > 20) & (z < 345)
    aorta = (perp2 <= 12.0 ** 2) & zrange
    aneurysm = (perp2 <= (aneurysm_mm / 2) ** 2) & (z > 70) & (z < 115)
    put(aorta | aneurysm, blood_hu, "aorta")
    asc = ((x + 30) ** 2 + (y - 45) ** 2 <= 16.0 ** 2) & (z > 300) & (z < 345)
    put(asc, blood_hu, "aorta")

    for name, z0 in VERTEBRAE.items():
        inbody = (z >= z0) & (z < z0 + 24)
        cyl = (x ** 2 + (y + 55) ** 2 <= 18.0 ** 2) & inbody
        trab = (x ** 2 + (y + 55) ** 2 <= 16.5 ** 2) & (z >= z0 + 2.5) & (z < z0 + 21.5)
        post = (np.abs(x) <= 5) & (y < -73) & (y > -84) & inbody
        put(cyl | post, 600, f"vertebrae_{name}")
        put(trab, l1_hu if name == "L1" else 150.0, f"vertebrae_{name}")

    # Calcified plaques (ground truth for Agatston scoring), added after anatomy.
    cx_at = lambda zz: p0[0] + 0.25 * (zz - p0[2])  # noqa: E731
    plaque_a = (np.abs(x - cx_at(135.0)) <= 2.0) & (y >= -14.5) & (y <= -10.5) & (z >= 130) & (z < 140)
    plaque_a = plaque_a & np.broadcast_to(lab == NAME_TO_ID["aorta"], SHAPE)
    hu[plaque_a] = 350.0
    plaque_h = (np.abs(x - 18) <= 1.5) & (np.abs(y - 22) <= 2.0) & (z >= 285) & (z < 292.5)
    plaque_h = np.broadcast_to(plaque_h, SHAPE) & (lab == NAME_TO_ID["heart"])
    hu[plaque_h] = 450.0

    rng = np.random.default_rng(seed)
    noisy = hu + rng.normal(0, noise_sd, SHAPE).astype(np.float32) if noise_sd else hu
    px = sx * sy
    lung_ids = [NAME_TO_ID["lung_upper_lobe_left"], NAME_TO_ID["lung_upper_lobe_right"]]
    lungs = np.isin(lab, lung_ids)  # final labels: heart/aorta overwrite part of the lung ellipsoids
    del lung_l, lung_r
    slab = sz / 3.0
    zi = int((VERTEBRAE["L3"] + 12) / sz)
    truth.update(
        l1_hu=l1_hu, liver_hu=liver_hu, spleen_hu=spleen_hu,
        aorta_mm=24.0, aneurysm_mm=aneurysm_mm, ascending_mm=32.0,
        laa950=float(100.0 * ((hu < -950) & lungs).sum() / lungs.sum()),
        lung_volume_l=float(lungs.sum() * px * sz / 1e6),
        aorta_agatston=float(plaque_a.sum() * px * 3 * slab),
        heart_agatston=float(plaque_h.sum() * px * 4 * slab),
        muscle_cm2=float(((hu[:, :, zi] == 50)).sum() * px / 100.0),
        liver_ml=float((lab == NAME_TO_ID["liver"]).sum() * px * sz / 1000.0),
    )
    affine = np.diag([sx, sy, sz, 1.0])
    study = Study("ct", np.rint(noisy).astype(np.int16), SPACING, affine,
                  {"modality": "CT", "kvp": 120.0, "slice_thickness": sz, "kernel": "B30f", "source_format": "phantom",
                   "n_slices": SHAPE[2]}, [])
    return study, Segmentation(lab, NAME_TO_ID, source="phantom_truth"), truth


def write_dicom_series(study: Study, out_dir, kvp: float = 120.0, patient_id: str = "PHANTOM-001",
                       skip: tuple[int, ...] = (), intercept: int = -1024) -> list:
    """Write a CT volume as a standard axial DICOM series (for tests and demos)."""
    from pathlib import Path

    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vol = study.pixels
    X, Y, Z = vol.shape
    sx, sy, sz = study.spacing
    study_uid, series_uid = generate_uid(), generate_uid()
    paths = []
    for k in range(Z):
        if k in skip:
            continue
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        path = out / f"IM{(k * 7919) % 100000:05d}.dcm"  # filenames deliberately not in slice order
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
        ds.Modality, ds.PatientID, ds.PatientName = "CT", patient_id, "Phantom^Test"
        ds.PatientSex, ds.PatientAge, ds.PatientSize = "M", "064Y", 1.75
        ds.StudyDate, ds.KVP, ds.SliceThickness, ds.ConvolutionKernel = "20240101", kvp, sz, "B30f"
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [-(X - 1) * sx, -(Y - 1) * sy, k * sz]
        ds.PixelSpacing = [sy, sx]
        ds.InstanceNumber = Z - k
        px = vol[::-1, ::-1, k].T.astype(np.int32) - intercept  # [row, col] = [posterior-ward, left-ward]
        ds.Rows, ds.Columns = px.shape
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.RescaleSlope, ds.RescaleIntercept = 1, intercept
        ds.PixelData = np.clip(px, 0, 65535).astype(np.uint16).tobytes()
        ds.save_as(str(path))
        paths.append(path)
    return paths
