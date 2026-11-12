import random
import zipfile

import nibabel as nib
import numpy as np
import pytest

from wiqaya import io, phantom
from wiqaya.types import Study


@pytest.fixture(scope="module")
def small():
    study, _, _ = phantom.build(noise_sd=0)
    return Study("ct", study.pixels[::2, ::2, 40:80].copy(), (2.0, 2.0, 2.5), np.diag([2.0, 2.0, 2.5, 1.0]), dict(study.meta))


def test_dicom_roundtrip_is_exact_despite_shuffled_files(small, tmp_path):
    paths = phantom.write_dicom_series(small, tmp_path / "s")
    assert [p.name for p in paths] != sorted(p.name for p in paths)  # order really is scrambled
    got = io.load_any(tmp_path / "s")
    assert got.kind == "ct" and got.warnings == []
    assert got.spacing == pytest.approx(small.spacing)
    assert np.array_equal(got.pixels, small.pixels)  # geometry + HU rescale + orientation all correct
    assert got.meta["kvp"] == 120 and got.meta["sex"] == "M" and got.meta["age"] == 64 and got.meta["height_m"] == 1.75


def test_patient_identifiers_are_pseudonymised(small, tmp_path):
    phantom.write_dicom_series(small, tmp_path / "s", patient_id="MRN-123456")
    got = io.load_any(tmp_path / "s")
    assert "MRN-123456" not in repr(got.meta) and "Phantom" not in repr(got.meta)
    assert got.meta["patient_pseudo_id"] == io.pseudonym("MRN-123456")


def test_missing_slice_is_flagged(small, tmp_path):
    phantom.write_dicom_series(small, tmp_path / "s", skip=(17,))
    assert "nonuniform_slice_spacing" in io.load_any(tmp_path / "s").warnings


def test_zip_of_dicom(small, tmp_path):
    paths = phantom.write_dicom_series(small, tmp_path / "s")
    random.Random(0).shuffle(paths)
    with zipfile.ZipFile(tmp_path / "study.zip", "w") as z:
        for p in paths:
            z.write(p, f"DICOM/{p.name}")
    assert np.array_equal(io.load_any(tmp_path / "study.zip").pixels, small.pixels)


def test_zip_path_traversal_is_refused(tmp_path):
    with zipfile.ZipFile(tmp_path / "evil.zip", "w") as z:
        z.writestr("../../escape.txt", "x")
    with pytest.raises(io.IngestError):
        io.load_any(tmp_path / "evil.zip")
    assert not (tmp_path.parent / "escape.txt").exists()


@pytest.mark.parametrize("flip", [(False, False, False), (True, False, False), (False, True, True)])
def test_nifti_any_orientation_becomes_canonical(small, tmp_path, flip):
    data, aff = small.pixels.astype(np.int16), small.affine.copy()
    for ax, f in enumerate(flip):
        if f:
            data = np.flip(data, ax)
            aff[:3, 3] += aff[:3, ax] * (data.shape[ax] - 1)
            aff[:3, ax] *= -1
    nib.save(nib.Nifti1Image(data, aff), str(tmp_path / "ct.nii.gz"))
    got = io.load_any(tmp_path / "ct.nii.gz")
    assert got.kind == "ct" and np.array_equal(got.pixels, small.pixels)
    assert "no_acquisition_metadata" in got.warnings


def test_garbage_is_rejected_cleanly(tmp_path):
    (tmp_path / "x.dcm").write_bytes(b"not a dicom file at all")
    with pytest.raises(io.IngestError):
        io.load_any(tmp_path / "x.dcm")
    with pytest.raises(io.IngestError):
        io.load_any(tmp_path / "missing")


def test_plain_image_is_never_treated_as_ct(tmp_path):
    from PIL import Image

    Image.fromarray((np.random.default_rng(0).random((64, 64)) * 255).astype(np.uint8)).save(tmp_path / "a.png")
    got = io.load_any(tmp_path / "a.png")
    assert got.kind == "image2d" and "uncalibrated_image" in got.warnings
