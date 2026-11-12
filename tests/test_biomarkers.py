import numpy as np
import pytest

from wiqaya import biomarkers as B, phantom
from wiqaya.selftest import CHECKS, measure


@pytest.fixture(scope="module")
def measured(ph):
    study, seg, truth = ph
    return measure(study, seg), truth


@pytest.mark.parametrize("key,tkey,tol", CHECKS)
def test_phantom_ground_truth(measured, key, tkey, tol):
    got, truth = measured
    assert key in got, f"{key} was not produced"
    assert got[key].value == pytest.approx(truth[tkey], abs=tol)


def test_agatston_analytic():
    hu = np.full((40, 40, 4), 40.0, dtype=np.float32)
    roi = np.ones(hu.shape, dtype=bool)
    hu[10:14, 10:14, 1] = 250      # 16 px × 0.25 mm² = 4 mm², weight 2
    hu[20:22, 20:23, 2] = 450      # 6 px = 1.5 mm², weight 4
    hu[30, 30, 0] = 900            # single pixel: below minimum lesion size
    res = B.agatston(hu, roi, (0.5, 0.5, 3.0))
    assert res["score"] == pytest.approx(4 * 2 + 1.5 * 4)
    assert res["n_lesions"] == 2
    assert res["volume_mm3"] == pytest.approx((4 + 1.5) * 3.0)


def test_agatston_respects_roi_and_metal():
    hu = np.full((30, 30, 3), 40.0, dtype=np.float32)
    roi = np.zeros(hu.shape, dtype=bool)
    roi[:15] = True
    hu[20:24, 5:9, 1] = 500        # outside ROI
    hu[5:9, 5:9, 1] = 3000         # metal inside ROI
    res = B.agatston(hu, roi, (1.0, 1.0, 3.0))
    assert res["score"] == 0 and res["metal_excluded"] == 1


def test_agatston_thin_slices_are_averaged_to_3mm():
    # A 3 mm-thick plaque on 1 mm slices must score like the same plaque on one 3 mm slice.
    thin = np.full((30, 30, 9), 40.0, dtype=np.float32)
    thin[10:14, 10:14, 3:6] = 320
    thick = np.full((30, 30, 3), 40.0, dtype=np.float32)
    thick[10:14, 10:14, 1] = 320
    roi_thin, roi_thick = np.ones(thin.shape, bool), np.ones(thick.shape, bool)
    a = B.agatston(thin, roi_thin, (1.0, 1.0, 1.0))["score"]
    b = B.agatston(thick, roi_thick, (1.0, 1.0, 3.0))["score"]
    assert a == pytest.approx(b) == pytest.approx(16 * 3)


@pytest.mark.parametrize("a,b", [(20, 20), (30, 15), (40, 12.5)])
def test_inscribed_diameter_is_minor_axis_of_ellipse(a, b):
    yy, xx = np.mgrid[-60:61, -60:61]
    rng = np.random.default_rng(0)
    for _ in range(10):  # any sub-pixel position of the centre
        ox, oy = rng.random(2)
        ell = ((xx - ox) / a) ** 2 + ((yy - oy) / b) ** 2 <= 1.0
        assert B.inscribed_diameter_mm(ell, 0.8) == pytest.approx(2 * b * 0.8, abs=0.5)  # ≤0.6 px
        equivalent_area_d = 2 * np.sqrt(ell.sum() / np.pi) * 0.8
        if a > b:
            assert equivalent_area_d > 2 * b * 0.8 + 3  # the naive estimator would overcall


def test_contrast_detection_blocks_hu_threshold_biomarkers():
    study, seg, _ = phantom.build(blood_hu=180.0)
    c = B.contrast_state(study, seg)
    assert c["state"] == "enhanced"
    ms = {m.key: m for m in B.liver_spleen(study, seg, c)}
    assert ms["liver.attenuation_hu"].reliability == "invalid"
    assert B.calcium(study, seg, c, 76.5)[0] == []


def test_calcium_abstains_when_noise_would_fake_lesions():
    study, seg, _ = phantom.build(noise_sd=70.0)
    c = B.contrast_state(study, seg)
    ms, info = B.calcium(study, seg, c, 76.5)
    assert ms == [] and info["abstained"] == "high_noise"


def test_bone_falls_back_to_neighbour_when_l1_missing(ph):
    study, seg, _ = ph
    labels = seg.labels.copy()
    labels[labels == seg.names["vertebrae_L1"]] = 0
    from wiqaya.types import Segmentation

    ms, info = B.bone_density(study, Segmentation(labels, seg.names))
    assert info["chosen"] == "T12" and "level_substituted" in ms[0].notes


def test_truncated_vertebra_is_not_measured(ph):
    study, seg, _ = ph
    from wiqaya.types import Segmentation, Study

    z0 = 60  # cut through L1 (z index 60..69)
    sub = Study("ct", study.pixels[:, :, z0 + 4:], study.spacing, study.affine, dict(study.meta))
    ms, info = B.bone_density(sub, Segmentation(seg.labels[:, :, z0 + 4:], seg.names))
    assert info["levels"]["L1"]["status"] == "truncated" and info["chosen"] != "L1"


def test_results_are_deterministic(ph):
    study, seg, _ = ph
    a, b = measure(study, seg), measure(study, seg)
    assert {k: v.value for k, v in a.items()} == {k: v.value for k, v in b.items()}
