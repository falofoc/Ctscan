import pytest

from wiqaya import rules
from wiqaya.types import Measurement

UNENH = {"state": "unenhanced"}


def ids(ms, meta=None, contrast=UNENH, **kw):
    f, cov = rules.evaluate({m.key: m for m in ms}, meta or {}, contrast, **kw)
    return {x.id: x for x in f}, {c["domain"]: c for c in cov}


@pytest.mark.parametrize("hu,expected", [(60, "bone.low"), (110, "bone.low"), (110.1, "bone.indeterminate"),
                                         (160, "bone.indeterminate"), (160.1, "bone.normal"), (230, "bone.normal")])
def test_bone_thresholds(hu, expected):
    f, _ = ids([Measurement("bone.trabecular_hu", hu, "HU", extra={"level": "L1"})])
    assert expected in f


@pytest.mark.parametrize("hu,expected", [(25, "liver.steatosis_modsev"), (40, "liver.steatosis_modsev"),
                                         (44, "liver.steatosis"), (48, "liver.normal"), (62, "liver.normal")])
def test_liver_thresholds(hu, expected):
    f, _ = ids([Measurement("liver.attenuation_hu", hu, "HU")])
    assert expected in f


def test_liver_not_assessed_with_contrast_is_stated_not_called_normal():
    m = Measurement("liver.attenuation_hu", 110, "HU", "invalid", ["contrast_enhanced"])
    f, cov = ids([m], contrast={"state": "enhanced"})
    assert not any(k.startswith("liver.") for k in f)
    assert cov["liver"]["assessed"] is False and "تباين" in cov["liver"]["reason_ar"]


@pytest.mark.parametrize("d,fid,sev", [(22, "aorta.abdominal_normal", 0), (27, "aorta.ectasia", 1), (30, "aorta.aaa", 2),
                                       (45, "aorta.aaa", 2), (52, "aorta.aaa", 3), (60, "aorta.aaa", 3)])
def test_abdominal_aorta(d, fid, sev):
    f, _ = ids([Measurement("aorta.abdominal_max_diameter_mm", d, "mm", "moderate")])
    assert f[fid].severity == sev
    assert any("يدوي" in c for c in f[fid].caveats_ar)  # always asks for manual confirmation


@pytest.mark.parametrize("score,fid,sev", [(0, "cardiac.calcium_none", 0), (40, "cardiac.calcium", 1),
                                           (100, "cardiac.calcium", 2), (900, "cardiac.calcium", 2)])
def test_cardiac_calcium(score, fid, sev):
    f, _ = ids([Measurement("heart.calcium_agatston", score, "Agatston", "moderate", ["non_gated"])])
    assert f[fid].severity == sev


def test_low_reliability_never_raises_alarm_level():
    f, _ = ids([Measurement("aorta.abdominal_max_diameter_mm", 60, "mm", "low")])
    assert f["aorta.aaa"].severity == 1


def test_nothing_in_view_yields_no_findings_and_full_not_assessed_list():
    f, cov = ids([])
    assert f == {} and all(not c["assessed"] and c["reason_ar"] for c in cov.values())


def test_sarcopenia_needs_height_and_sex():
    area = Measurement("body.l3_muscle_area_cm2", 120, "cm²", "moderate")
    f, _ = ids([area])
    assert "body.composition" in f
    smi = Measurement("body.l3_smi_cm2_m2", 39.0, "cm²/m²", "moderate")
    f, _ = ids([area, smi], meta={"sex": "M"})
    assert f["body.sarcopenia"].severity == 1
    f, _ = ids([area, smi], meta={"sex": "F"})
    assert "body.muscle_normal" in f


def test_every_finding_cites_known_evidence():
    ms = [Measurement("bone.trabecular_hu", 90, "HU", extra={"level": "L1"}), Measurement("liver.attenuation_hu", 30, "HU"),
          Measurement("aorta.abdominal_max_diameter_mm", 41, "mm", "moderate"), Measurement("aorta.ascending_max_diameter_mm", 46, "mm", "moderate"),
          Measurement("heart.calcium_agatston", 250, "Agatston", "moderate"), Measurement("lung.laa950_percent", 12, "%"),
          Measurement("aorta.abdominal_calcium_agatston", 800, "Agatston", "moderate", extra={"volume_mm3": 600})]
    f, _ = ids(ms)
    assert len(f) == 7
    for x in f.values():
        assert x.evidence and all(e in rules.EVIDENCE for e in x.evidence)
        assert x.severity == 0 or x.recommendation_ar
