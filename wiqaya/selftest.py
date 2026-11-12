"""Installation self-check: every biomarker must recover the phantom's known ground truth."""
from __future__ import annotations

from . import biomarkers as B, phantom

# (measurement key, truth key, absolute tolerance)
CHECKS = [
    ("bone.trabecular_hu", "l1_hu", 3.0), ("liver.attenuation_hu", "liver_hu", 2.0),
    ("spleen.attenuation_hu", "spleen_hu", 2.0), ("aorta.abdominal_max_diameter_mm", "aneurysm_mm", 1.0),
    ("aorta.descending_max_diameter_mm", "aorta_mm", 1.0), ("aorta.ascending_max_diameter_mm", "ascending_mm", 1.0),
    ("aorta.abdominal_calcium_agatston", "aorta_agatston", 1.0), ("heart.calcium_agatston", "heart_agatston", 1.0),
    ("lung.laa950_percent", "laa950", 0.15), ("lung.volume_l", "lung_volume_l", 0.01),
    ("body.l3_muscle_area_cm2", "muscle_cm2", 2.5), ("liver.volume_ml", "liver_ml", 1.0),
]


def measure(study, seg) -> dict:
    c = B.contrast_state(study, seg)
    ms, bone = B.bone_density(study, seg)
    ms += B.liver_spleen(study, seg, c)
    am, geo = B.aorta_geometry(study, seg)
    ms += am
    ms += B.calcium(study, seg, c, geo.get("z_split"))[0]
    ms += B.lungs(study, seg)
    l3 = bone["levels"].get("L3", {})
    ms += B.body_composition(study, seg, c, l3.get("centre"))[0]
    return {m.key: m for m in ms}


def run() -> int:
    study, seg, truth = phantom.build()
    got = measure(study, seg)
    failed = 0
    for key, tkey, tol in CHECKS:
        v = got[key].value if key in got else None
        ok = v is not None and abs(v - truth[tkey]) <= tol
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {key:38s} measured={v if v is None else round(v, 2)}  truth={round(truth[tkey], 2)}  tol=±{tol}")
    print("selftest:", "ALL PASS" if not failed else f"{failed} FAILED")
    return 1 if failed else 0
