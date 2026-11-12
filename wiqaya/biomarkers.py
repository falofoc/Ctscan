"""Quantitative CT biomarkers computed deterministically from Hounsfield units + anatomy masks.

No learned component lives here: given the same volume and segmentation the output is
bit-for-bit reproducible, and every number can be traced to a region shown in the report.
Arrays are [x, y, z] in RAS+ orientation (+y anterior, +z superior).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .types import Measurement, Segmentation, Study

VERTEBRA_LEVELS = ["T11", "T12", "L1", "L2", "L3", "L4"]
LEVEL_PREFERENCE = ["L1", "T12", "L2", "L3", "T11", "L4"]  # Pickhardt 2013: similar performance at all levels
LUNG_LOBES = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
              "lung_middle_lobe_right", "lung_lower_lobe_right"]
TS_MUSCLES = ["autochthon_left", "autochthon_right", "iliopsoas_left", "iliopsoas_right"]
VISCERA = ["spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach", "pancreas",
           "adrenal_gland_right", "adrenal_gland_left", "small_bowel", "duodenum", "colon",
           "urinary_bladder", "prostate", "kidney_cyst_left", "kidney_cyst_right", "aorta",
           "inferior_vena_cava", "portal_vein_and_splenic_vein", "iliac_artery_left",
           "iliac_artery_right", "iliac_vena_left", "iliac_vena_right", "esophagus"]

MUSCLE_HU = (-29, 150)
SAT_HU = (-190, -30)
VAT_HU = (-150, -50)
CALCIUM_HU = 130
METAL_HU = 1800


# ----------------------------------------------------------------------------- helpers


def bbox(mask: np.ndarray, margin: int = 0) -> tuple[slice, ...] | None:
    out = []
    for ax in range(mask.ndim):
        other = tuple(a for a in range(mask.ndim) if a != ax)
        idx = np.flatnonzero(mask.any(axis=other))
        if idx.size == 0:
            return None
        out.append(slice(max(int(idx[0]) - margin, 0), min(int(idx[-1]) + 1 + margin, mask.shape[ax])))
    return tuple(out)


def touches_z_border(mask: np.ndarray) -> bool:
    return bool(mask[:, :, 0].any() or mask[:, :, -1].any())


def _bone_names(seg: Segmentation) -> list[str]:
    pref = ("vertebrae_", "rib_", "sacrum", "sternum", "costal_cartilages", "hip_", "femur_",
            "scapula_", "clavicula_", "humerus_", "skull")
    return [n for n in seg.names if n.startswith(pref)]


def _disk(radius_px: float) -> np.ndarray:
    r = max(int(np.ceil(radius_px)), 1)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx**2 + yy**2) <= radius_px**2 + 1e-6


def _dilate_inplane(mask: np.ndarray, mm: float, spacing) -> np.ndarray:
    """Dilate each axial slice by `mm` (in-plane), cheap and adequate for exclusion margins."""
    st = _disk(mm / spacing[0])[:, :, None]
    return ndi.binary_dilation(mask, structure=st)


def core_values(hu: np.ndarray, mask: np.ndarray, spacing, erode_mm: float) -> np.ndarray:
    """HU samples from the interior of a structure (≥ erode_mm from its surface).

    Sampling the core avoids partial-volume voxels and small segmentation boundary errors,
    which is what makes a mean attenuation robust to an imperfect mask.
    """
    box = bbox(mask, 1)
    if box is None:
        return np.empty(0, dtype=np.float32)
    step = tuple(max(int(round(1.5 / s)), 1) for s in spacing)  # ~1.5 mm sampling is plenty
    sub = tuple(slice(b.start, b.stop, st) for b, st in zip(box, step))
    m = mask[sub]
    if not m.any():
        return np.empty(0, dtype=np.float32)
    padded = np.pad(m, 1)
    dist = ndi.distance_transform_edt(padded, sampling=[s * st for s, st in zip(spacing, step)])[1:-1, 1:-1, 1:-1]
    return hu[sub][dist >= erode_mm].astype(np.float32)


# ----------------------------------------------------------------------------- contrast phase


def contrast_state(study: Study, seg: Segmentation) -> dict:
    """Classify the scan as unenhanced / enhanced from blood-pool attenuation.

    Unenhanced blood measures roughly 30–55 HU; IV contrast raises it well above that.
    HU-threshold biomarkers (liver fat, calcium) are only valid without contrast, so an
    uncertain result is treated as 'enhanced' by the rules that depend on it.
    """
    probes = {}
    for name, erode in (("aorta", 3.0), ("heart", 8.0), ("inferior_vena_cava", 3.0),
                        ("kidney_left", 4.0), ("kidney_right", 4.0)):
        vals = core_values(study.pixels, seg.mask(name), study.spacing, erode)
        if vals.size >= 50:
            probes[name] = float(np.median(vals))
    if not probes:
        return {"state": "unknown", "probes": probes}
    peak = max(probes.values())
    state = "enhanced" if peak > 80 else ("uncertain" if peak > 60 else "unenhanced")
    return {"state": state, "probes": {k: round(v, 1) for k, v in probes.items()}, "peak_blood_hu": round(peak, 1)}


# ----------------------------------------------------------------------------- bone


def vertebral_roi(hu: np.ndarray, mask: np.ndarray, spacing) -> dict | None:
    """Trabecular ROI at the centre of a vertebral body.

    The vertebral body is by far the thickest part of a vertebra, so the maximum of the
    3D distance transform of the whole-vertebra mask lands at the mid-body centre without
    needing a separate body/posterior-element model. The ROI is a short cylinder well
    inside the cortex (0.6·R in-plane, ±0.3·R cranio-caudal; R = inscribed radius).
    """
    box = bbox(mask, 2)
    if box is None:
        return None
    m = mask[box]
    dist = ndi.distance_transform_edt(m, sampling=spacing)
    r = float(dist.max())
    if r < 5.0:  # collapsed vertebra or a fragmentary mask: no trustworthy ROI
        return {"status": "too_small", "inscribed_radius_mm": r}
    c = np.array(np.unravel_index(int(np.argmax(dist)), dist.shape))
    xs, ys, zs = np.ogrid[:m.shape[0], :m.shape[1], :m.shape[2]]
    inplane = ((xs - c[0]) * spacing[0])**2 + ((ys - c[1]) * spacing[1])**2 <= (0.6 * r)**2
    half_h = max(0.3 * r, spacing[2] * 0.51)
    roi = inplane & (np.abs((zs - c[2]) * spacing[2]) <= half_h) & (dist >= 0.25 * r)
    vals = hu[box][roi].astype(np.float32)
    if vals.size < 20:
        return {"status": "too_small", "inscribed_radius_mm": r}
    full = np.zeros(mask.shape, dtype=bool)
    full[box] = roi
    centre = tuple(int(b.start + ci) for b, ci in zip(box, c))
    return {"status": "ok", "mean_hu": float(vals.mean()), "sd_hu": float(vals.std()), "n_voxels": int(vals.size),
            "inscribed_radius_mm": r, "centre": centre, "roi": full}


def bone_density(study: Study, seg: Segmentation) -> tuple[list[Measurement], dict]:
    levels, rois = {}, {}
    for lv in VERTEBRA_LEVELS:
        mask = seg.mask(f"vertebrae_{lv}")
        if not mask.any():
            continue
        vol_ml = mask.sum() * study.voxel_volume_mm3 / 1000.0
        if touches_z_border(mask) or vol_ml < 12.0:
            levels[lv] = {"status": "truncated"}
            continue
        res = vertebral_roi(study.pixels, mask, study.spacing)
        if res is None:
            continue
        if res["status"] == "ok":
            rois[lv] = res.pop("roi")
            if res["mean_hu"] > 350:
                res["status"] = "nonphysiologic"  # hardware, cement, dense sclerotic lesion
        levels[lv] = res
    ms: list[Measurement] = []
    chosen = next((lv for lv in LEVEL_PREFERENCE if levels.get(lv, {}).get("status") == "ok"), None)
    summary = {k: {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in levels.items()}
    if chosen is None:
        return ms, {"levels": summary, "chosen": None, "rois": {}}
    notes = []
    if chosen != "L1":
        notes.append("level_substituted")
    ok_vals = [v["mean_hu"] for v in levels.values() if v.get("status") == "ok"]
    if len(ok_vals) >= 3 and abs(levels[chosen]["mean_hu"] - float(np.median(ok_vals))) > 60:
        notes.append("level_outlier")  # focal lesion or fracture at the measured level
    kvp = study.meta.get("kvp")
    if kvp is not None and abs(kvp - 120) > 5:
        notes.append("kvp_not_120")
    elif kvp is None:
        notes.append("kvp_unknown")
    sem = levels[chosen]["sd_hu"] / np.sqrt(levels[chosen]["n_voxels"])
    if sem > 8.0:
        notes.append("high_noise")  # the ROI mean itself is uncertain by more than ~±16 HU
    rel = "moderate" if {"level_outlier", "kvp_not_120", "high_noise"} & set(notes) else "high"
    ms.append(Measurement("bone.trabecular_hu", levels[chosen]["mean_hu"], "HU", rel, notes,
                          {"level": chosen, "sd_hu": round(levels[chosen]["sd_hu"], 1), "sem_hu": round(float(sem), 1)}))
    return ms, {"levels": summary, "chosen": chosen, "rois": {chosen: rois[chosen]}}


# ----------------------------------------------------------------------------- liver / spleen


def organ_attenuation(study: Study, seg: Segmentation, name: str, erode_mm: float) -> dict | None:
    mask = seg.mask(name)
    if not mask.any():
        return None
    vals = core_values(study.pixels, mask, study.spacing, erode_mm)
    vals = vals[(vals > -100) & (vals < 200)]  # drop calcifications, clips, gas
    if vals.size < 200:
        return None
    return {"median_hu": float(np.median(vals)), "iqr_hu": float(np.subtract(*np.percentile(vals, [75, 25]))),
            "volume_ml": float(mask.sum() * study.voxel_volume_mm3 / 1000.0),
            "truncated": touches_z_border(mask), "n": int(vals.size)}


def liver_spleen(study: Study, seg: Segmentation, contrast: dict) -> list[Measurement]:
    ms: list[Measurement] = []
    liver = organ_attenuation(study, seg, "liver", 8.0)
    spleen = organ_attenuation(study, seg, "spleen", 5.0)
    unenh = contrast["state"] == "unenhanced"
    notes = [] if unenh else ["contrast_" + contrast["state"]]
    rel = "high" if unenh else "invalid"
    if liver:
        ln = notes + (["organ_truncated"] if liver["truncated"] else [])
        ms.append(Measurement("liver.attenuation_hu", liver["median_hu"], "HU", rel, ln, {"iqr_hu": round(liver["iqr_hu"], 1)}))
        ms.append(Measurement("liver.volume_ml", liver["volume_ml"], "mL",
                              "invalid" if liver["truncated"] else "moderate",
                              ["organ_truncated"] if liver["truncated"] else []))
    if spleen:
        ms.append(Measurement("spleen.attenuation_hu", spleen["median_hu"], "HU", rel, notes))
        ms.append(Measurement("spleen.volume_ml", spleen["volume_ml"], "mL",
                              "invalid" if spleen["truncated"] else "moderate",
                              ["organ_truncated"] if spleen["truncated"] else []))
    if liver and spleen:
        ms.append(Measurement("liver.minus_spleen_hu", liver["median_hu"] - spleen["median_hu"], "HU", rel, notes))
    return ms


# ----------------------------------------------------------------------------- aorta


def inscribed_diameter_mm(component: np.ndarray, px: float) -> float:
    """Largest inscribed-circle diameter of a 2D cross-section.

    For a tube cut obliquely the section is an ellipse whose *minor* axis is the true
    diameter; the inscribed circle measures exactly that, so tortuosity does not inflate
    the result the way an area-equivalent or bounding-box diameter would.
    Discretisation: staircase edges make the inscribed circle of a pixelated shape smaller
    than the true one — by 1.33 px for a disc and ≈0.35 px for an elongated ellipse
    (measured over random sub-pixel positions). The midpoint 0.85 px is added back, leaving
    a residual shape-dependent error within ±0.5 px. The 4× upsampling only makes the
    distance transform itself precise.
    """
    f = 4
    up = np.kron(np.pad(component, 1), np.ones((f, f), dtype=bool))
    d = ndi.distance_transform_edt(up) * (px / f)
    return max(2.0 * float(d.max()) - px / f + 0.85 * px, 0.0)


def _landmark_z(seg: Segmentation, name: str) -> float | None:
    m = seg.mask(name)
    if not m.any():
        return None
    z = np.flatnonzero(m.any(axis=(0, 1)))
    return float(z.mean())


def aorta_geometry(study: Study, seg: Segmentation) -> tuple[list[Measurement], dict]:
    mask = seg.mask("aorta")
    box = bbox(mask, 1)
    if box is None:
        return [], {}
    sx, sy, sz = study.spacing
    if abs(sx - sy) > 0.02 * sx:
        return [], {"error": "anisotropic_inplane"}
    m = mask[box]
    z0 = box[2].start
    # Diaphragmatic boundary between thoracic and abdominal aorta ≈ T12 body.
    z_split = _landmark_z(seg, "vertebrae_T12")
    if z_split is None:
        z_l1 = _landmark_z(seg, "vertebrae_L1")
        lung = seg.union(LUNG_LOBES)
        if z_l1 is not None:
            z_split = z_l1 + 30.0 / sz
        elif lung.any():
            z_split = float(np.flatnonzero(lung.any(axis=(0, 1)))[0])
        elif seg.has("kidney_left") or seg.has("kidney_right") or seg.has("vertebrae_L3"):
            z_split = float(mask.shape[2])  # only abdomen in view
        else:
            z_split = None

    rows = []  # (z, segment, diameter, centroid xy)
    min_area_px = 80.0 / (sx * sy)
    per_slice = []
    for k in range(m.shape[2]):
        lab, n = ndi.label(m[:, :, k])
        comps = []
        for i in range(1, n + 1):
            c = lab == i
            if c.sum() < min_area_px:
                continue
            cb = bbox(c, 0)
            d = inscribed_diameter_mm(c[cb], sx)
            cy = float(np.nonzero(c)[1].mean())
            cx = float(np.nonzero(c)[0].mean())
            comps.append((d, cx + box[0].start, cy + box[1].start))
        per_slice.append(comps)
    two_comp_z = [k + z0 for k, comps in enumerate(per_slice) if len(comps) >= 2]
    arch_floor = max(two_comp_z) if two_comp_z else None
    for k, comps in enumerate(per_slice):
        z = k + z0
        if not comps:
            continue
        if z_split is not None and z < z_split:
            d, cx, cy = max(comps)
            rows.append((z, "abdominal", d, cx, cy))
        elif z_split is None:
            d, cx, cy = max(comps)
            rows.append((z, "unspecified", d, cx, cy))
        elif len(comps) >= 2:
            comps = sorted(comps, key=lambda c: c[2])  # by y: posterior first
            rows.append((z, "descending", *comps[0]))
            rows.append((z, "ascending", *comps[-1]))
        elif arch_floor is not None and z > arch_floor:
            rows.append((z, "arch", *comps[0]))
        else:
            rows.append((z, "descending", *comps[0]))

    ms, where = [], {}
    win = max(int(round(6.0 / sz)), 1) | 1  # ≥6 mm rolling median rejects single-slice mask spikes
    for segname in ("ascending", "arch", "descending", "abdominal", "unspecified"):
        seq = sorted(r for r in rows if r[1] == segname)
        if len(seq) < max(win, 3):
            continue
        d = np.array([r[2] for r in seq])
        smooth = ndi.median_filter(d, size=win, mode="nearest") if win > 1 else d
        i = int(np.argmax(smooth))
        notes = ["segmentation_fast_model"] if seg.source.endswith("fast") else []
        ms.append(Measurement(f"aorta.{segname}_max_diameter_mm", float(smooth[i]), "mm", "moderate", notes,
                              {"extent_mm": round(len(seq) * sz, 0)}))
        where[segname] = {"z": int(seq[i][0]), "x": seq[i][3], "y": seq[i][4], "d": float(smooth[i])}
    return ms, {"where": where, "z_split": z_split}


# ----------------------------------------------------------------------------- calcium


def agatston(hu: np.ndarray, roi: np.ndarray, spacing, min_area_mm2: float = 1.0) -> dict:
    """Agatston-method calcium score inside `roi` (Agatston 1990).

    Per axial slice: connected lesions ≥130 HU and ≥1 mm² score area × density weight
    (1: 130–199, 2: 200–299, 3: 300–399, 4: ≥400 HU). Thin slices are first averaged to
    ≈3 mm slabs (the thickness the method was defined on, and far less noisy); the sum is
    scaled by slab thickness / 3 mm. Lesions reaching metal density are excluded.
    """
    box = bbox(roi, 0)
    empty = {"score": 0.0, "volume_mm3": 0.0, "n_lesions": 0, "metal_excluded": 0, "lesion_mask": None, "box": box}
    if box is None:
        return empty
    h = hu[box].astype(np.float32)
    r = roi[box]
    sx, sy, sz = spacing
    n = max(int(round(3.0 / sz)), 1)
    if n > 1:
        nz = (h.shape[2] // n) * n
        if nz == 0:
            return empty
        h = h[:, :, :nz].reshape(h.shape[0], h.shape[1], nz // n, n).mean(axis=3)
        r = r[:, :, :nz].reshape(r.shape[0], r.shape[1], nz // n, n).mean(axis=3) >= 0.5
    slab = sz * n
    px_area = sx * sy
    min_px = max(int(np.ceil(min_area_mm2 / px_area)), 3)
    cand = r & (h >= CALCIUM_HU)
    lesion = np.zeros(cand.shape, dtype=bool)
    score, n_les, metal = 0.0, 0, 0
    eight = np.ones((3, 3), dtype=bool)
    for k in np.flatnonzero(cand.any(axis=(0, 1))):
        lab, cnt = ndi.label(cand[:, :, k], structure=eight)
        if cnt == 0:
            continue
        idx = np.arange(1, cnt + 1)
        sizes = ndi.sum(cand[:, :, k], lab, idx)
        peaks = ndi.maximum(h[:, :, k], lab, idx)
        for i, size, peak in zip(idx, sizes, peaks):
            if size < min_px:
                continue
            if peak >= METAL_HU:
                metal += 1
                continue
            w = 1 if peak < 200 else 2 if peak < 300 else 3 if peak < 400 else 4
            score += size * px_area * w
            n_les += 1
            lesion[:, :, k] |= lab == i
    return {"score": score * slab / 3.0, "volume_mm3": float(lesion.sum() * px_area * slab), "n_lesions": n_les,
            "metal_excluded": metal, "lesion_mask": lesion, "box": box, "slab_n": n}


def noise_sd(study: Study, seg: Segmentation) -> float | None:
    """Image noise (HU SD) estimated in homogeneous blood pool / liver parenchyma."""
    for name, erode in (("aorta", 4.0), ("heart", 10.0), ("liver", 10.0)):
        vals = core_values(study.pixels, seg.mask(name), study.spacing, erode)
        if vals.size >= 200:
            lo, hi = np.percentile(vals, [5, 95])
            core = vals[(vals >= lo) & (vals <= hi)]
            return float(core.std() / 0.79)  # 5–95% trimmed SD → Gaussian-equivalent SD
    return None


def calcium(study: Study, seg: Segmentation, contrast: dict, z_split: float | None) -> tuple[list[Measurement], dict]:
    ms: list[Measurement] = []
    overlay = {}
    if contrast["state"] != "unenhanced":
        return ms, overlay  # opacified blood exceeds 130 HU: the method is undefined
    sp = study.spacing
    notes = []
    thick = study.meta.get("slice_thickness") or sp[2]
    if thick > 5.0:
        notes.append("thick_slices")
    sd = noise_sd(study, seg)
    n_avg = max(int(round(3.0 / sp[2])), 1)
    eff_sd = sd / np.sqrt(n_avg) if sd is not None else None
    if eff_sd is not None and eff_sd > 35:
        # Noise alone produces ≥130 HU clusters: a score here would be fiction, so abstain.
        return ms, {"abstained": "high_noise", "noise_sd": round(float(eff_sd), 1)}
    if eff_sd is not None and eff_sd > 25:
        notes.append("high_noise")
    if seg.source.endswith("fast"):
        # On real scans the score moved by ~100 units between the 3 mm model's masks and
        # reference masks (docs/validation): boundary-sensitive, so never above "low" here.
        notes.append("segmentation_fast_model")
    rel = "low" if {"high_noise", "segmentation_fast_model"} & set(notes) else "moderate"
    bone = _dilate_inplane(seg.union(_bone_names(seg)), 2.0, sp)

    aorta = seg.mask("aorta")
    if aorta.any() and z_split is not None:
        abd = aorta.copy()
        abd[:, :, int(np.ceil(z_split)):] = False
        if abd.sum() * sp[2] * sp[0] * sp[1] > 3000:
            roi = _dilate_inplane(abd, 2.0, sp) & ~bone
            res = agatston(study.pixels, roi, sp)
            extra = {"volume_mm3": round(res["volume_mm3"], 1), "n_lesions": res["n_lesions"], "metal_excluded": res["metal_excluded"]}
            ms.append(Measurement("aorta.abdominal_calcium_agatston", res["score"], "Agatston", rel,
                                  notes + (["metal_excluded"] if res["metal_excluded"] else []), extra))
            overlay["aorta"] = res

    heart = seg.mask("heart")
    if heart.any() and not touches_z_border(heart):
        st = ndi.generate_binary_structure(3, 1)
        it = max(int(round(4.0 / sp[0])), 1)
        box = bbox(heart, it + 2)
        region = np.zeros(heart.shape, dtype=bool)
        hb = ndi.binary_dilation(_dilate_inplane(heart[box], 4.0, sp), structure=st, iterations=max(int(round(3.0 / sp[2])), 1))
        lab = seg.labels[box]
        allowed = (lab == 0) | heart[box] | seg.mask("atrial_appendage_left")[box]
        ao = _dilate_inplane(aorta[box], 2.0, sp)
        region[box] = hb & allowed & ~ao & ~bone[box]
        res = agatston(study.pixels, region, sp)
        extra = {"volume_mm3": round(res["volume_mm3"], 1), "n_lesions": res["n_lesions"], "metal_excluded": res["metal_excluded"]}
        ms.append(Measurement("heart.calcium_agatston", res["score"], "Agatston", rel,
                              notes + ["non_gated", "includes_valvular"] + (["metal_excluded"] if res["metal_excluded"] else []), extra))
        overlay["heart"] = res
    return ms, overlay


# ----------------------------------------------------------------------------- lungs

SHARP_KERNEL_HINTS = ("LUNG", "BONE", "EDGE", "SHARP", "B60", "B70", "B80", "BR6", "BL6", "I70", "FC5", "FC8", "YC", "YD")


def lungs(study: Study, seg: Segmentation) -> list[Measurement]:
    mask = seg.union(LUNG_LOBES)
    box = bbox(mask, 0)
    if box is None:
        return []
    vals = study.pixels[box][mask[box]]
    vol_l = vals.size * study.voxel_volume_mm3 / 1e6
    truncated = touches_z_border(mask)
    if vol_l < 1.0:
        return []  # only lung bases/apices in view: an emphysema index would be meaningless
    notes = []
    if truncated:
        notes.append("lungs_truncated")
    kernel = (study.meta.get("kernel") or "").upper()
    if any(h in kernel for h in SHARP_KERNEL_HINTS):
        notes.append("sharp_kernel")  # edge enhancement inflates the −950 HU voxel fraction
    elif not kernel:
        notes.append("kernel_unknown")
    thick = study.meta.get("slice_thickness") or study.spacing[2]
    if thick > 3.0:
        notes.append("thick_slices")
    sd = noise_sd(study, seg)
    if sd is not None and sd > 40:
        notes.append("high_noise")  # noise alone pushes normal lung voxels below −950 HU
    rel = "low" if {"sharp_kernel", "high_noise"} & set(notes) else ("moderate" if notes else "high")
    laa = float((vals < -950).mean() * 100.0)
    return [
        Measurement("lung.laa950_percent", laa, "%", rel, notes,
                    {"perc15_hu": round(float(np.percentile(vals, 15)), 1), "mean_hu": round(float(vals.mean()), 1)}),
        Measurement("lung.volume_l", vol_l, "L", "invalid" if truncated else "moderate", ["lungs_truncated"] if truncated else []),
    ]


# ----------------------------------------------------------------------------- body composition (L3)


def _largest_component(mask2d: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask2d)
    if n <= 1:
        return mask2d
    sizes = ndi.sum(mask2d, lab, np.arange(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def _drop_small(mask2d: np.ndarray, min_px: float) -> np.ndarray:
    lab, n = ndi.label(mask2d)
    if n == 0:
        return mask2d
    sizes = ndi.sum(mask2d, lab, np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_px
    return keep[lab]


def _circular_fill(values: np.ndarray, smooth: int) -> np.ndarray:
    """Interpolate NaNs around the circle, then median-smooth (rejects angular spikes)."""
    n = values.size
    ok = ~np.isnan(values)
    idx = np.arange(n)
    filled = np.interp(idx, idx[ok], values[ok], period=n)
    return ndi.median_filter(filled, size=smooth, mode="wrap")


def l3_slice_composition(hu2d: np.ndarray, lab2d: np.ndarray, seg: Segmentation, px: float) -> dict:
    """Tissue areas on one axial slice by standard HU windows constrained by anatomy.

    Skeletal muscle: −29…150 HU, SAT: −190…−30 HU, VAT: −150…−50 HU (the windows used by
    Prado 2008 / Martin 2013). Anatomy constraints: segmented viscera and bone are never
    muscle. The muscular abdominal wall is traced in polar coordinates around the body
    centroid: its outer radius separates subcutaneous fat, its inner radius bounds the
    visceral compartment; gaps (linea alba, flanks) are bridged by circular interpolation.
    """
    area = px * px
    nb = 360

    def ids(names):
        return [seg.names[n] for n in names if n in seg.names]

    body = ndi.binary_opening(hu2d > -500, structure=_disk(3.0 / px))
    body = ndi.binary_fill_holes(_largest_component(body))
    truncated = bool(body[0, :].any() or body[-1, :].any() or body[:, 0].any() or body[:, -1].any())
    inner = ndi.binary_erosion(body, structure=_disk(3.0 / px))  # removes skin
    bone = ndi.binary_dilation(np.isin(lab2d, ids(_bone_names(seg))), structure=_disk(2.0 / px))
    visc = ndi.binary_dilation(np.isin(lab2d, ids(VISCERA)), structure=_disk(2.0 / px))
    lean = (hu2d >= MUSCLE_HU[0]) & (hu2d <= MUSCLE_HU[1])
    ts_muscle = np.isin(lab2d, ids(TS_MUSCLES))
    cand = inner & lean & ~visc & ~bone & ~ts_muscle
    cand = ndi.binary_opening(cand, structure=_disk(1.5 / px))  # thin strands: vessels, fascia, bowel wall
    cand = _drop_small(cand, 50.0 / area)
    big = _drop_small(cand, 300.0 / area)

    cx, cy = ndi.center_of_mass(body)
    ii, jj = np.indices(hu2d.shape)
    rad = np.hypot(ii - cx, jj - cy) * px
    ang = ((np.arctan2(jj - cy, ii - cx) + np.pi) / (2 * np.pi) * nb).astype(int) % nb
    bins = np.arange(nb)

    def polar(mask, func, fill):
        out = np.asarray(func(np.where(mask, rad, fill), ang, bins), dtype=float)
        out[out == fill] = np.nan
        return out

    r_body = _circular_fill(polar(body, ndi.maximum, -1.0), 1)
    wall = big | ts_muscle | bone
    r_out = polar(wall, ndi.maximum, -1.0)
    coverage = float(np.mean(~np.isnan(r_out)))
    closed = coverage >= 0.55
    if closed:
        thick = _circular_fill(r_body - r_out, 9)          # subcutaneous thickness per angle
        r_wall = r_body - np.clip(thick, 0, None)
        r_in = polar(wall & (rad >= 0.5 * r_wall[ang]), ndi.minimum, 1e9)
        r_in = np.minimum(_circular_fill(r_in, 15), r_wall)
        inside_wall = rad <= r_wall[ang]
        cavity = rad < r_in[ang]
        # Lean tissue lying wholly inside the visceral compartment is unsegmented viscera.
        lab, n = ndi.label(cand)
        if n:
            frac = np.asarray(ndi.mean(rad < (r_in[ang] - 4.0), lab, np.arange(1, n + 1)))
            drop = np.zeros(n + 1, dtype=bool)
            drop[1:] = frac > 0.5
            cand = cand & ~drop[lab]
    muscle = (ts_muscle & lean) | cand
    fat_sat = inner & (hu2d >= SAT_HU[0]) & (hu2d <= SAT_HU[1])
    fat_vat = inner & (hu2d >= VAT_HU[0]) & (hu2d <= VAT_HU[1])
    if closed:
        sat = fat_sat & ~inside_wall
        vat = fat_vat & cavity & ~muscle
    else:
        hull = ndi.binary_fill_holes(ndi.binary_closing(visc | ts_muscle | bone, structure=_disk(15.0 / px)))
        vat = fat_vat & hull & ~muscle
        sat = fat_sat & ~hull & ~muscle
    vals = hu2d[muscle]
    return {"muscle_cm2": muscle.sum() * area / 100.0, "sat_cm2": sat.sum() * area / 100.0,
            "vat_cm2": vat.sum() * area / 100.0,
            "muscle_hu": float(vals.mean()) if vals.size else float("nan"),
            "closed": bool(closed), "wall_coverage": coverage, "truncated": truncated,
            "masks": {"muscle": muscle, "sat": sat, "vat": vat}}


def body_composition(study: Study, seg: Segmentation, contrast: dict, l3_centre: tuple | None) -> tuple[list[Measurement], dict]:
    if l3_centre is None:
        return [], {}
    sx, sy, sz = study.spacing
    if abs(sx - sy) > 0.02 * sx:
        return [], {}
    zc = l3_centre[2]
    # Reference technique is a ~5 mm soft-tissue slice: average thin slices into one slab.
    half = int(round(2.5 / sz - 0.5)) if sz < 5 else 0
    z0, z1 = max(zc - half, 0), min(zc + half + 1, study.pixels.shape[2])
    slab = study.pixels[:, :, z0:z1].astype(np.float32).mean(axis=2)
    lab2d = seg.labels[:, :, zc]
    notes = []
    ref = np.isin(lab2d, [seg.names[n] for n in TS_MUSCLES]) & (slab > -29) & (slab < 150)
    ref = ndi.binary_erosion(ref, iterations=2)
    if ref.sum() >= 50:
        v = slab[ref]
        sd = float(1.4826 * np.median(np.abs(v - np.median(v))))
        if sd > 12.0:
            # HU windows classify single voxels, so noise must be brought to a standard
            # level first; Gaussian σ chosen to reach ≈12 HU (white-noise approximation).
            sigma_px = min(sd / 42.5, 2.5 / sx)
            slab = ndi.gaussian_filter(slab, sigma_px)
        if sd > 45.0:
            notes.append("high_noise")
    res = l3_slice_composition(slab, lab2d, seg, sx)
    if res["truncated"]:
        notes.append("body_outside_fov")
    if not res["closed"]:
        notes.append("abdominal_wall_not_closed")
    if seg.source.endswith("fast"):
        notes.append("segmentation_fast_model")
    rel = "low" if {"abdominal_wall_not_closed", "high_noise"} & set(notes) else "moderate"
    fat_rel = "invalid" if "body_outside_fov" in notes else rel
    ms = [Measurement("body.l3_muscle_area_cm2", res["muscle_cm2"], "cm²", rel, notes),
          Measurement("body.l3_muscle_hu", res["muscle_hu"], "HU", rel,
                      notes + ([] if contrast["state"] == "unenhanced" else ["contrast_" + contrast["state"]])),
          Measurement("body.l3_vat_cm2", res["vat_cm2"], "cm²", rel, notes),
          Measurement("body.l3_sat_cm2", res["sat_cm2"], "cm²", fat_rel, notes)]
    h = study.meta.get("height_m")
    if h:
        ms.append(Measurement("body.l3_smi_cm2_m2", res["muscle_cm2"] / (h * h), "cm²/m²", rel, notes))
    if res["sat_cm2"] > 1 and fat_rel != "invalid":
        ms.append(Measurement("body.vat_sat_ratio", res["vat_cm2"] / res["sat_cm2"], "", rel, notes))
    return ms, {"z": zc, "masks": res["masks"]}
