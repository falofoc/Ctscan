"""End-to-end analysis: ingest → quality control → anatomy → biomarkers → findings."""
from __future__ import annotations

import base64
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__, biomarkers as B, render as R, rules
from .io import load_any
from .types import Measurement, Segmentation, Study

INGEST_WARNINGS_AR = {
    "modality_inferred": ("info", "نوع الفحص استُنتج من قيم الصورة (ملف NIfTI لا يحمل وسم الجهاز)."),
    "no_acquisition_metadata": ("warn", "لا توجد بيانات اكتساب (kVp، مرشح إعادة البناء، سماكة المقطع)؛ بعض ضوابط الجودة غير ممكنة."),
    "nonuniform_slice_spacing": ("error", "تباعد المقاطع غير منتظم أو توجد مقاطع مفقودة؛ الحجوم والمسافات غير موثوقة."),
    "duplicate_slice_positions": ("warn", "وُجدت مقاطع مكررة الموضع وتم تجاهلها."),
    "mixed_orientation_slices_dropped": ("warn", "استُبعدت مقاطع باتجاه مختلف عن بقية السلسلة."),
    "gantry_tilt_or_oblique": ("warn", "السلسلة مائلة (Gantry tilt)؛ القياسات الهندسية تقريبية."),
    "slice_gap": ("warn", "توجد فجوات بين المقاطع (التباعد أكبر من السماكة)."),
    "uncalibrated_image": ("warn", "صورة عادية غير معايرة (ليست DICOM)؛ لا يمكن التحقق من وضعية التصوير أو المقياس."),
    "multiframe_middle_frame_used": ("info", "ملف متعدد الإطارات؛ استُخدم الإطار الأوسط فقط."),
}


def _qc(study: Study) -> list[dict]:
    items = []
    for code in study.warnings:
        level, text = INGEST_WARNINGS_AR.get(code, ("info", code))
        items.append({"code": code, "level": level, "text_ar": text})
    if study.kind != "ct":
        return items
    px = study.pixels
    lo = float(np.percentile(px[::4, ::4, ::2], 0.5))
    if lo > -850:
        items.append({"code": "hu_calibration_suspect", "level": "warn",
                      "text_ar": f"لا يظهر هواء بقيمة قريبة من −1000 HU (أدنى القيم ≈ {lo:.0f})؛ يُشتبه بخلل في معايرة HU أو مجال رؤية ضيق جدًا."})
    sx, sy, sz = study.spacing
    if abs(sx - sy) > 0.02 * sx:
        items.append({"code": "anisotropic_inplane", "level": "error", "text_ar": "أبعاد البكسل غير متساوية داخل المقطع؛ القياسات الهندسية معطّلة."})
    thick = study.meta.get("slice_thickness") or sz
    if max(thick, sz) > 5.0:
        items.append({"code": "thick_slices", "level": "warn", "text_ar": f"سماكة/تباعد المقاطع {max(thick, sz):.1f} مم (>5 مم)؛ دقة القياسات منخفضة."})
    if px.shape[2] * sz < 60:
        items.append({"code": "short_coverage", "level": "warn", "text_ar": "التغطية الطولية للفحص قصيرة جدًا."})
    kvp = study.meta.get("kvp")
    if kvp is not None and abs(kvp - 120) > 5:
        items.append({"code": "kvp_not_120", "level": "warn", "text_ar": f"جهد الأنبوب {kvp:.0f} kVp؛ عتبات HU المرجعية مشتقة عند 120 kVp."})
    return items


def _b64(png: bytes) -> str:
    return base64.b64encode(png).decode()


def _images_ct(study: Study, seg: Segmentation, bone_info: dict, aorta_info: dict, calc_overlay: dict, bc_info: dict) -> list[dict]:
    hu = study.pixels
    X, Y, Z = hu.shape
    sx, sy, sz = study.spacing
    out = []

    rois = bone_info.get("rois") or {}
    if rois:
        lv, roi = next(iter(rois.items()))
        cx, cy, cz = bone_info["levels"][lv]["centre"]
        g = R.window(R.sagittal(hu, cx), 300, 1400)
        m = R.sagittal(roi, cx)
        rs, cs = R.crop_around(g.shape, (Z - 1 - cz, Y - 1 - cy), int(90 / min(sy, sz)))
        out.append({"key": "bone", "title_ar": f"منطقة قياس كثافة العظم — الفقرة {lv} (مقطع سهمي)",
                    "png": _b64(R.compose(g[rs, cs], [(m[rs, cs], "green", "fill")], aspect=sz / sy))})

    liver = seg.mask("liver")
    if liver.any():
        z = int(np.argmax(liver.sum(axis=(0, 1))))
        g = R.window(R.axial(hu, z), 50, 350)
        out.append({"key": "liver", "title_ar": "الكبد (أخضر) والطحال (سماوي) — حدود التقطيع",
                    "png": _b64(R.compose(g, [(R.axial(liver, z), "green", "edge"), (R.axial(seg.mask("spleen"), z), "cyan", "edge")]))})

    where = aorta_info.get("where") or {}
    if where:
        name, w = max(where.items(), key=lambda kv: kv[1]["d"] / (30.0 if kv[0] == "abdominal" else 40.0))
        z = w["z"]
        g = R.window(R.axial(hu, z), 60, 400)
        rs, cs = R.crop_around(g.shape, (int(Y - 1 - w["y"]), int(X - 1 - w["x"])), int(70 / sx))
        ar = {"abdominal": "البطني", "ascending": "الصاعد", "descending": "النازل", "arch": "القوس", "unspecified": ""}[name]
        out.append({"key": "aorta", "title_ar": f"الأبهر {ar} عند موضع أقصى قطر (≈ {w['d']:.0f} مم)",
                    "png": _b64(R.compose(g[rs, cs], [(R.axial(seg.mask("aorta"), z)[rs, cs], "red", "edge")]))})

    for key, title in (("heart", "تكلّسات منطقة القلب (أحمر)"), ("aorta", "تكلّسات الأبهر البطني (أحمر)")):
        res = calc_overlay.get(key)
        if not res or res.get("lesion_mask") is None or not res["lesion_mask"].any():
            continue
        les, box, n = res["lesion_mask"], res["box"], res.get("slab_n", 1)
        k = int(np.argmax(les.sum(axis=(0, 1))))
        z = min(box[2].start + k * n + n // 2, Z - 1)
        full = np.zeros((X, Y), dtype=bool)
        full[box[0], box[1]] = les[:, :, k]
        xs, ys = np.nonzero(full)
        g = R.window(R.axial(hu, z), 100, 700)
        m = full.T[::-1, ::-1]
        rs, cs = R.crop_around(g.shape, (int(Y - 1 - ys.mean()), int(X - 1 - xs.mean())), int(80 / sx))
        out.append({"key": f"calcium_{key}", "title_ar": title, "png": _b64(R.compose(g[rs, cs], [(m[rs, cs], "red", "fill")]))})

    if bc_info.get("masks"):
        z = bc_info["z"]
        g = R.window(R.axial(hu, z), 40, 400)
        mk = {k: v.T[::-1, ::-1] for k, v in bc_info["masks"].items()}
        out.append({"key": "body", "title_ar": "تركيب الجسم عند L3: عضلات (أحمر)، دهن حشوي (أصفر)، دهن تحت الجلد (سماوي)",
                    "png": _b64(R.compose(g, [(mk["sat"], "cyan", "fill"), (mk["vat"], "yellow", "fill"), (mk["muscle"], "red", "fill")]))})

    lung = seg.union(B.LUNG_LOBES)
    if lung.any():
        y = int(np.median(np.nonzero(lung.any(axis=(0, 2)))[0]))
        g = R.window(R.coronal(hu, y), -600, 1500)
        lm = R.coronal(lung, y)
        low = lm & (R.coronal(hu, y) < -950)
        out.append({"key": "lung", "title_ar": "الرئتان (مقطع إكليلي): المناطق تحت −950 HU بالأحمر",
                    "png": _b64(R.compose(g, [(lm, "cyan", "edge"), (low, "red", "fill")], aspect=sz / sx))})
    return out


def analyze_ct(study: Study, seg: Segmentation | None = None, progress=None) -> dict:
    def step(msg):
        if progress:
            progress(msg)

    timings = {}
    qc = _qc(study)
    if seg is None:
        from . import segment

        step("تقطيع البنى التشريحية")
        t = time.time()
        seg = segment.segment_ct(study)
        timings["segmentation_s"] = round(time.time() - t, 1)
    step("حساب المؤشرات الحيوية")
    t = time.time()
    n_struct = len(seg._present)
    geometry_ok = not any(q["code"] == "anisotropic_inplane" for q in qc)

    contrast = B.contrast_state(study, seg)
    ms: list[Measurement] = []
    bone_ms, bone_info = B.bone_density(study, seg)
    ms += bone_ms
    ms += B.liver_spleen(study, seg, contrast)
    aorta_ms, aorta_info = B.aorta_geometry(study, seg) if geometry_ok else ([], {})
    ms += aorta_ms
    calc_ms, calc_overlay = B.calcium(study, seg, contrast, aorta_info.get("z_split"))
    ms += calc_ms
    ms += B.lungs(study, seg)
    l3 = bone_info["levels"].get("L3", {})
    bc_ms, bc_info = B.body_composition(study, seg, contrast, l3.get("centre") if l3.get("status") in ("ok", "nonphysiologic") else None)
    ms += bc_ms

    # A corrupted geometry invalidates everything that is a size; HU means survive.
    if any(q["code"] == "nonuniform_slice_spacing" for q in qc):
        for m in ms:
            if m.unit in ("mL", "L", "mm", "Agatston") and m.reliability != "invalid":
                m.reliability = "low"
                m.notes.append("nonuniform_slice_spacing")

    by_key = {m.key: m for m in ms}
    findings, coverage = rules.evaluate(by_key, study.meta, contrast, bool(calc_overlay.get("abstained")))
    timings["biomarkers_s"] = round(time.time() - t, 1)
    if n_struct < 5:
        qc.append({"code": "anatomy_not_recognised", "level": "error",
                   "text_ar": "لم يتعرّف النظام على بنى تشريحية كافية (قد لا يكون الفحص للجذع، أو الجودة غير كافية)."})
    step("توليد الصور المفتاحية")
    images = _images_ct(study, seg, bone_info, aorta_info, calc_overlay, bc_info)
    contrast_ar = {"unenhanced": "دون تباين وريدي", "enhanced": "بتباين وريدي", "uncertain": "غير مؤكد", "unknown": "غير محدد"}[contrast["state"]]
    return {
        "status": "ok", "kind": "ct", "meta": study.meta, "spacing_mm": [round(s, 3) for s in study.spacing],
        "shape": list(study.pixels.shape),
        "qc": {"items": qc, "contrast": {**contrast, "label_ar": contrast_ar}, "segmentation": seg.source, "structures_found": n_struct},
        "measurements": [m.to_dict() for m in ms], "findings": [f.to_dict() for f in findings],
        "coverage": coverage, "images": images, "bone_levels": bone_info.get("levels", {}), "timings": timings,
    }


def analyze(path: str | Path | Study, patient: dict | None = None, declared: str | None = None,
            seg: Segmentation | None = None, progress=None) -> dict:
    """Analyse any supported input. `patient` may supply age/sex/height_m/weight_kg."""
    t0 = time.time()
    study = path if isinstance(path, Study) else load_any(path)
    for k, v in (patient or {}).items():
        if v not in (None, ""):
            study.meta[k] = v  # values confirmed by the operator override DICOM header values
    if study.kind == "ct":
        res = analyze_ct(study, seg, progress)
    elif study.kind == "xray" or (study.kind == "image2d" and declared == "chest_xray"):
        from . import cxr

        res = cxr.analyze(study, progress)
    else:
        res = _unsupported(study)
    res.update(version=__version__, created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               scope_ar=rules.SCOPE_AR, scope_en=rules.SCOPE_EN,
               evidence={k: v for k, v in rules.EVIDENCE.items()
                         if any(k in f["evidence"] for f in res.get("findings", []))})
    res.setdefault("timings", {})["total_s"] = round(time.time() - t0, 1)
    return res


def _unsupported(study: Study) -> dict:
    modality = study.meta.get("modality", "UNKNOWN")
    images = []
    try:
        a = study.pixels
        sl = a[:, :, a.shape[2] // 2].T[::-1, ::-1] if a.ndim == 3 else a
        lo, hi = np.percentile(sl, [1, 99])
        g = (np.clip((sl - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        images.append({"key": "preview", "title_ar": "معاينة", "png": _b64(R.compose(g, []))})
    except Exception:
        pass
    msg = (f"تمت قراءة الفحص (النوع: {modality}) والتحقق من سلامته، لكن لا توجد في هذا الإصدار خوارزمية كمية "
           "مُتحقَّق منها علميًا لهذا النوع. لم يُجرَ أي تحليل تنبؤي، ولا يُستنتج من ذلك سلامة الفحص.")
    return {"status": "unsupported_modality", "kind": study.kind, "meta": study.meta, "message_ar": msg,
            "qc": {"items": _qc(study)}, "measurements": [], "findings": [], "coverage": [], "images": images}
