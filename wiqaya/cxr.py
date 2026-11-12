"""Frontal chest radiograph analyser (EXPERIMENTAL tier).

Primary output is geometric and inspectable: the cardiothoracic ratio (CTR) measured on
a lung/heart segmentation that is drawn on the key image. The multi-label classifier
scores are shown only as unvalidated triage hints and never yield a "normal" statement.
Models: TorchXRayVision (Cohen et al., MIDL 2022) — research weights, not a medical device.
"""
from __future__ import annotations

import base64
import contextlib
import io as _io

import numpy as np

from . import render as R
from .rules import CAVEATS_AR, DOMAINS_AR, EVIDENCE
from .types import Finding, Measurement, Study

CTR_THRESHOLD = 0.50
HINT_THRESHOLD = 0.70
HINT_AR = {"Cardiomegaly": "تضخم القلب", "Effusion": "انصباب جنبي", "Nodule": "عقيدة", "Mass": "كتلة",
           "Emphysema": "نفاخ", "Fibrosis": "تليّف", "Consolidation": "تكثّف", "Pneumothorax": "استرواح صدر",
           "Edema": "وذمة", "Atelectasis": "انخماص", "Lung Opacity": "عتامة رئوية", "Pneumonia": "التهاب رئوي",
           "Pleural_Thickening": "تسمّك جنبي", "Fracture": "كسر", "Lung Lesion": "آفة رئوية",
           "Enlarged Cardiomediastinum": "اتساع المنصف", "Infiltration": "ارتشاح", "Hernia": "فتق"}

EVIDENCE.setdefault("danzer1919", {"cite": "Danzer CS. The cardiothoracic ratio: an index of cardiac enlargement. Am J Med Sci 1919;157:513-21.",
                                   "doi": "10.1097/00000441-191904000-00007", "pmid": None})
EVIDENCE.setdefault("cohen2022", {"cite": "Cohen JP, et al. TorchXRayVision: a library of chest X-ray datasets and models. Proc MIDL 2022, PMLR 172:231-49.",
                                  "doi": "10.48550/arXiv.2111.00595", "pmid": None})
CAVEATS_AR.update({
    "view_not_pa": "وضعية التصوير ليست PA قائمة (أو غير معروفة)؛ عتبة 0.50 للنسبة القلبية الصدرية صالحة فقط لصورة PA بشهيق كافٍ.",
    "experimental_model": "نموذج بحثي غير معتمد سريريًا؛ النتيجة تلميح للفرز فقط ويجب التحقق منها بصريًا.",
})

_models = {}


def _load():
    if not _models:
        import torch
        import torchxrayvision as xrv

        with contextlib.redirect_stdout(_io.StringIO()):
            _models["seg"] = xrv.baseline_models.chestx_det.PSPNet().eval()
            _models["cls"] = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
        _models["torch"] = torch
    return _models


def _prepare(img: np.ndarray, size: int) -> np.ndarray:
    from skimage.transform import resize

    lo, hi = np.percentile(img, [0.5, 99.5])
    x = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    h, w = x.shape
    s = min(h, w)
    x = x[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    x = resize(x, (size, size), order=1, anti_aliasing=True, preserve_range=True)
    return (x * 2048.0 - 1024.0).astype(np.float32)


def cardiothoracic_ratio(heart: np.ndarray, lungs: np.ndarray) -> dict | None:
    """CTR = widest transverse cardiac diameter / widest transverse thoracic (lung-field) diameter."""
    if heart.sum() < 0.01 * heart.size or lungs.sum() < 0.06 * lungs.size:
        return None
    hc = np.flatnonzero(heart.any(axis=0))
    widths = np.array([(np.flatnonzero(r)[-1] - np.flatnonzero(r)[0] + 1) if r.any() else 0 for r in lungs])
    row = int(np.argmax(widths))
    lc = np.flatnonzero(lungs[row])
    return {"ctr": float((hc[-1] - hc[0] + 1) / widths[row]), "heart_cols": (int(hc[0]), int(hc[-1])),
            "thorax_row": row, "thorax_cols": (int(lc[0]), int(lc[-1]))}


def analyze(study: Study, progress=None) -> dict:
    m = _load()
    torch = m["torch"]
    view = (study.meta.get("view_position") or "").upper()
    qc = []
    if view in ("LL", "LATERAL", "RL", "RLD", "LLD"):
        return {"status": "unsupported_modality", "kind": "xray", "meta": study.meta, "measurements": [], "findings": [],
                "coverage": [], "images": [], "qc": {"items": []},
                "message_ar": "الصورة جانبية؛ التحليل متاح للصورة الأمامية فقط."}
    if progress:
        progress("تحليل أشعة الصدر")
    x512 = _prepare(study.pixels, 512)
    with torch.no_grad():
        seg = torch.sigmoid(m["seg"](torch.from_numpy(x512)[None, None]))[0].numpy() > 0.5
        scores = m["cls"](torch.from_numpy(_prepare(study.pixels, 224))[None, None])[0].numpy()
    t = m["seg"].targets
    heart = seg[t.index("Heart")]
    lungs = seg[t.index("Left Lung")] | seg[t.index("Right Lung")]
    both = seg[t.index("Left Lung")].sum() > 0.02 * heart.size and seg[t.index("Right Lung")].sum() > 0.02 * heart.size
    geo = cardiothoracic_ratio(heart, lungs) if both else None

    ms, findings, images = [], [], []
    coverage = [{"domain": "cxr", "domain_ar": DOMAINS_AR["cxr"], "assessed": geo is not None,
                 "reason_ar": "" if geo else "لم يُتعرَّف على رئتين وقلب بشكل موثوق (قد لا تكون الصورة أشعة صدر أمامية أو الجودة غير كافية)"}]
    g = R.window(x512, 0, 2048)
    if geo:
        notes = ["experimental_model"] + ([] if view == "PA" else ["view_not_pa"])
        rel = "moderate" if view == "PA" else "low"
        ms.append(Measurement("cxr.cardiothoracic_ratio", geo["ctr"], "", rel, notes))
        cav = [CAVEATS_AR[n] for n in notes]
        if geo["ctr"] > CTR_THRESHOLD:
            findings.append(Finding("cxr.ctr_high", "cxr", 1, "ارتفاع النسبة القلبية الصدرية — تضخم قلب محتمل",
                "Raised cardiothoracic ratio — possible cardiomegaly",
                f"النسبة القلبية الصدرية ≈ {geo['ctr']:.2f} (العتبة >{CTR_THRESHOLD:.2f} على صورة PA قائمة).",
                f"CTR ≈ {geo['ctr']:.2f} (>{CTR_THRESHOLD:.2f} on an erect PA film).",
                "تأكيد القياس يدويًا والتحقق من وضعية التصوير، ثم تقييم سريري (ضغط الدم، أعراض قصور القلب) والنظر في تخطيط صدى القلب.",
                "Confirm manually and check projection; clinical assessment, consider echocardiography.",
                ["danzer1919", "cohen2022"], rel, cav, ["cxr.cardiothoracic_ratio"]))
        else:
            findings.append(Finding("cxr.ctr_normal", "cxr", 0, "النسبة القلبية الصدرية ضمن الحد الطبيعي", "Cardiothoracic ratio within normal limit",
                f"النسبة القلبية الصدرية ≈ {geo['ctr']:.2f}.", f"CTR ≈ {geo['ctr']:.2f}.", "", "",
                ["danzer1919", "cohen2022"], rel, cav, ["cxr.cardiothoracic_ratio"]))
        lines = np.zeros_like(heart)
        hr = int(np.median(np.nonzero(heart)[0]))
        lines[hr - 1:hr + 2, geo["heart_cols"][0]:geo["heart_cols"][1] + 1] = True
        tl = np.zeros_like(heart)
        tl[geo["thorax_row"] - 1:geo["thorax_row"] + 2, geo["thorax_cols"][0]:geo["thorax_cols"][1] + 1] = True
        images.append({"key": "cxr", "title_ar": f"النسبة القلبية الصدرية ≈ {geo['ctr']:.2f}: القلب (أحمر)، الرئتان (سماوي)",
                       "png": base64.b64encode(R.compose(g, [(lungs, "cyan", "edge"), (heart, "red", "edge"), (lines, "red", "fill"), (tl, "cyan", "fill")])).decode()})
        hints = sorted(((float(s), p) for s, p in zip(scores, m["cls"].pathologies) if p and s >= HINT_THRESHOLD), reverse=True)
        if hints:
            txt = "، ".join(f"{HINT_AR.get(p, p)} ({s:.2f})" for s, p in hints[:6])
            findings.append(Finding("cxr.hints", "cxr", 1, "تلميحات آلية تستدعي نظرة الأخصائي (تجريبي)", "Automated triage hints (experimental)",
                f"درجات المصنّف المرتفعة: {txt}. الدرجات ليست احتمالات معايَرة.", "Classifier scores above threshold: " + ", ".join(f"{p} ({s:.2f})" for s, p in hints[:6]),
                "مراجعة الصورة بصريًا للتحقق من هذه العلامات.", "Visual review required.", ["cohen2022"], "low",
                [CAVEATS_AR["experimental_model"]], []))
    else:
        images.append({"key": "cxr", "title_ar": "معاينة", "png": base64.b64encode(R.compose(g, [])).decode()})
    findings.sort(key=lambda f: -f.severity)
    from .pipeline import _qc

    return {"status": "ok", "kind": "xray", "meta": study.meta, "qc": {"items": _qc(study) + qc},
            "measurements": [x.to_dict() for x in ms], "findings": [f.to_dict() for f in findings],
            "coverage": coverage, "images": images}
