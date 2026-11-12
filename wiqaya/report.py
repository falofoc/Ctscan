"""Render a result as a self-contained, printable Arabic (RTL) HTML report."""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import cxr  # noqa: F401  (registers its evidence entries)
from .rules import EVIDENCE

_env = Environment(loader=FileSystemLoader(str(Path(__file__).parent / "web")), autoescape=select_autoescape(["html"]))

SEVERITY = {0: ("طبيعي / معلومة", "s0"), 1: ("يستدعي الانتباه", "s1"), 2: ("نتيجة مهمة", "s2"), 3: ("مراجعة عاجلة", "s3")}
RELIABILITY_AR = {"high": "عالية", "moderate": "متوسطة", "low": "منخفضة", "invalid": "غير صالح"}
MEASURE_AR = {
    "bone.trabecular_hu": "توهين العظم الإسفنجي الفقري", "liver.attenuation_hu": "توهين الكبد",
    "liver.volume_ml": "حجم الكبد", "spleen.attenuation_hu": "توهين الطحال", "spleen.volume_ml": "حجم الطحال",
    "liver.minus_spleen_hu": "فرق توهين الكبد − الطحال", "aorta.abdominal_max_diameter_mm": "أقصى قطر للأبهر البطني",
    "aorta.ascending_max_diameter_mm": "أقصى قطر للأبهر الصاعد", "aorta.arch_max_diameter_mm": "أقصى قطر لقوس الأبهر",
    "aorta.descending_max_diameter_mm": "أقصى قطر للأبهر الصدري النازل", "aorta.unspecified_max_diameter_mm": "أقصى قطر للأبهر",
    "aorta.abdominal_calcium_agatston": "تكلّس الأبهر البطني", "heart.calcium_agatston": "تكلّس منطقة القلب (غير مبوَّب)",
    "lung.laa950_percent": "نسبة الرئة تحت −950 HU", "lung.volume_l": "حجم الرئتين",
    "body.l3_muscle_area_cm2": "مساحة العضلات الهيكلية عند L3", "body.l3_muscle_hu": "توهين العضلات عند L3",
    "body.l3_vat_cm2": "مساحة الدهن الحشوي عند L3", "body.l3_sat_cm2": "مساحة الدهن تحت الجلد عند L3",
    "body.l3_smi_cm2_m2": "مؤشر العضلات الهيكلية (SMI)", "body.vat_sat_ratio": "نسبة الدهن الحشوي / تحت الجلد",
    "cxr.cardiothoracic_ratio": "النسبة القلبية الصدرية (CTR)",
}


def render(record: dict, interactive: bool = False) -> str:
    res = record.get("result") or {}
    findings = res.get("findings", [])
    return _env.get_template("report.html").render(
        rec=record, res=res, findings=findings, review=record.get("review"),
        flagged=[f for f in findings if f["severity"] > 0], normal=[f for f in findings if f["severity"] == 0],
        SEVERITY=SEVERITY, REL=RELIABILITY_AR, MEASURE_AR=MEASURE_AR, EVIDENCE=EVIDENCE, interactive=interactive)
