"""Evidence-linked interpretation rules: measurements → preventive findings.

Every threshold below is taken from a published, citable source (see EVIDENCE) and is
applied only under the acquisition conditions it was derived for. When those conditions
are not met the rule abstains ("not assessed") instead of guessing — an abstention is
reported explicitly so that it is never mistaken for a normal result.
"""
from __future__ import annotations

from .types import Finding, Measurement

EVIDENCE = {
    "pickhardt2013": {
        "cite": "Pickhardt PJ, et al. Opportunistic screening for osteoporosis using abdominal CT scans obtained for other indications. Ann Intern Med 2013;158:588-95.",
        "doi": "10.7326/0003-4819-158-8-201304160-00003", "pmid": "23588747"},
    "jang2019": {
        "cite": "Jang S, et al. Opportunistic osteoporosis screening at routine abdominal and thoracic CT: normative L1 trabecular attenuation values in more than 20 000 adults. Radiology 2019;291:360-7.",
        "doi": "10.1148/radiol.2019181648", "pmid": "30912719"},
    "boyce2010": {
        "cite": "Boyce CJ, et al. Hepatic steatosis (fatty liver disease) in asymptomatic adults identified by unenhanced low-dose CT. AJR 2010;194:623-8.",
        "doi": "10.2214/AJR.09.2590", "pmid": "20173137"},
    "pickhardt2012": {
        "cite": "Pickhardt PJ, et al. Specificity of unenhanced CT for non-invasive diagnosis of hepatic steatosis. Eur Radiol 2012;22:1075-82.",
        "doi": "10.1007/s00330-011-2349-2", "pmid": "22138733"},
    "prado2008": {
        "cite": "Prado CMM, et al. Prevalence and clinical implications of sarcopenic obesity in patients with solid tumours of the respiratory and gastrointestinal tracts. Lancet Oncol 2008;9:629-35.",
        "doi": "10.1016/S1470-2045(08)70153-0", "pmid": "18539529"},
    "martin2013": {
        "cite": "Martin L, et al. Cancer cachexia in the age of obesity: skeletal muscle depletion is a powerful prognostic factor, independent of body mass index. J Clin Oncol 2013;31:1539-47.",
        "doi": "10.1200/JCO.2012.45.2722", "pmid": "23530101"},
    "pickhardt2020": {
        "cite": "Pickhardt PJ, et al. Automated CT biomarkers for opportunistic prediction of future cardiovascular events and mortality in an asymptomatic screening population. Lancet Digit Health 2020;2:e192-200.",
        "doi": "10.1016/S2589-7500(20)30025-X", "pmid": "32864598"},
    "agatston1990": {
        "cite": "Agatston AS, et al. Quantification of coronary artery calcium using ultrafast computed tomography. J Am Coll Cardiol 1990;15:827-32.",
        "doi": "10.1016/0735-1097(90)90282-t", "pmid": "2407762"},
    "hecht2017": {
        "cite": "Hecht HS, et al. 2016 SCCT/STR guidelines for coronary artery calcium scoring of noncontrast noncardiac chest CT scans. J Thorac Imaging 2017;32:W54-66.",
        "doi": "10.1097/RTI.0000000000000287", "pmid": "28832417"},
    "chaikof2018": {
        "cite": "Chaikof EL, et al. The Society for Vascular Surgery practice guidelines on the care of patients with an abdominal aortic aneurysm. J Vasc Surg 2018;67:2-77.",
        "doi": "10.1016/j.jvs.2017.10.044", "pmid": "29268916"},
    "isselbacher2022": {
        "cite": "Isselbacher EM, et al. 2022 ACC/AHA guideline for the diagnosis and management of aortic disease. Circulation 2022;146:e334-482.",
        "doi": "10.1161/CIR.0000000000001106", "pmid": None},
    "lynch2015": {
        "cite": "Lynch DA, et al. CT-definable subtypes of chronic obstructive pulmonary disease: a statement of the Fleischner Society. Radiology 2015;277:192-205.",
        "doi": "10.1148/radiol.2015141579", "pmid": "25961632"},
    "wasserthal2023": {
        "cite": "Wasserthal J, et al. TotalSegmentator: robust segmentation of 104 anatomic structures in CT images. Radiol Artif Intell 2023;5:e230024.",
        "doi": "10.1148/ryai.230024", "pmid": None},
}

# Thresholds (single source of truth; mirrored in docs/SCIENCE.md).
BONE_SPECIFIC_HU = 110.0   # ≤110 HU: >90% specific for osteoporosis vs DXA (Pickhardt 2013)
BONE_SENSITIVE_HU = 160.0  # ≤160 HU: 90% sensitive (Pickhardt 2013)
LIVER_MODSEV_HU = 40.0     # ≤40 HU: moderate-to-severe steatosis (Boyce 2010)
LIVER_SPECIFIC_HU = 48.0   # <48 HU: 100% specific for ≥30% biopsy steatosis (Pickhardt 2012)
AAA_MM = 30.0              # ≥3.0 cm abdominal aortic aneurysm (Chaikof 2018)
AAA_ECTASIA_MM = 25.0
THORACIC_DILATED_MM = 40.0  # Isselbacher 2022
THORACIC_ANEURYSM_MM = 50.0
LAA950_PERCENT = 5.0
SMI_PRADO = {"M": 52.4, "F": 38.5}  # cm²/m² (Prado 2008)

CAVEATS_AR = {
    "level_substituted": "قيست فقرة بديلة لعدم توفر L1 بشكل صالح؛ الأداء التشخيصي متقارب بين المستويات T12–L5.",
    "level_outlier": "قيمة الفقرة المقيسة تختلف كثيرًا عن الفقرات المجاورة — يُحتمل وجود آفة بؤرية أو كسر؛ تُراجع بصريًا.",
    "kvp_not_120": "جهد الأنبوب ليس 120 kVp؛ عتبات HU للعظم مشتقة عند 120 kVp وقد لا تنطبق بدقة.",
    "kvp_unknown": "جهد الأنبوب غير معروف (لا توجد بيانات اكتساب)؛ العتبات مشتقة عند 120 kVp.",
    "contrast_enhanced": "الفحص بعد حقن مادة تباين وريدية.",
    "contrast_uncertain": "تعذّر الجزم بغياب مادة التباين.",
    "contrast_unknown": "تعذّر تحديد حالة التباين.",
    "organ_truncated": "العضو غير مشمول كاملًا في مجال الفحص.",
    "segmentation_fast_model": "استُخدم نموذج التقطيع السريع (3 مم)؛ الحدود تقريبية.",
    "thick_slices": "سماكة المقاطع كبيرة؛ قد تُفوَّت آفات صغيرة ويقل ضبط القياس.",
    "high_noise": "ضجيج الصورة مرتفع (جرعة منخفضة/مقاطع رقيقة)؛ دقة هذا القياس منخفضة.",
    "non_gated": "فحص غير مُبوَّب بتخطيط القلب؛ القيمة تقديرية وليست بديلًا عن فحص كالسيوم تاجي مخصص.",
    "includes_valvular": "المنطقة المقيسة تشمل كامل القلب: قد تتضمن تكلسات صمامية/حلقية غير تاجية.",
    "metal_excluded": "استُبعدت بنى بكثافة معدنية (دعامات/أسلاك/صمامات).",
    "lungs_truncated": "الرئتان غير مشمولتين كاملًا في مجال الفحص.",
    "sharp_kernel": "مرشح إعادة بناء حاد؛ يرفع نسبة المناطق منخفضة التوهين بشكل مصطنع.",
    "kernel_unknown": "مرشح إعادة البناء غير معروف.",
    "body_outside_fov": "جزء من الجسم خارج مجال الرؤية؛ مساحة الدهن تحت الجلد غير صالحة.",
    "abdominal_wall_not_closed": "تعذّر تحديد جدار البطن العضلي بشكل مغلق؛ فصل الدهن الحشوي تقريبي.",
    "manual_confirmation": "قياس آلي من قناع التقطيع؛ يجب تأكيده بقياس يدوي عمودي على محور الوعاء.",
    "young_adult": "العمر أقل من 50 سنة؛ العتبات مشتقة أساسًا من بالغين أكبر سنًا.",
    "oncology_cutoffs": "حدود القطع مشتقة من مرضى أورام وقد لا تنطبق على عموم السكان.",
}

DOMAINS_AR = {"bone": "العظام", "liver": "الكبد", "aorta": "الأبهر", "cardiac": "القلب والشرايين التاجية",
              "lung": "الرئتان", "body": "تركيب الجسم", "cxr": "أشعة الصدر"}


def _caveats(m: Measurement, extra: list[str] = ()) -> list[str]:
    return [CAVEATS_AR[c] for c in list(m.notes) + list(extra) if c in CAVEATS_AR]


def _cap(sev: int, m: Measurement) -> int:
    """A low-reliability measurement can prompt a look, never an alarm."""
    return min(sev, 1) if m.reliability == "low" else sev


def evaluate(ms: dict[str, Measurement], meta: dict, contrast: dict, calcium_abstained: bool = False) -> tuple[list[Finding], list[dict]]:
    out: list[Finding] = []
    coverage: list[dict] = []

    def cover(domain, assessed, reason_ar=""):
        coverage.append({"domain": domain, "domain_ar": DOMAINS_AR[domain], "assessed": assessed, "reason_ar": reason_ar})

    age, sex = meta.get("age"), meta.get("sex")
    enhanced = contrast.get("state") != "unenhanced"
    contrast_reason = {"enhanced": "الفحص بمادة تباين وريدية", "uncertain": "تعذّر الجزم بغياب مادة التباين",
                       "unknown": "تعذّر تحديد حالة التباين"}.get(contrast.get("state"), "")

    # ------------------------------------------------------------------ bone
    m = ms.get("bone.trabecular_hu")
    if m and m.reliability != "invalid":
        v, lv = m.value, m.extra.get("level", "L1")
        extra = []
        if enhanced:
            extra.append("contrast_" + contrast.get("state", "unknown"))
        if age is not None and age < 50:
            extra.append("young_adult")
        cav = _caveats(m, extra)
        base = dict(domain="bone", evidence=["pickhardt2013", "jang2019"], reliability=m.reliability,
                    caveats_ar=cav, measurement_keys=[m.key])
        if v <= BONE_SPECIFIC_HU:
            out.append(Finding("bone.low", severity=_cap(2, m),
                title_ar="كثافة عظمية منخفضة — هشاشة العظام مرجّحة", title_en="Low vertebral bone density — osteoporosis likely",
                detail_ar=f"متوسط توهين العظم الإسفنجي في الفقرة {lv} = {v:.0f} HU (العتبة ≤{BONE_SPECIFIC_HU:.0f} HU ذات نوعية تفوق 90% لهشاشة العظام مقارنةً بـ DXA).",
                detail_en=f"{lv} trabecular attenuation {v:.0f} HU (≤{BONE_SPECIFIC_HU:.0f} HU is >90% specific for DXA-defined osteoporosis).",
                recommendation_ar="يُنصح بقياس كثافة العظام بـ DXA، وتقدير خطر الكسور (FRAX)، ومراجعة عوامل الخطر (فيتامين د، الكالسيوم، الستيرويدات، التدخين) والوقاية من السقوط.",
                recommendation_en="Recommend DXA, fracture-risk assessment (FRAX) and review of modifiable risk factors.", **base))
        elif v <= BONE_SENSITIVE_HU:
            out.append(Finding("bone.indeterminate", severity=1,
                title_ar="كثافة عظمية حدّية — لا يمكن استبعاد هشاشة العظام", title_en="Borderline vertebral bone density — osteoporosis not excluded",
                detail_ar=f"متوسط توهين العظم الإسفنجي في الفقرة {lv} = {v:.0f} HU (بين {BONE_SPECIFIC_HU:.0f} و{BONE_SENSITIVE_HU:.0f} HU: منطقة غير حاسمة).",
                detail_en=f"{lv} trabecular attenuation {v:.0f} HU (indeterminate range {BONE_SPECIFIC_HU:.0f}–{BONE_SENSITIVE_HU:.0f} HU).",
                recommendation_ar="يُنظر في إجراء DXA بحسب العمر والجنس وعوامل الخطر السريرية.",
                recommendation_en="Consider DXA according to age, sex and clinical risk factors.", **base))
        else:
            out.append(Finding("bone.normal", severity=0,
                title_ar="كثافة عظمية مطمئنة", title_en="Vertebral bone density reassuring",
                detail_ar=f"متوسط توهين العظم الإسفنجي في الفقرة {lv} = {v:.0f} HU (أعلى من {BONE_SENSITIVE_HU:.0f} HU؛ العتبة ذات حساسية 90%، فهشاشة العظام غير مرجّحة).",
                detail_en=f"{lv} trabecular attenuation {v:.0f} HU (>{BONE_SENSITIVE_HU:.0f} HU; osteoporosis unlikely).", **base))
        cover("bone", True)
    else:
        cover("bone", False, "لا توجد فقرة قطنية/صدرية سفلية كاملة وصالحة للقياس ضمن مجال الفحص")

    # ------------------------------------------------------------------ liver
    m = ms.get("liver.attenuation_hu")
    if m and m.reliability != "invalid":
        v = m.value
        base = dict(domain="liver", evidence=["boyce2010", "pickhardt2012"], reliability=m.reliability,
                    caveats_ar=_caveats(m), measurement_keys=[m.key, "liver.minus_spleen_hu"])
        rec_ar = ("تقييم عوامل الخطر الاستقلابية (السكري، دهون الدم، السمنة، الضغط) وتعاطي الكحول والأدوية، "
                  "وفحص إنزيمات الكبد، وحساب مؤشر FIB-4 لتقدير خطر التليّف وتحديد الحاجة للإحالة.")
        rec_en = "Assess metabolic risk factors and alcohol intake, check liver enzymes, and stratify fibrosis risk (FIB-4)."
        if v <= LIVER_MODSEV_HU:
            out.append(Finding("liver.steatosis_modsev", severity=_cap(2, m),
                title_ar="تشحّم كبدي متوسط إلى شديد", title_en="Moderate-to-severe hepatic steatosis",
                detail_ar=f"توهين الكبد = {v:.0f} HU على فحص دون تباين (≤{LIVER_MODSEV_HU:.0f} HU).",
                detail_en=f"Unenhanced liver attenuation {v:.0f} HU (≤{LIVER_MODSEV_HU:.0f} HU).",
                recommendation_ar=rec_ar, recommendation_en=rec_en, **base))
        elif v < LIVER_SPECIFIC_HU:
            out.append(Finding("liver.steatosis", severity=1,
                title_ar="تشحّم كبدي مرجّح", title_en="Hepatic steatosis likely",
                detail_ar=f"توهين الكبد = {v:.0f} HU على فحص دون تباين (أقل من {LIVER_SPECIFIC_HU:.0f} HU؛ عتبة عالية النوعية للتشحّم المتوسط فأكثر في دراسة مرجعية بالخزعة).",
                detail_en=f"Unenhanced liver attenuation {v:.0f} HU (<{LIVER_SPECIFIC_HU:.0f} HU, a highly specific threshold in a biopsy-referenced cohort).",
                recommendation_ar=rec_ar, recommendation_en=rec_en, **base))
        else:
            out.append(Finding("liver.normal", severity=0,
                title_ar="لا دليل على تشحّم كبدي متوسط أو شديد", title_en="No CT evidence of moderate/severe steatosis",
                detail_ar=f"توهين الكبد = {v:.0f} HU على فحص دون تباين. (التصوير المقطعي غير حساس للتشحّم الخفيف.)",
                detail_en=f"Unenhanced liver attenuation {v:.0f} HU. (CT is insensitive to mild steatosis.)", **base))
        cover("liver", True)
    elif m:
        cover("liver", False, f"{contrast_reason}: عتبات توهين الكبد صالحة فقط دون تباين")
    else:
        cover("liver", False, "الكبد غير مشمول بشكل كافٍ في مجال الفحص")

    # ------------------------------------------------------------------ aorta diameter
    aorta_seen = False
    m = ms.get("aorta.abdominal_max_diameter_mm")
    if m:
        aorta_seen = True
        d = m.value
        base = dict(domain="aorta", evidence=["chaikof2018"], reliability=m.reliability,
                    caveats_ar=_caveats(m, ["manual_confirmation"]), measurement_keys=[m.key])
        if d >= AAA_MM:
            if d >= 55:
                sev, plan = 3, "إحالة عاجلة إلى جراحة الأوعية لتقييم الإصلاح."
            elif d >= 50:
                sev, plan = 3, "إحالة إلى جراحة الأوعية؛ مراقبة تصويرية كل 6 أشهر."
            elif d >= 40:
                sev, plan = 2, "مراقبة تصويرية كل 12 شهرًا وإحالة إلى جراحة الأوعية."
            else:
                sev, plan = 2, "مراقبة تصويرية كل 3 سنوات."
            out.append(Finding("aorta.aaa", severity=_cap(sev, m),
                title_ar="أم دم في الأبهر البطني", title_en="Abdominal aortic aneurysm",
                detail_ar=f"أقصى قطر للأبهر البطني ≈ {d:.0f} مم (≥{AAA_MM:.0f} مم).",
                detail_en=f"Maximum abdominal aortic diameter ≈ {d:.0f} mm (≥{AAA_MM:.0f} mm).",
                recommendation_ar=plan + " ضبط عوامل الخطر: الإقلاع عن التدخين، ضغط الدم، الدهون.",
                recommendation_en="Confirm manually; surveillance/referral per SVS guideline; risk-factor control.", **base))
        elif d >= AAA_ECTASIA_MM:
            out.append(Finding("aorta.ectasia", severity=1,
                title_ar="توسّع خفيف في الأبهر البطني", title_en="Abdominal aortic ectasia",
                detail_ar=f"أقصى قطر للأبهر البطني ≈ {d:.0f} مم (25–29 مم).",
                detail_en=f"Maximum abdominal aortic diameter ≈ {d:.0f} mm (25–29 mm).",
                recommendation_ar="تأكيد القياس يدويًا؛ إعادة التصوير لاحقًا وفق الإرشادات وضبط عوامل الخطر الوعائية.",
                recommendation_en="Confirm manually; interval re-imaging per guideline.", **base))
        else:
            out.append(Finding("aorta.abdominal_normal", severity=0,
                title_ar="قطر الأبهر البطني ضمن الطبيعي", title_en="Abdominal aortic calibre normal",
                detail_ar=f"أقصى قطر ≈ {d:.0f} مم.", detail_en=f"Maximum diameter ≈ {d:.0f} mm.", **base))
    for part, ar, en in (("ascending", "الأبهر الصاعد", "ascending aorta"), ("arch", "قوس الأبهر", "aortic arch"),
                         ("descending", "الأبهر الصدري النازل", "descending thoracic aorta")):
        m = ms.get(f"aorta.{part}_max_diameter_mm")
        if not m:
            continue
        aorta_seen = True
        d = m.value
        base = dict(domain="aorta", evidence=["isselbacher2022"], reliability=m.reliability,
                    caveats_ar=_caveats(m, ["manual_confirmation"]), measurement_keys=[m.key])
        if d >= THORACIC_DILATED_MM:
            big = d >= THORACIC_ANEURYSM_MM
            out.append(Finding(f"aorta.{part}_dilated", severity=_cap(2 if big else 1, m),
                title_ar=f"{'أم دم' if big else 'توسّع'} في {ar}", title_en=f"{'Aneurysm' if big else 'Dilatation'} of the {en}",
                detail_ar=f"أقصى قطر ≈ {d:.0f} مم (≥{(THORACIC_ANEURYSM_MM if big else THORACIC_DILATED_MM):.0f} مم).",
                detail_en=f"Maximum diameter ≈ {d:.0f} mm.",
                recommendation_ar="تأكيد القياس يدويًا (ويُفضّل فحص مخصص للأبهر)، ضبط ضغط الدم، ومتابعة/إحالة إلى أمراض القلب أو جراحة القلب والأوعية حسب الإرشادات.",
                recommendation_en="Confirm with dedicated measurement; blood-pressure control; follow-up/referral per guideline.", **base))
    if aorta_seen:
        cover("aorta", True)
    else:
        cover("aorta", False, "الأبهر غير مشمول بشكل كافٍ في مجال الفحص")

    # ------------------------------------------------------------------ calcium
    m = ms.get("aorta.abdominal_calcium_agatston")
    if m and m.value is not None:
        if m.value > 0:
            out.append(Finding("aorta.calcification", severity=1, domain="aorta",
                title_ar="تكلّس في جدار الأبهر البطني (تصلّب عصيدي)", title_en="Abdominal aortic calcification (atherosclerosis)",
                detail_ar=f"درجة التكلّس بطريقة Agatston ≈ {m.value:.0f} (حجم ≈ {m.extra.get('volume_mm3', 0):.0f} مم³).",
                detail_en=f"Agatston-method score ≈ {m.value:.0f}.",
                recommendation_ar="التكلّس الأبهري مؤشر على تصلّب الشرايين ومرتبط بالأحداث القلبية الوعائية المستقبلية؛ يُنصح بتقييم رسمي للخطر القلبي الوعائي وضبط عوامل الخطر.",
                recommendation_en="Marker of atherosclerosis associated with future cardiovascular events; formal risk assessment advised.",
                evidence=["pickhardt2020", "agatston1990"], reliability=m.reliability, caveats_ar=_caveats(m), measurement_keys=[m.key]))
    m = ms.get("heart.calcium_agatston")
    if m and m.value is not None:
        s = m.value
        base = dict(domain="cardiac", evidence=["hecht2017", "agatston1990"], reliability=m.reliability,
                    caveats_ar=_caveats(m), measurement_keys=[m.key])
        rec_ar = ("تأكيد أن التكلّس تاجي المنشأ من قبل أخصائي الأشعة، ثم تقدير الخطر القلبي الوعائي الإجمالي (ASCVD) "
                  "ومناقشة العلاج الوقائي (الستاتين، ضبط الضغط والسكر، الإقلاع عن التدخين) وفق الإرشادات.")
        if s <= 0:
            out.append(Finding("cardiac.calcium_none", severity=0,
                title_ar="لا تكلّس قلبي/تاجي مرئي", title_en="No cardiac/coronary calcium detected",
                detail_ar="لم تُرصد آفات متكلّسة (≥130 HU) في منطقة القلب.", detail_en="No calcified lesions (≥130 HU) in the cardiac region.", **base))
        else:
            grade_ar, grade_en, sev = (("خفيف", "mild", 1) if s < 100 else ("متوسط", "moderate", 2) if s < 400 else ("شديد", "severe", 2))
            out.append(Finding("cardiac.calcium", severity=_cap(sev, m),
                title_ar=f"تكلّس قلبي/تاجي {grade_ar}", title_en=f"Cardiac/coronary calcification, {grade_en}",
                detail_ar=f"درجة تقديرية بطريقة Agatston ≈ {s:.0f} (الفئات: 1–99 خفيف، 100–399 متوسط، ≥400 شديد).",
                detail_en=f"Estimated Agatston-method score ≈ {s:.0f} (1–99 mild, 100–399 moderate, ≥400 severe).",
                recommendation_ar=rec_ar, recommendation_en="Radiologist to confirm coronary origin; global ASCVD risk assessment and preventive therapy per guideline.", **base))
        cover("cardiac", True)
    elif enhanced:
        cover("cardiac", False, f"{contrast_reason}: تقييم الكالسيوم يتطلب فحصًا دون تباين")
    elif calcium_abstained:
        cover("cardiac", False, "ضجيج الصورة مرتفع جدًا لتقييم الكالسيوم بعتبة 130 HU دون إيجابيات كاذبة")
    else:
        cover("cardiac", False, "القلب غير مشمول كاملًا في مجال الفحص")

    # ------------------------------------------------------------------ lungs
    m = ms.get("lung.laa950_percent")
    if m and m.reliability != "invalid":
        v = m.value
        base = dict(domain="lung", evidence=["lynch2015"], reliability=m.reliability, caveats_ar=_caveats(m), measurement_keys=[m.key])
        if v >= LAA950_PERCENT:
            out.append(Finding("lung.emphysema", severity=1,
                title_ar="مناطق رئوية منخفضة التوهين — نُفاخ رئوي محتمل", title_en="Low-attenuation lung areas — possible emphysema",
                detail_ar=f"نسبة حجم الرئة تحت −950 HU = {v:.1f}% (≥{LAA950_PERCENT:.0f}%). القياس صالح لفحص شهيقي كامل بمرشح ناعم.",
                detail_en=f"LAA-950 = {v:.1f}% (≥{LAA950_PERCENT:.0f}%). Valid for full-inspiration, smooth-kernel CT.",
                recommendation_ar="مراجعة بصرية لنمط النفاخ، ربط النتيجة بتاريخ التدخين والأعراض، وقياس وظائف الرئة (Spirometry)، والمساعدة على الإقلاع عن التدخين.",
                recommendation_en="Visual confirmation, smoking history, spirometry, smoking-cessation support.", **base))
        else:
            out.append(Finding("lung.normal", severity=0,
                title_ar="لا زيادة كمية في المناطق منخفضة التوهين", title_en="No quantitative excess of low-attenuation lung",
                detail_ar=f"نسبة حجم الرئة تحت −950 HU = {v:.1f}%.", detail_en=f"LAA-950 = {v:.1f}%.", **base))
        cover("lung", True)
    else:
        cover("lung", False, "الرئتان غير مشمولتين بشكل كافٍ في مجال الفحص")

    # ------------------------------------------------------------------ body composition
    m = ms.get("body.l3_muscle_area_cm2")
    if m:
        smi = ms.get("body.l3_smi_cm2_m2")
        vat, sat = ms.get("body.l3_vat_cm2"), ms.get("body.l3_sat_cm2")
        fat_txt = ""
        if vat:
            fat_txt = f" الدهن الحشوي ≈ {vat.value:.0f} سم²"
            if sat and sat.reliability != "invalid":
                fat_txt += f"، تحت الجلد ≈ {sat.value:.0f} سم²"
            fat_txt += "."
        base = dict(domain="body", evidence=["prado2008", "martin2013", "pickhardt2020"], reliability=m.reliability,
                    measurement_keys=[m.key, "body.l3_smi_cm2_m2", "body.l3_vat_cm2", "body.l3_sat_cm2", "body.l3_muscle_hu"])
        if smi and sex in SMI_PRADO:
            cut = SMI_PRADO[sex]
            low = smi.value < cut
            out.append(Finding("body.sarcopenia" if low else "body.muscle_normal", severity=1 if low else 0,
                title_ar="كتلة عضلية هيكلية منخفضة (ساركوبينيا محتملة)" if low else "مؤشر الكتلة العضلية ضمن الحد المرجعي",
                title_en="Low skeletal muscle index (possible sarcopenia)" if low else "Skeletal muscle index within reference",
                detail_ar=f"مؤشر العضلات الهيكلية عند L3 = {smi.value:.1f} سم²/م² (حد القطع {cut} للـ{'ذكور' if sex == 'M' else 'إناث'})." + fat_txt,
                detail_en=f"L3 skeletal muscle index {smi.value:.1f} cm²/m² (cut-off {cut}).",
                recommendation_ar="تقييم وظيفي (قوة القبضة، سرعة المشي)، تقييم تغذوي، وبرنامج تمارين مقاومة عند تأكيد التشخيص." if low else "",
                recommendation_en="Functional and nutritional assessment; resistance exercise if confirmed." if low else "",
                caveats_ar=_caveats(m, ["oncology_cutoffs"]), **base))
        else:
            missing = "الطول" if not smi else "الجنس"
            out.append(Finding("body.composition", severity=0,
                title_ar="قياسات تركيب الجسم عند مستوى L3", title_en="Body composition at L3",
                detail_ar=f"مساحة العضلات الهيكلية ≈ {m.value:.0f} سم²." + fat_txt + f" (أدخل {missing} لحساب مؤشر العضلات ومقارنته بحدود القطع.)",
                detail_en=f"Skeletal muscle area ≈ {m.value:.0f} cm². Height/sex needed for index-based classification.",
                caveats_ar=_caveats(m), **base))
        cover("body", True)
    else:
        cover("body", False, "مستوى الفقرة L3 غير مشمول في مجال الفحص")

    out.sort(key=lambda f: (-f.severity, f.domain))
    return out, coverage


SCOPE_AR = ("هذا النظام أداة دعم قرار للفحص الانتهازي الوقائي فقط. لا يكشف الأورام أو العقيدات الرئوية أو الكسور أو "
            "النزف أو الانصمام أو أي حالة حادة، ولا يغني عن قراءة أخصائي الأشعة. غياب نتيجة لا يعني السلامة.")
SCOPE_EN = ("Decision support for opportunistic preventive screening only. It does not detect tumours, lung nodules, "
            "fractures, haemorrhage, embolism or any acute condition and does not replace the radiologist's read.")
