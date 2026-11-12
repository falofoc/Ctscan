# Scientific basis, methods and limits

Wiqaya implements *opportunistic screening*: quantitative biomarkers read from imaging
performed for another indication. The strongest outcome evidence is Pickhardt et al.
(Lancet Digit Health 2020, n = 9223, median follow-up 8.8 y): automated CT aortic
calcification, muscle density, visceral/subcutaneous fat ratio, liver density and vertebral
density each predicted death and cardiovascular events, aortic calcification better than
the Framingham score (5-year AUROC 0.743 vs 0.688).

Wiqaya reports measurements and guideline thresholds. It does **not** output an individual
probability of future disease: no published, externally validated model with usable
coefficients exists for that, and inventing one would be unscientific.

## Methods

| Biomarker | Method | Abstains / downgrades when |
|---|---|---|
| Contrast phase | Median HU in aorta, heart, IVC, kidney cores; >80 HU enhanced, 60–80 uncertain | no blood pool in view → unknown |
| Vertebral trabecular HU | Max of 3D distance transform of the vertebra mask locates the body centre; ROI = cylinder 0.6·R in-plane, ±0.3·R axial (R = inscribed radius). L1 preferred, else T12, L2, L3, T11, L4 | vertebra truncated or collapsed; mean >350 HU (hardware); kVp ≠ 120; ROI standard error >8 HU; outlier vs neighbours |
| Liver / spleen HU | Median of voxels ≥8 mm (liver) / ≥5 mm (spleen) inside the mask | contrast enhanced or uncertain → invalid |
| Aortic diameter | Per axial slice, largest inscribed circle of each mask component (equals true diameter for oblique cuts), 6 mm rolling median, max per segment; segments split at T12 and by anterior/posterior position | anisotropic pixels; always "confirm manually" |
| Calcium (Agatston 1990) | ≥130 HU, lesions ≥1 mm² and ≥3 px, density weights 1–4, thin slices averaged to ≈3 mm slabs, scaled by slab/3 mm; bone and metal (≥1800 HU) excluded | contrast; noise SD >35 HU after averaging; heart truncated |
| Emphysema index | % lung voxels < −950 HU, plus Perc15 | lung volume <1 L in view; sharp kernel or noise SD >40 HU → low |
| L3 body composition | 5 mm slab, noise-adaptive smoothing, HU windows (muscle −29…150, SAT −190…−30, VAT −150…−50); abdominal wall traced in polar coordinates to separate compartments | L3 not in view; body outside FOV (SAT invalid); wall found in <55% of angles |
| Cardiothoracic ratio | Heart width / widest lung-field width on PSPNet (ChestX-Det) masks | lungs/heart not found; not PA → low |

## Thresholds and sources

| Rule | Source |
|---|---|
| L1 ≤110 HU: >90% specific for DXA osteoporosis; ≤160 HU: 90% sensitive | Pickhardt 2013, doi:10.7326/0003-4819-158-8-201304160-00003 |
| L1 normative values, contrast effect negligible ≥40 y | Jang 2019, doi:10.1148/radiol.2019181648 |
| Liver ≤40 HU: moderate–severe steatosis | Boyce 2010, doi:10.2214/AJR.09.2590 |
| Liver <48 HU: 100% specific (53.8% sensitive) for ≥30% biopsy steatosis | Pickhardt 2012, doi:10.1007/s00330-011-2349-2 |
| AAA ≥3.0 cm; surveillance by size | Chaikof 2018 (SVS), doi:10.1016/j.jvs.2017.10.044 |
| Thoracic aorta ≥4.0 cm dilated, ≥5.0 cm aneurysm range | Isselbacher 2022 (ACC/AHA), doi:10.1161/CIR.0000000000001106 |
| Calcium scoring method | Agatston 1990, doi:10.1016/0735-1097(90)90282-t |
| Report CAC on non-gated chest CT; categories 0 / 1–99 / 100–399 / ≥400 | Hecht 2017 (SCCT/STR), doi:10.1097/RTI.0000000000000287 |
| Quantitative emphysema | Lynch 2015 (Fleischner), doi:10.1148/radiol.2015141579 |
| SMI <52.4 (men), <38.5 (women) cm²/m² | Prado 2008, doi:10.1016/S1470-2045(08)70153-0 |
| Muscle attenuation and index as prognostic factors | Martin 2013, doi:10.1200/JCO.2012.45.2722 |
| Segmentation model | Wasserthal 2023, doi:10.1148/ryai.230024 |

Citations were checked against PubMed records. The 5% LAA-950 cut-off and the 25–29 mm
"ectasia" band are common conventions rather than single-study thresholds; they yield
"attention" level findings only.

## Known limits

- Automated L1 ROIs read higher than manual ones by about 21 HU on average (Jang 2019),
  so the bone thresholds, derived mostly from manual ROIs, are slightly conservative here.
- The cardiac calcium score covers the whole heart region on a non-gated scan: it includes
  valvular and annular calcium and is an estimate, not a coronary Agatston score.
- Aortic diameter is measured on the segmentation mask; mural thrombus on enhanced scans
  may be excluded by the mask and the diameter underestimated.
- SMI cut-offs come from oncology cohorts; no population-specific reference values for
  Arab populations are built in.
- LAA-950 cannot detect that a scan was acquired in expiration.
- Vertebral level labels come from the segmentation model; transitional anatomy can shift them.
- The CPU default uses the 3 mm segmentation model; boundaries are coarser than on GPU.
- Not validated in children, pregnancy, or after major surgery/implants.
