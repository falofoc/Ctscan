# Agreement: automatic vs reference segmentation

Cases: 16 public CT scans (TotalSegmentator dataset, CC BY 4.0), of which 4 are held-out (s0747, s1233, s1287, s1337); the rest were part of the segmentation model's training set, so their agreement is optimistic.
Segmentation: TotalSegmentator 3 mm fast model on CPU. Same biomarker code on both segmentations.

| Measurement | n | mean diff (auto − ref) | MAE | 95% limits of agreement | produced by only one side |
|---|---|---|---|---|---|
| `aorta.abdominal_calcium_agatston` | 2 | -60.75 | 162.00 | -509.79 … +388.29 | 0 |
| `aorta.abdominal_max_diameter_mm` | 15 | -0.87 | 1.02 | -2.53 … +0.79 | 0 |
| `aorta.arch_max_diameter_mm` | 12 | -0.28 | 1.36 | -4.27 … +3.71 | 1 |
| `aorta.ascending_max_diameter_mm` | 14 | -1.33 | 1.33 | -2.23 … -0.43 | 0 |
| `aorta.descending_max_diameter_mm` | 15 | -0.91 | 1.13 | -2.80 … +0.98 | 0 |
| `body.l3_muscle_area_cm2` | 7 | -1.33 | 1.89 | -4.49 … +1.82 | 0 |
| `body.l3_muscle_hu` | 7 | +0.18 | 0.24 | -0.37 … +0.73 | 0 |
| `body.l3_sat_cm2` | 7 | +4.50 | 6.85 | -23.16 … +32.17 | 0 |
| `body.l3_vat_cm2` | 7 | -4.11 | 5.84 | -29.73 … +21.50 | 0 |
| `body.vat_sat_ratio` | 5 | +0.02 | 0.02 | -0.04 … +0.09 | 0 |
| `bone.trabecular_hu` | 15 | +5.64 | 9.19 | -16.66 … +27.94 | 0 |
| `heart.calcium_agatston` | 2 | +356.62 | 356.62 | -332.53 … +1045.78 | 0 |
| `liver.attenuation_hu` | 15 | -0.13 | 0.13 | -0.82 … +0.56 | 0 |
| `liver.minus_spleen_hu` | 15 | -0.13 | 0.27 | -1.39 … +1.12 | 0 |
| `liver.volume_ml` | 15 | -14.00 | 17.33 | -54.32 … +26.31 | 0 |
| `lung.laa950_percent` | 15 | -0.04 | 0.04 | -0.20 … +0.13 | 0 |
| `lung.volume_l` | 15 | -0.03 | 0.04 | -0.11 … +0.06 | 0 |
| `spleen.attenuation_hu` | 15 | +0.00 | 0.27 | -1.05 … +1.05 | 0 |
| `spleen.volume_ml` | 15 | -2.28 | 4.39 | -14.50 … +9.94 | 0 |

| Structure | n | median Dice | min Dice |
|---|---|---|---|
| aorta | 15 | 0.929 | 0.912 |
| heart | 15 | 0.947 | 0.922 |
| liver | 15 | 0.971 | 0.961 |
| spleen | 15 | 0.953 | 0.902 |
| vertebrae_L1 | 15 | 0.895 | 0.698 |
| vertebrae_L3 | 7 | 0.884 | 0.874 |

This measures robustness of the numbers to segmentation error only. It is **not** a clinical accuracy study: there is no DXA, biopsy, gated calcium score or outcome reference here.
