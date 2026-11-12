# Agreement: automatic vs reference segmentation

Cases: 5 public CT scans (TotalSegmentator dataset, CC BY 4.0), of which 4 are held-out (s0747, s1233, s1287, s1337); the rest were part of the segmentation model's training set, so their agreement is optimistic.
Segmentation: TotalSegmentator 3 mm fast model on CPU. Same biomarker code on both segmentations.

| Measurement | n | mean diff (auto − ref) | MAE | 95% limits of agreement | produced by only one side |
|---|---|---|---|---|---|
| `aorta.abdominal_calcium_agatston` | 1 | +101.25 | 101.25 | +101.25 … +101.25 | 0 |
| `aorta.abdominal_max_diameter_mm` | 5 | -0.56 | 0.56 | -1.38 … +0.26 | 0 |
| `aorta.arch_max_diameter_mm` | 4 | +0.79 | 2.13 | -5.88 … +7.46 | 1 |
| `aorta.ascending_max_diameter_mm` | 5 | -1.02 | 1.02 | -1.89 … -0.14 | 0 |
| `aorta.descending_max_diameter_mm` | 5 | -0.27 | 0.68 | -1.86 … +1.32 | 0 |
| `body.l3_muscle_area_cm2` | 3 | -0.53 | 1.82 | -4.73 … +3.66 | 0 |
| `body.l3_muscle_hu` | 3 | +0.23 | 0.32 | -0.61 … +1.07 | 0 |
| `body.l3_sat_cm2` | 3 | -0.98 | 2.66 | -8.53 … +6.57 | 0 |
| `body.l3_vat_cm2` | 3 | -0.22 | 0.64 | -2.10 … +1.66 | 0 |
| `body.vat_sat_ratio` | 3 | +0.01 | 0.01 | -0.03 … +0.05 | 0 |
| `bone.trabecular_hu` | 5 | +8.42 | 8.42 | -4.79 … +21.63 | 0 |
| `heart.calcium_agatston` | 1 | +108.00 | 108.00 | +108.00 … +108.00 | 0 |
| `liver.attenuation_hu` | 5 | +0.00 | 0.00 | +0.00 … +0.00 | 0 |
| `liver.minus_spleen_hu` | 5 | -0.20 | 0.20 | -1.08 … +0.68 | 0 |
| `liver.volume_ml` | 5 | -11.22 | 12.41 | -33.27 … +10.84 | 0 |
| `lung.laa950_percent` | 5 | -0.02 | 0.03 | -0.10 … +0.06 | 0 |
| `lung.volume_l` | 5 | -0.04 | 0.04 | -0.11 … +0.04 | 0 |
| `spleen.attenuation_hu` | 5 | +0.20 | 0.20 | -0.68 … +1.08 | 0 |
| `spleen.volume_ml` | 5 | -1.10 | 1.32 | -3.53 … +1.33 | 0 |

| Structure | n | median Dice | min Dice |
|---|---|---|---|
| aorta | 5 | 0.940 | 0.917 |
| heart | 5 | 0.956 | 0.937 |
| liver | 5 | 0.972 | 0.961 |
| spleen | 5 | 0.946 | 0.928 |
| vertebrae_L1 | 5 | 0.901 | 0.885 |
| vertebrae_L3 | 3 | 0.894 | 0.874 |

This measures robustness of the numbers to segmentation error only. It is **not** a clinical accuracy study: there is no DXA, biopsy, gated calcium score or outcome reference here.
