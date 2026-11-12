"""Summarise agreement_raw.csv → docs/validation/SUMMARY.md (bias, MAE, 95% limits of agreement)."""
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/validation")
rows = list(csv.DictReader(open(out / "agreement_raw.csv")))
cases = sorted({r["case"] for r in rows})
held = sorted({r["case"] for r in rows if r["split"] != "train"})
pairs, missed, dice = defaultdict(list), defaultdict(int), defaultdict(list)
for r in rows:
    if r["key"] == "_dice":
        for k, v in json.loads(r["reference"]).items():
            dice[k].append(v)
    elif r["reference"] and r["automatic"]:
        pairs[r["key"]].append((float(r["reference"]), float(r["automatic"])))
    elif r["reference"] or r["automatic"]:
        missed[r["key"]] += 1
lines = ["# Agreement: automatic vs reference segmentation", "",
         f"Cases: {len(cases)} public CT scans (TotalSegmentator dataset, CC BY 4.0), of which {len(held)} are held-out "
         f"({', '.join(held)}); the rest were part of the segmentation model's training set, so their agreement is optimistic.",
         "Segmentation: TotalSegmentator 3 mm fast model on CPU. Same biomarker code on both segmentations.", "",
         "| Measurement | n | mean diff (auto − ref) | MAE | 95% limits of agreement | produced by only one side |", "|---|---|---|---|---|---|"]
for k in sorted(pairs):
    d = [b - a for a, b in pairs[k]]
    sd = st.stdev(d) if len(d) > 1 else 0.0
    m = st.mean(d)
    lines.append(f"| `{k}` | {len(d)} | {m:+.2f} | {st.mean(abs(x) for x in d):.2f} | {m - 1.96 * sd:+.2f} … {m + 1.96 * sd:+.2f} | {missed.get(k, 0)} |")
lines += ["", "| Structure | n | median Dice | min Dice |", "|---|---|---|---|"]
for k in sorted(dice):
    lines.append(f"| {k} | {len(dice[k])} | {st.median(dice[k]):.3f} | {min(dice[k]):.3f} |")
lines += ["", "This measures robustness of the numbers to segmentation error only. It is **not** a clinical accuracy study: "
          "there is no DXA, biopsy, gated calcium score or outcome reference here."]
(out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
