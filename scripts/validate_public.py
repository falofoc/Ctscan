"""Agreement study on the public TotalSegmentator dataset (CC BY 4.0, Zenodo 10047263).

For every case the biomarkers are computed twice: once from the expert-reviewed reference
segmentation shipped with the dataset and once from the model's own segmentation. The
difference isolates the error contributed by automatic segmentation.

    python scripts/validate_public.py <dataset_dir> --n 30 --out docs/validation
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wiqaya import io, segment  # noqa: E402
from wiqaya.labels import NAME_TO_ID  # noqa: E402
from wiqaya.selftest import measure  # noqa: E402
from wiqaya.types import Segmentation  # noqa: E402

DICE_STRUCTS = ["liver", "spleen", "aorta", "heart", "vertebrae_L1", "vertebrae_L3"]


def reference_segmentation(case: Path, shape) -> Segmentation:
    lab = np.zeros(shape, dtype=np.uint8)
    for name, idx in NAME_TO_ID.items():
        f = case / "segmentations" / f"{name}.nii.gz"
        if f.exists():
            m = np.asarray(nib.as_closest_canonical(nib.load(str(f))).dataobj) > 0
            lab[m] = idx
    return Segmentation(lab, NAME_TO_ID, source="reference")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="docs/validation")
    args = ap.parse_args()
    root, out = Path(args.dataset), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {r["image_id"]: r for r in csv.DictReader(open(root / "meta.csv", encoding="utf-8-sig"), delimiter=";")}
    cases = sorted(p.name for p in root.iterdir() if (p / "ct.nii.gz").exists())
    # held-out cases first (the model never saw them), then training cases
    cases = [c for c in cases if meta[c]["split"] != "train"] + [c for c in cases if meta[c]["split"] == "train"]
    rows = []
    for case in cases[: args.n]:
        t = time.time()
        study = io.load_any(root / case / "ct.nii.gz")
        study.meta["kvp"] = float(meta[case]["kvp"]) if meta[case]["kvp"] else None
        ref = reference_segmentation(root / case, study.pixels.shape)
        cache = root / case / "pred_fast.npy"
        if cache.exists():
            pred = segment.from_label_volume(np.load(cache), "totalsegmentator_fast")
        else:
            pred = segment.segment_ct(study, fast=True)
            np.save(cache, pred.labels)
        a, b = measure(study, ref), measure(study, pred)
        dice = {}
        for s in DICE_STRUCTS:
            x, y = ref.mask(s), pred.mask(s)
            if x.sum() > 500:
                dice[s] = round(float(2 * (x & y).sum() / (x.sum() + y.sum())), 4)
        for key in sorted(set(a) | set(b)):
            ma, mb = a.get(key), b.get(key)
            rows.append({"case": case, "split": meta[case]["split"], "key": key,
                         "reference": None if ma is None else round(float(ma.value), 3),
                         "automatic": None if mb is None else round(float(mb.value), 3),
                         "rel_reference": ma.reliability if ma else "", "rel_automatic": mb.reliability if mb else ""})
        rows.append({"case": case, "split": meta[case]["split"], "key": "_dice", "reference": json.dumps(dice)})
        print(case, meta[case]["split"], f"{time.time() - t:.0f}s", dice, flush=True)
        with open(out / "agreement_raw.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["case", "split", "key", "reference", "automatic", "rel_reference", "rel_automatic"])
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
