"""Command line: `python -m wiqaya analyze <path>` | `serve` | `selftest`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wiqaya", description="Opportunistic preventive screening from medical imaging.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyse one study (DICOM folder/zip, NIfTI, image)")
    a.add_argument("path")
    a.add_argument("--out", default="wiqaya_out", help="output directory (report.html + result.json)")
    a.add_argument("--sex", choices=["M", "F"])
    a.add_argument("--age", type=float)
    a.add_argument("--height-cm", type=float)
    a.add_argument("--weight-kg", type=float)
    a.add_argument("--chest-xray", action="store_true", help="declare a plain image as a frontal chest radiograph")
    s = sub.add_parser("serve", help="start the hospital web service")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    sub.add_parser("selftest", help="verify every biomarker against the digital reference phantom")
    args = ap.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("wiqaya.server:app", host=args.host, port=args.port, workers=1)
        return 0
    if args.cmd == "selftest":
        from .selftest import run

        return run()

    from . import report
    from .io import IngestError
    from .pipeline import analyze

    patient = {"sex": args.sex, "age": args.age, "height_m": args.height_cm / 100 if args.height_cm else None,
               "weight_kg": args.weight_kg}
    try:
        res = analyze(args.path, patient=patient, declared="chest_xray" if args.chest_xray else None,
                      progress=lambda m: print("…", m, file=sys.stderr))
    except IngestError as e:
        print(f"تعذّرت قراءة الفحص: {e}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"id": "cli", "label": Path(args.path).name, "status": "done", "result": res, "review": None}
    (out / "report.html").write_text(report.render(rec), encoding="utf-8")
    slim = {**res, "images": [{k: v for k, v in im.items() if k != "png"} for im in res.get("images", [])]}
    (out / "result.json").write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    for f in res.get("findings", []):
        print(f"[{f['severity']}] {f['title_ar']} — {f['detail_ar']}")
    for c in res.get("coverage", []):
        if not c["assessed"]:
            print(f"[—] {c['domain_ar']}: لم يُقيَّم ({c['reason_ar']})")
    print(f"التقرير: {out / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
