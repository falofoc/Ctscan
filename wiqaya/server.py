"""Hospital-side web service: upload → queued analysis → reviewable report.

Runs entirely on the local network; nothing is sent outside the machine.
Environment: WIQAYA_DATA (storage dir), WIQAYA_TOKEN (optional shared API token),
WIQAYA_DEVICE=cpu|gpu, WIQAYA_FAST=0|1, WIQAYA_KEEP_UPLOADS=1.
"""
from __future__ import annotations

import hmac
import os
import queue
import shutil
import threading
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from . import __version__, report, segment
from .io import IngestError
from .pipeline import analyze
from .store import Store

DATA = Path(os.environ.get("WIQAYA_DATA", "runs")).resolve()
TOKEN = os.environ.get("WIQAYA_TOKEN") or None
MAX_UPLOAD = int(os.environ.get("WIQAYA_MAX_UPLOAD_MB", "4096")) * 1024 * 1024
WEB = Path(__file__).parent / "web"

store = Store(DATA / "wiqaya.sqlite")
jobs: "queue.Queue[str]" = queue.Queue()


def _process(sid: str) -> None:
    rec = store.get(sid)
    if rec is None:
        return
    updir = DATA / "uploads" / sid
    store.update(sid, status="processing", stage="قراءة الفحص")
    try:
        files = [p for p in updir.rglob("*") if p.is_file()] if updir.exists() else []
        if not files:
            raise IngestError("الملفات المرفوعة غير موجودة (ربما حُذفت بعد إعادة تشغيل)")
        target = files[0] if len(files) == 1 else updir
        res = analyze(target, patient=rec["patient"], declared=rec["declared"],
                      progress=lambda msg: store.update(sid, stage=msg))
        store.update(sid, status="done", stage="", result=res)
        store.audit(sid, "system", "analysed", f"kind={res.get('kind')} findings={len(res.get('findings', []))}")
    except IngestError as e:
        store.update(sid, status="failed", stage="", error=f"تعذّرت قراءة الفحص: {e}")
        store.audit(sid, "system", "failed", str(e))
    except Exception as e:  # never let one study kill the worker
        store.update(sid, status="failed", stage="", error=f"خطأ داخلي أثناء التحليل: {type(e).__name__}: {e}")
        store.audit(sid, "system", "failed", traceback.format_exc()[-2000:])
    finally:
        if os.environ.get("WIQAYA_KEEP_UPLOADS") != "1":
            shutil.rmtree(updir, ignore_errors=True)  # raw images with PHI are not retained


def _worker() -> None:
    while True:
        sid = jobs.get()
        try:
            _process(sid)
        finally:
            jobs.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    for sid in store.pending():  # resume work interrupted by a restart
        jobs.put(sid)
    threading.Thread(target=_worker, daemon=True, name="wiqaya-worker").start()
    yield


app = FastAPI(title="Wiqaya", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None)


def auth(request: Request) -> None:
    if TOKEN is None:
        return
    given = request.headers.get("authorization", "").removeprefix("Bearer ").strip() or request.query_params.get("token", "")
    if not hmac.compare_digest(given, TOKEN):
        raise HTTPException(401, "رمز الدخول غير صحيح")


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health():
    fast, device = segment.default_mode() if segment.available() else (None, None)
    return {"version": __version__, "segmentation_available": segment.available(), "device": device,
            "fast_model": fast, "queue": jobs.qsize(), "auth_required": TOKEN is not None}


def _num(v: str | None, lo: float, hi: float, name: str) -> float | None:
    if v in (None, ""):
        return None
    try:
        x = float(v)
    except ValueError:
        raise HTTPException(422, f"قيمة غير صالحة: {name}")
    if not lo <= x <= hi:
        raise HTTPException(422, f"قيمة خارج المدى المعقول: {name}")
    return x


@app.post("/api/studies", dependencies=[Depends(auth)])
async def create_study(files: list[UploadFile] = File(...), sex: str = Form(""), age: str = Form(""),
                       height_cm: str = Form(""), weight_kg: str = Form(""), declared: str = Form("")):
    if sex not in ("", "M", "F"):
        raise HTTPException(422, "قيمة غير صالحة: الجنس")
    h = _num(height_cm, 100, 230, "الطول")
    patient = {"sex": sex or None, "age": _num(age, 0, 120, "العمر"),
               "height_m": h / 100 if h else None, "weight_kg": _num(weight_kg, 20, 350, "الوزن")}
    label = files[0].filename if len(files) == 1 else f"{len(files)} ملفات"
    sid = store.create(Path(label or "study").name[:120], patient, declared if declared in ("chest_xray",) else None)
    updir = DATA / "uploads" / sid
    updir.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        for i, f in enumerate(files):
            suffix = "".join(Path(f.filename or "").suffixes[-2:])[:12]
            suffix = suffix if all(c.isalnum() or c == "." for c in suffix) else ""
            with open(updir / f"{i:06d}{suffix}", "wb") as out:  # client filenames never touch the filesystem
                while chunk := await f.read(1 << 20):
                    total += len(chunk)
                    if total > MAX_UPLOAD:
                        raise HTTPException(413, "حجم الرفع يتجاوز الحد المسموح")
                    out.write(chunk)
    except HTTPException:
        shutil.rmtree(updir, ignore_errors=True)
        store.delete(sid)
        raise
    jobs.put(sid)
    return {"id": sid}


@app.get("/api/studies", dependencies=[Depends(auth)])
def list_studies():
    return store.list()


def _get(sid: str) -> dict:
    rec = store.get(sid)
    if rec is None:
        raise HTTPException(404, "السجل غير موجود")
    return rec


@app.get("/api/studies/{sid}", dependencies=[Depends(auth)])
def get_study(sid: str, images: bool = True):
    rec = _get(sid)
    if not images and rec.get("result"):
        rec["result"]["images"] = [{k: v for k, v in im.items() if k != "png"} for im in rec["result"].get("images", [])]
    rec["audit"] = store.audit_for(sid)
    return rec


@app.get("/studies/{sid}", response_class=HTMLResponse, dependencies=[Depends(auth)])
def study_page(sid: str):
    rec = _get(sid)
    if rec["status"] != "done":
        raise HTTPException(409, rec.get("error") or "التحليل لم يكتمل بعد")
    store.audit(sid, "user", "viewed")
    return report.render(rec, interactive=True)


class Review(BaseModel):
    reviewer: str = Field(min_length=2, max_length=120)
    decisions: dict[str, str]
    comment: str = Field(default="", max_length=4000)


@app.post("/api/studies/{sid}/review", dependencies=[Depends(auth)])
def review_study(sid: str, body: Review):
    rec = _get(sid)
    if rec["status"] != "done":
        raise HTTPException(409, "التحليل لم يكتمل بعد")
    if rec.get("review"):
        raise HTTPException(409, "التقرير معتمد مسبقًا ولا يمكن تعديله")
    findings = {f["id"]: f for f in rec["result"].get("findings", [])}
    if any(k not in findings or v not in ("confirmed", "rejected", "unsure") for k, v in body.decisions.items()):
        raise HTTPException(422, "قرارات غير صالحة")
    missing = [k for k, f in findings.items() if f["severity"] > 0 and k not in body.decisions]
    if missing:
        raise HTTPException(422, "يجب تحديد قرار لكل نتيجة غير طبيعية")
    review = {"reviewer": body.reviewer.strip(), "decisions": body.decisions, "comment": body.comment.strip(),
              "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    store.update(sid, review=review)
    store.audit(sid, review["reviewer"], "reviewed", str(body.decisions))
    return {"ok": True}


@app.delete("/api/studies/{sid}", dependencies=[Depends(auth)])
def delete_study(sid: str):
    _get(sid)
    shutil.rmtree(DATA / "uploads" / sid, ignore_errors=True)
    store.delete(sid)
    return {"ok": True}
