import importlib
import time

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch, ph):
    study, seg, _ = ph
    monkeypatch.setenv("WIQAYA_DATA", str(tmp_path / "data"))
    monkeypatch.delenv("WIQAYA_TOKEN", raising=False)
    import wiqaya.segment as segment
    import wiqaya.server as server

    server = importlib.reload(server)
    monkeypatch.setattr(segment, "segment_ct", lambda s, *a, **k: seg)  # the model itself is tested separately
    nib.save(nib.Nifti1Image(study.pixels, study.affine), str(tmp_path / "ct.nii.gz"))
    with TestClient(server.app) as c:
        c.nifti = tmp_path / "ct.nii.gz"
        c.server = server
        yield c


def wait(client, sid, timeout=60):
    t = time.time()
    while time.time() - t < timeout:
        rec = client.get(f"/api/studies/{sid}").json()
        if rec["status"] in ("done", "failed"):
            return rec
        time.sleep(0.2)
    raise AssertionError("analysis did not finish")


def upload(client, **data):
    with open(client.nifti, "rb") as f:
        return client.post("/api/studies", files={"files": ("ct.nii.gz", f)}, data=data)


def test_full_flow_upload_report_review(client):
    r = upload(client, sex="M", age="64", height_cm="175")
    assert r.status_code == 200
    sid = r.json()["id"]
    rec = wait(client, sid)
    assert rec["status"] == "done", rec.get("error")
    res = rec["result"]
    found = {f["id"] for f in res["findings"]}
    assert {"bone.low", "liver.steatosis_modsev", "aorta.aaa", "cardiac.calcium"} <= found
    assert res["meta"]["height_m"] == 1.75 and len(res["images"]) >= 5
    assert not list((client.server.DATA / "uploads").glob("*"))  # raw upload removed after analysis

    page = client.get(f"/studies/{sid}")
    assert page.status_code == 200 and "مسودة آلية" in page.text and "أم دم في الأبهر البطني" in page.text

    flagged = {f["id"]: "confirmed" for f in res["findings"] if f["severity"] > 0}
    partial = dict(list(flagged.items())[:1])
    assert client.post(f"/api/studies/{sid}/review", json={"reviewer": "د. اختبار", "decisions": partial}).status_code == 422
    assert client.post(f"/api/studies/{sid}/review", json={"reviewer": "د. اختبار", "decisions": {"nope": "confirmed"}}).status_code == 422
    assert client.post(f"/api/studies/{sid}/review", json={"reviewer": "د. اختبار", "decisions": flagged}).status_code == 200
    assert client.post(f"/api/studies/{sid}/review", json={"reviewer": "آخر", "decisions": flagged}).status_code == 409  # immutable
    assert "معتمد من: د. اختبار" in client.get(f"/studies/{sid}").text
    actions = [a["action"] for a in client.get(f"/api/studies/{sid}").json()["audit"]]
    assert actions[:2] == ["created", "analysed"] and "reviewed" in actions

    listing = client.get("/api/studies").json()
    assert listing[0]["id"] == sid and listing[0]["max_severity"] == 2 and listing[0]["reviewed"] == 1
    assert client.delete(f"/api/studies/{sid}").status_code == 200
    assert client.get(f"/api/studies/{sid}").status_code == 404


def test_bad_upload_fails_gracefully_and_worker_survives(client):
    r = client.post("/api/studies", files={"files": ("x.dcm", b"garbage")})
    rec = wait(client, r.json()["id"])
    assert rec["status"] == "failed" and "تعذّرت قراءة الفحص" in rec["error"]
    assert wait(client, upload(client).json()["id"])["status"] == "done"  # next study still processed


def test_input_validation(client):
    assert upload(client, height_cm="17").status_code == 422
    assert upload(client, sex="X").status_code == 422


def test_token_protects_api(tmp_path, monkeypatch):
    monkeypatch.setenv("WIQAYA_DATA", str(tmp_path / "d2"))
    monkeypatch.setenv("WIQAYA_TOKEN", "s3cret")
    import wiqaya.server as server

    server = importlib.reload(server)
    with TestClient(server.app) as c:
        assert c.get("/api/studies").status_code == 401
        assert c.get("/api/studies", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.get("/api/studies", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/api/health").json()["auth_required"] is True
