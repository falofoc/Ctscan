"""Local persistence: study records and an append-only audit trail (SQLite, single file)."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS studies(
                    id TEXT PRIMARY KEY, created TEXT, updated TEXT, status TEXT, stage TEXT,
                    label TEXT, patient TEXT, declared TEXT, result TEXT, error TEXT, review TEXT);
                CREATE TABLE IF NOT EXISTS audit(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, study_id TEXT, actor TEXT, action TEXT, detail TEXT);
            """)

    def audit(self, study_id: str | None, actor: str, action: str, detail: str = "") -> None:
        with self._lock, self._db:
            self._db.execute("INSERT INTO audit(ts, study_id, actor, action, detail) VALUES(?,?,?,?,?)",
                             (_now(), study_id, actor, action, detail))

    def create(self, label: str, patient: dict, declared: str | None) -> str:
        sid = uuid.uuid4().hex[:12]
        with self._lock, self._db:
            self._db.execute("INSERT INTO studies(id, created, updated, status, stage, label, patient, declared) VALUES(?,?,?,?,?,?,?,?)",
                             (sid, _now(), _now(), "queued", "", label, json.dumps(patient), declared))
        self.audit(sid, "system", "created", label)
        return sid

    def update(self, sid: str, **fields) -> None:
        for k in ("result", "review"):
            if k in fields and not isinstance(fields[k], (str, type(None))):
                fields[k] = json.dumps(fields[k], ensure_ascii=False)
        fields["updated"] = _now()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock, self._db:
            self._db.execute(f"UPDATE studies SET {cols} WHERE id=?", (*fields.values(), sid))

    def get(self, sid: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM studies WHERE id=?", (sid,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        for k in ("patient", "result", "review"):
            d[k] = json.loads(d[k]) if d[k] else None
        return d

    def list(self, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, created, updated, status, stage, label, error, review IS NOT NULL AS reviewed, "
                "json_extract(result, '$.kind') AS kind, "
                "(SELECT MAX(json_extract(value, '$.severity')) FROM json_each(result, '$.findings')) AS max_severity "
                "FROM studies ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def pending(self) -> list[str]:
        with self._lock:
            return [r["id"] for r in self._db.execute(
                "SELECT id FROM studies WHERE status IN ('queued','processing') ORDER BY created")]

    def delete(self, sid: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM studies WHERE id=?", (sid,))
        self.audit(sid, "user", "deleted")

    def audit_for(self, sid: str) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute("SELECT ts, actor, action, detail FROM audit WHERE study_id=? ORDER BY id", (sid,))]
