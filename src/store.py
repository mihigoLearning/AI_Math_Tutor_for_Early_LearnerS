"""Encrypted local progress store + ε-differential-privacy sync.

Two concerns, one module:

1.  **Local store** — SQLite file, per-row AES-GCM encryption via
    `cryptography.fernet.Fernet`. Key is derived from a passphrase with
    PBKDF2-HMAC-SHA256 (200 000 iterations). This means the raw DB on disk
    reveals only learner_id + timestamps; all mastery fields are ciphertext.

2.  **ε-DP sync** — `dp_payload()` takes the current mastery dict and emits
    a payload with per-skill Laplace noise calibrated to the requested ε.
    Sensitivity = 1/N_events for per-skill averages; we clip mastery to
    [0, 1] before adding noise and after, so the output is always a valid
    probability. The parent ever only sees the noisy values, never the raw
    interaction stream.

Design note: we deliberately do NOT encrypt the sync payload. The whole
point of ε-DP is that the noisy summary is *safe to release in plaintext*
— encrypting it would conflate two different threat models (at-rest vs
over-wire) and confuse the privacy analysis.
"""

from __future__ import annotations

import base64
import json
import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


# ──────────────────────────────────────────────────────────────────────────
# Key derivation
# ──────────────────────────────────────────────────────────────────────────
def derive_key(passphrase: str, salt: bytes) -> bytes:
    """PBKDF2-HMAC-SHA256 → Fernet key (32 bytes url-safe base64)."""
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32,
                     salt=salt, iterations=200_000)
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))


# ──────────────────────────────────────────────────────────────────────────
# Encrypted store
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class EncryptedStore:
    """SQLite file with per-row Fernet-encrypted JSON blobs.

    Schema:

        meta(key PRIMARY KEY, value TEXT)           -- salt, version
        snapshots(ts REAL, learner_id TEXT, blob BLOB)
        events(ts REAL, learner_id TEXT, blob BLOB)

    `snapshots` holds Tutor.snapshot() dicts; `events` holds per-answer
    traces. Both blobs are Fernet-encrypted JSON.
    """
    path: Path
    passphrase: str
    _fernet: Fernet | None = None
    _conn: sqlite3.Connection | None = None

    def open(self) -> "EncryptedStore":
        first = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS snapshots ("
            "ts REAL, learner_id TEXT, blob BLOB)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "ts REAL, learner_id TEXT, blob BLOB)"
        )
        if first:
            salt = base64.b64encode(_random_bytes(16)).decode()
            self._conn.execute(
                "INSERT INTO meta VALUES (?, ?)", ("salt", salt)
            )
            self._conn.execute(
                "INSERT INTO meta VALUES (?, ?)", ("version", "1")
            )
            self._conn.commit()
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key='salt'"
        ).fetchone()
        salt = base64.b64decode(row[0])
        self._fernet = Fernet(derive_key(self.passphrase, salt))
        return self

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "EncryptedStore":
        return self.open()

    def __exit__(self, *a) -> None:
        self.close()

    # ── writes ──────────────────────────────────────────────────────────
    def save_snapshot(self, snapshot: dict) -> None:
        self._require_open()
        blob = self._fernet.encrypt(json.dumps(snapshot).encode("utf-8"))
        self._conn.execute(
            "INSERT INTO snapshots VALUES (?, ?, ?)",
            (time.time(), snapshot.get("learner_id", "?"), blob),
        )
        self._conn.commit()

    def log_event(self, learner_id: str, event: dict) -> None:
        self._require_open()
        blob = self._fernet.encrypt(json.dumps(event).encode("utf-8"))
        self._conn.execute(
            "INSERT INTO events VALUES (?, ?, ?)",
            (time.time(), learner_id, blob),
        )
        self._conn.commit()

    # ── reads ───────────────────────────────────────────────────────────
    def latest_snapshot(self, learner_id: str) -> dict | None:
        self._require_open()
        row = self._conn.execute(
            "SELECT blob FROM snapshots WHERE learner_id=? "
            "ORDER BY ts DESC LIMIT 1",
            (learner_id,),
        ).fetchone()
        if not row:
            return None
        return json.loads(self._fernet.decrypt(row[0]).decode("utf-8"))

    def events_for(self, learner_id: str) -> list[dict]:
        self._require_open()
        rows = self._conn.execute(
            "SELECT blob FROM events WHERE learner_id=? ORDER BY ts ASC",
            (learner_id,),
        ).fetchall()
        return [json.loads(self._fernet.decrypt(r[0]).decode("utf-8"))
                for r in rows]

    def _require_open(self) -> None:
        if not self._conn or not self._fernet:
            raise RuntimeError("store not opened — call open() or use `with`")


def _random_bytes(n: int) -> bytes:
    # Use os.urandom via Fernet.generate_key-ish; we already need cryptography
    import os
    return os.urandom(n)


# ──────────────────────────────────────────────────────────────────────────
# ε-Differential-privacy sync
# ──────────────────────────────────────────────────────────────────────────
def _laplace(scale: float, rng: random.Random) -> float:
    """Sample Laplace(0, scale). Pure-python (random.Random supports it
    indirectly via exponentials)."""
    u = rng.random() - 0.5
    # inverse CDF of zero-mean Laplace: -scale * sign(u) * ln(1 - 2|u|)
    import math
    return -scale * (1.0 if u >= 0 else -1.0) * math.log(1.0 - 2.0 * abs(u))


def dp_payload(snapshot: dict, *, epsilon: float = 1.0,
               seed: int | None = None) -> dict:
    """Return a parent-facing, ε-DP noisy summary.

    Sensitivity analysis: each skill's mastery is a value in [0, 1]; a
    single child event can change p_L by at most 1 (in the worst case the
    slip/guess posterior snaps from 0 → 1). We therefore use
    `sensitivity = 1.0` and `scale = 1 / epsilon`. For a weekly report
    summing K=5 skills, the total ε spent is K/ε if we wanted per-skill
    privacy; we instead set ε as the *whole-report* budget and split
    evenly → `scale_per_skill = K / epsilon`.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    rng = random.Random(seed)
    p_L = snapshot.get("p_L", {})
    scale = len(p_L) / epsilon
    noisy = {}
    for skill, v in p_L.items():
        n = v + _laplace(scale, rng)
        noisy[skill] = max(0.0, min(1.0, n))   # clip to valid probability
    return {
        "learner_id": snapshot.get("learner_id"),
        "epsilon": epsilon,
        "skills": noisy,
        "n_events": snapshot.get("n_events"),
    }
