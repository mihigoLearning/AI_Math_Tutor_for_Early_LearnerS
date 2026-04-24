"""Tests for src/store.py"""

import json
import sqlite3
from pathlib import Path

import pytest
from cryptography.fernet import InvalidToken

from store import EncryptedStore, derive_key, dp_payload


def test_key_derivation_is_deterministic():
    salt = b"0123456789abcdef"
    a = derive_key("passphrase", salt)
    b = derive_key("passphrase", salt)
    assert a == b
    # different passphrase → different key
    assert derive_key("other", salt) != a


def test_encrypted_store_roundtrip(tmp_path):
    db = tmp_path / "t.db"
    with EncryptedStore(db, "secret") as s:
        s.save_snapshot({"learner_id": "k1", "p_L": {"addition": 0.3}, "n_events": 2})
    with EncryptedStore(db, "secret") as s:
        snap = s.latest_snapshot("k1")
    assert snap["p_L"]["addition"] == 0.3
    assert snap["n_events"] == 2


def test_encrypted_store_is_actually_encrypted(tmp_path):
    db = tmp_path / "t.db"
    with EncryptedStore(db, "secret") as s:
        s.save_snapshot({"learner_id": "k1", "p_L": {"addition": 0.3}})
    # Read raw
    raw = sqlite3.connect(db).execute("SELECT blob FROM snapshots").fetchone()[0]
    assert b"addition" not in raw
    assert b"0.3" not in raw


def test_wrong_passphrase_fails(tmp_path):
    db = tmp_path / "t.db"
    with EncryptedStore(db, "correct") as s:
        s.save_snapshot({"learner_id": "k1", "p_L": {"a": 0.5}})
    with EncryptedStore(db, "wrong") as s:
        with pytest.raises(InvalidToken):
            s.latest_snapshot("k1")


def test_event_logging_and_readback(tmp_path):
    db = tmp_path / "t.db"
    with EncryptedStore(db, "pw") as s:
        s.log_event("k1", {"skill": "counting", "correct": True})
        s.log_event("k1", {"skill": "addition", "correct": False})
    with EncryptedStore(db, "pw") as s:
        evs = s.events_for("k1")
    assert len(evs) == 2
    assert evs[0]["skill"] == "counting"
    assert evs[1]["correct"] is False


# ── ε-DP payload ────────────────────────────────────────────────────────
def test_dp_payload_clips_to_unit_interval():
    snap = {"learner_id": "k", "p_L": {"a": 0.5, "b": 0.3}, "n_events": 1}
    pay = dp_payload(snap, epsilon=0.1, seed=0)  # heavy noise → likely needs clipping
    for v in pay["skills"].values():
        assert 0.0 <= v <= 1.0


def test_dp_payload_includes_learner_and_epsilon():
    snap = {"learner_id": "k", "p_L": {"a": 0.5}}
    pay = dp_payload(snap, epsilon=1.0, seed=0)
    assert pay["learner_id"] == "k"
    assert pay["epsilon"] == 1.0


def test_dp_payload_tighter_epsilon_has_less_noise():
    snap = {"learner_id": "k",
            "p_L": {"counting": 0.5, "addition": 0.5, "subtraction": 0.5,
                    "number_sense": 0.5, "word_problem": 0.5}}
    # Run many trials to average out RNG
    import statistics as stat
    def noise_mag(eps, n=200):
        out = []
        for i in range(n):
            p = dp_payload(snap, epsilon=eps, seed=i)
            out.append(abs(p["skills"]["addition"] - 0.5))
        return stat.mean(out)
    assert noise_mag(10.0) < noise_mag(1.0), "larger ε should reduce noise"


def test_dp_payload_zero_epsilon_raises():
    with pytest.raises(ValueError):
        dp_payload({"p_L": {}}, epsilon=0.0)
