import sqlite3
import numpy as np
import json
import os

DB_PATH = os.getenv("FEATURE_STORE_PATH", "store/features.db")


def _ensure_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            label INTEGER,
            vector TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            label TEXT,
            confidence REAL,
            model_used TEXT,
            source TEXT,
            flags TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_features(filename: str, label: int, vector: np.ndarray):
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO features (filename, label, vector) VALUES (?, ?, ?)",
                 (filename, label, json.dumps(vector.tolist())))
    conn.commit()
    conn.close()


def load_all_features():
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT label, vector FROM features").fetchall()
    conn.close()
    if not rows:
        return np.array([]), np.array([])
    X = np.array([json.loads(r[1]) for r in rows], dtype=np.float32)
    y = np.array([r[0] for r in rows], dtype=np.int32)
    return X, y


def save_prediction(filename: str, label: str, confidence: float,
                    model_used: str, source: str, flags: list):
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (filename, label, confidence, model_used, source, flags) VALUES (?,?,?,?,?,?)",
        (filename, label, confidence, model_used, source, json.dumps(flags))
    )
    conn.commit()
    conn.close()


def get_history(limit: int = 50):
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT filename, label, confidence, model_used, source, flags, created_at "
        "FROM predictions ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [
        {"filename": r[0], "label": r[1], "confidence": r[2],
         "model_used": r[3], "source": r[4], "flags": json.loads(r[5]),
         "created_at": r[6]}
        for r in rows
    ]


def get_stats():
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    real = conn.execute("SELECT COUNT(*) FROM features WHERE label=0").fetchone()[0]
    fake = conn.execute("SELECT COUNT(*) FROM features WHERE label=1").fetchone()[0]
    conn.close()
    return {"total_samples": total, "real_samples": real, "fake_samples": fake}
