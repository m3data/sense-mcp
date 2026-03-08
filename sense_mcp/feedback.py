"""Sense relevance feedback — learns to distinguish signal from noise.

Stores explicit feedback labels (useful/noise) for surfaced results and
computes per-file relevance weights that adjust search scoring over time.

Design:
  - Feedback lives in sense.db alongside chunks (longitudinal, survives reboots).
  - Weights are computed as a boost/penalty around 1.0 using a Bayesian prior
    so that a few early signals don't dominate.
  - The hook (read-only DB) can read weights; only the MCP server writes feedback.
"""

import sqlite3
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_feedback_table(conn: sqlite3.Connection) -> None:
    """Create the feedback table if it doesn't exist. Idempotent."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT NOT NULL,
            file_path TEXT NOT NULL,
            label TEXT NOT NULL CHECK(label IN ('useful', 'noise')),
            similarity REAL,
            mode TEXT,
            note TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_feedback_file ON feedback(file_path);
        CREATE INDEX IF NOT EXISTS idx_feedback_label ON feedback(label);
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def record_feedback(
    conn: sqlite3.Connection,
    query_text: str,
    file_path: str,
    label: str,
    similarity: float | None = None,
    mode: str | None = None,
    note: str | None = None,
) -> None:
    """Insert a feedback row."""
    if label not in ("useful", "noise"):
        raise ValueError(f"Invalid label: {label!r} (expected 'useful' or 'noise')")
    conn.execute(
        """INSERT INTO feedback
           (query_text, file_path, label, similarity, mode, note, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            query_text,
            file_path,
            label,
            similarity,
            mode,
            note,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Relevance weights
# ---------------------------------------------------------------------------

def load_relevance_weights(
    conn: sqlite3.Connection,
    boost_factor: float = 0.3,
    prior: float = 2.0,
) -> dict[str, float]:
    """Compute per-file relevance weights from accumulated feedback.

    Returns {file_path: weight} for files that have feedback.
    Files absent from the dict should be treated as weight=1.0.

    Formula: weight = 1.0 + boost_factor * (useful - noise) / (useful + noise + 2*prior)

    With defaults (boost=0.3, prior=2.0):
      - 3 useful, 0 noise  -> 1.0 + 0.3 * 3/7  = 1.13
      - 0 useful, 3 noise  -> 1.0 + 0.3 * -3/7  = 0.87
      - 5 useful, 0 noise  -> 1.0 + 0.3 * 5/9   = 1.17
      - 0 useful, 10 noise -> 1.0 + 0.3 * -10/14 = 0.79
    Conservative by design — first iteration.
    """
    rows = conn.execute("""
        SELECT file_path,
               SUM(CASE WHEN label = 'useful' THEN 1 ELSE 0 END) as useful,
               SUM(CASE WHEN label = 'noise' THEN 1 ELSE 0 END) as noise
        FROM feedback
        GROUP BY file_path
    """).fetchall()

    weights = {}
    for file_path, useful, noise in rows:
        total = useful + noise + 2 * prior
        weights[file_path] = 1.0 + boost_factor * (useful - noise) / total
    return weights


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def get_feedback_stats(conn: sqlite3.Connection) -> dict:
    """Aggregate feedback statistics for the stats tool."""
    total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

    by_label = {}
    for row in conn.execute(
        "SELECT label, COUNT(*) FROM feedback GROUP BY label"
    ).fetchall():
        by_label[row[0]] = row[1]

    top_noisy = conn.execute("""
        SELECT file_path, COUNT(*) as cnt
        FROM feedback WHERE label = 'noise'
        GROUP BY file_path ORDER BY cnt DESC LIMIT 10
    """).fetchall()

    top_useful = conn.execute("""
        SELECT file_path, COUNT(*) as cnt
        FROM feedback WHERE label = 'useful'
        GROUP BY file_path ORDER BY cnt DESC LIMIT 10
    """).fetchall()

    by_mode = {}
    for row in conn.execute("""
        SELECT COALESCE(mode, 'none'), label, COUNT(*)
        FROM feedback GROUP BY mode, label
    """).fetchall():
        mode_name = row[0]
        if mode_name not in by_mode:
            by_mode[mode_name] = {}
        by_mode[mode_name][row[1]] = row[2]

    # Compute current weights for context
    weights = load_relevance_weights(conn)
    weight_extremes = {}
    if weights:
        sorted_w = sorted(weights.items(), key=lambda kv: kv[1])
        weight_extremes["most_penalised"] = sorted_w[:5]
        weight_extremes["most_boosted"] = sorted_w[-5:][::-1]

    return {
        "total": total,
        "by_label": by_label,
        "top_noisy": top_noisy,
        "top_useful": top_useful,
        "by_mode": by_mode,
        "weight_extremes": weight_extremes,
    }
