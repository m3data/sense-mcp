#!/usr/bin/env python3
"""Sense Companion Dashboard — read-only view into Sense session data.

Serves a single HTML page and JSON API endpoints. Reads from:
  - sense.db (feedback table, read-only)
  - /tmp/sense-session-state.json (session state)
  - /tmp/sense-trajectory-history.jsonl (trajectory signal history)
  - ~/.vibe-harness/mode-history.jsonl (vibe mode)

No external dependencies beyond Python stdlib + sqlite3.

Usage:
    python sense-mcp/dashboard/serve.py [--port 8111] [--db path/to/sense.db]

SPEC-003 ADR-003: Single-file Python server, no build toolchain.
"""

import argparse
import json
import os
import sqlite3
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PORT = 8111
DEFAULT_DB = Path(__file__).resolve().parent.parent / "sense.db"
SESSION_STATE_PATH = "/tmp/sense-session-state.json"
TRAJECTORY_HISTORY_PATH = "/tmp/sense-trajectory-history.jsonl"
MODE_HISTORY_PATH = Path.home() / ".vibe-harness" / "mode-history.jsonl"
ECOSYSTEM_ROOT = Path(__file__).resolve().parent.parent.parent  # EarthianLabs/

# Resolved at startup
_db_path: Path = DEFAULT_DB


# ---------------------------------------------------------------------------
# Data readers
# ---------------------------------------------------------------------------

def read_session_state() -> dict:
    """Read current session state from shared JSON file."""
    try:
        with open(SESSION_STATE_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"surfaced": {}, "queries": [], "last_query_time": 0, "last_results": [], "trajectory_signal": {}}


def read_trajectory_history(since: float = 0) -> list[dict]:
    """Read trajectory history JSONL, optionally filtering by timestamp."""
    entries = []
    try:
        with open(TRAJECTORY_HISTORY_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("ts", 0) >= since:
                    entries.append(entry)
    except FileNotFoundError:
        pass
    return entries


def read_current_mode() -> dict:
    """Read current vibe mode from mode-history.jsonl."""
    try:
        if not MODE_HISTORY_PATH.exists():
            return {"mode": None, "history": []}
        with open(MODE_HISTORY_PATH, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            # Read last 4KB for recent history
            f.seek(max(0, size - 4096))
            tail = f.read().decode("utf-8", errors="replace")
        lines = [ln for ln in tail.strip().splitlines() if ln.strip()]
        history = []
        for line in lines:
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        current = history[-1].get("to_mode") if history else None
        return {"mode": current, "history": history[-20:]}
    except Exception:
        return {"mode": None, "history": []}


def query_feedback(db_path: Path, since: float = 0) -> list[dict]:
    """Query feedback rows from sense.db, optionally filtered by timestamp."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        if since > 0:
            from datetime import datetime, timezone
            since_iso = datetime.fromtimestamp(since, tz=timezone.utc).isoformat()
            rows = conn.execute(
                "SELECT * FROM feedback WHERE created_at >= ? ORDER BY id DESC",
                (since_iso,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM feedback ORDER BY id DESC LIMIT 200"
            ).fetchall()
        result = [dict(r) for r in rows]
        conn.close()
        return result
    except Exception:
        return []


def query_feedback_stats(db_path: Path, since_ts: float = 0) -> dict:
    """Aggregate feedback statistics, optionally scoped to a session window."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

        # Build WHERE clause for session scoping
        where = ""
        params: list = []
        if since_ts > 0:
            from datetime import datetime, timezone
            since_iso = datetime.fromtimestamp(since_ts, tz=timezone.utc).isoformat()
            where = " WHERE created_at >= ?"
            params = [since_iso]

        total = conn.execute(f"SELECT COUNT(*) FROM feedback{where}", params).fetchone()[0]
        if total == 0:
            conn.close()
            return {"total": 0, "useful": 0, "noise": 0, "hit_rate": 0,
                    "by_source": {}, "correction_rate": None, "recent_hit_rate": None}

        useful = conn.execute(
            f"SELECT COUNT(*) FROM feedback{where.replace('WHERE', 'WHERE label=? AND') if where else ' WHERE label=?'}",
            ["useful"] + params
        ).fetchone()[0]
        noise = conn.execute(
            f"SELECT COUNT(*) FROM feedback{where.replace('WHERE', 'WHERE label=? AND') if where else ' WHERE label=?'}",
            ["noise"] + params
        ).fetchone()[0]

        # Source breakdown
        by_source = {}
        try:
            for row in conn.execute(
                f"SELECT COALESCE(source, 'manual'), COUNT(*) FROM feedback{where} GROUP BY COALESCE(source, 'manual')",
                params
            ).fetchall():
                by_source[row[0]] = row[1]
        except sqlite3.OperationalError:
            pass

        # Recent hit rate (last 20 latest-wins labels, session-scoped)
        recent_rows = conn.execute(f"""
            WITH scoped AS (
                SELECT * FROM feedback{where}
            ),
            latest AS (
                SELECT label FROM scoped
                WHERE id IN (
                    SELECT MAX(id) FROM scoped GROUP BY file_path, query_text
                )
                ORDER BY id DESC LIMIT 20
            )
            SELECT label, COUNT(*) FROM latest GROUP BY label
        """, params).fetchall()
        recent_useful = sum(c for l, c in recent_rows if l == "useful")
        recent_total = sum(c for _, c in recent_rows)
        recent_hit_rate = recent_useful / recent_total if recent_total > 0 else None

        # Correction rate
        auto_count = sum(v for k, v in by_source.items() if k.startswith("auto:"))
        correction_count = sum(v for k, v in by_source.items() if k.startswith("corrected:"))
        correction_rate = correction_count / auto_count if auto_count > 0 else None

        conn.close()
        return {
            "total": total,
            "useful": useful,
            "noise": noise,
            "hit_rate": useful / (useful + noise) if (useful + noise) > 0 else 0,
            "by_source": by_source,
            "correction_rate": correction_rate,
            "recent_hit_rate": recent_hit_rate,
        }
    except Exception as e:
        return {"total": 0, "error": str(e)}


def record_correction(db_path: Path, file_path: str, query_text: str,
                       label: str, note: str = "",
                       similarity: float | None = None,
                       mode: str | None = None) -> dict:
    """Write a human correction to the feedback table (P1-6: includes similarity/mode)."""
    if label not in ("useful", "noise"):
        return {"error": f"Invalid label: {label}"}
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        from datetime import datetime, timezone
        conn.execute(
            """INSERT INTO feedback
               (query_text, file_path, label, similarity, mode, source, note, created_at)
               VALUES (?, ?, ?, ?, ?, 'corrected:mat', ?, ?)""",
            (query_text, file_path, label, similarity, mode, note or None,
             datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
        conn.close()
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class DashboardHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for dashboard endpoints."""

    def log_message(self, format, *args):
        pass  # suppress request logging

    def _json_response(self, data: dict | list, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _html_response(self, html: str):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            html_path = Path(__file__).parent / "index.html"
            if html_path.exists():
                self._html_response(html_path.read_text())
            else:
                self._html_response("<h1>Dashboard HTML not found</h1>")

        elif path == "/api/session":
            self._json_response(read_session_state())

        elif path == "/api/feedback":
            since = float(params.get("since", [0])[0])
            self._json_response(query_feedback(_db_path, since))

        elif path == "/api/feedback/stats":
            since_ts = float(params.get("since", [0])[0])
            self._json_response(query_feedback_stats(_db_path, since_ts))

        elif path == "/api/mode":
            self._json_response(read_current_mode())

        elif path == "/api/trajectory":
            since = float(params.get("since", [0])[0])
            history = read_trajectory_history(since)
            session = read_session_state()
            self._json_response({
                "current": session.get("trajectory_signal", {}),
                "history": history[-50:],
            })

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/feedback":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            result = record_correction(
                _db_path,
                file_path=body.get("file_path", ""),
                query_text=body.get("query_text", ""),
                label=body.get("label", ""),
                note=body.get("note", ""),
                similarity=body.get("similarity"),
                mode=body.get("mode"),
            )
            status = 200 if "ok" in result else 400
            self._json_response(result, status)

        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _db_path

    parser = argparse.ArgumentParser(description="Sense Companion Dashboard")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB))
    args = parser.parse_args()

    _db_path = Path(args.db)
    if not _db_path.exists():
        print(f"Warning: Database not found at {_db_path}")

    server = HTTPServer(("127.0.0.1", args.port), DashboardHandler)
    print(f"Sense Dashboard: http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
