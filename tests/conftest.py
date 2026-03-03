"""Shared test fixtures for the Sense MCP test suite.

Responsibilities:
  - Patch SENSE_CONFIG / SENSE_ROOT so every test loads the fixture config
    instead of any real sense.toml found on disk
  - Override cfg.db_path to a per-test tmp_path so DB state never leaks
  - Reset module-level session state (_session_surfaced, _session_queries)
    before and after each test
  - Reset lazy-init singletons (_db_conn, _openai_client) around each test
  - Provide an opt-in `db` fixture that wires up a real in-test SQLite
    connection with the schema initialised
"""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

import sense_mcp.config as config_module
import sense_mcp.server as server_module

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TEST_CONFIG_PATH = FIXTURES_DIR / "test_config.toml"
CORPUS_DIR = FIXTURES_DIR / "corpus"
FIXTURE_DB_PATH = FIXTURES_DIR / "sense_test.db"


# ---------------------------------------------------------------------------
# Core autouse fixture — isolates every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def test_env(monkeypatch, tmp_path):
    """Isolate each test: patch config, reset session state and singletons.

    Sets SENSE_CONFIG and SENSE_ROOT so config.py never picks up a real
    sense.toml.  Overrides cfg.db_path to a fresh tmp_path so DB state
    cannot leak between tests.  Tears everything down after yield.
    """
    # --- Config isolation ---
    monkeypatch.setenv("SENSE_CONFIG", str(TEST_CONFIG_PATH))
    monkeypatch.setenv("SENSE_ROOT", str(CORPUS_DIR))

    # Force the singleton to None so reload_config builds fresh
    monkeypatch.setattr(config_module, "_config", None)
    cfg = config_module.reload_config(TEST_CONFIG_PATH)

    # Point the DB at a per-test temp path (override the config-resolved path)
    cfg.db_path = tmp_path / "test.db"
    cfg.root = CORPUS_DIR

    # Patch server-module globals that were bound at import time
    monkeypatch.setattr(server_module, "cfg", cfg)
    monkeypatch.setattr(server_module, "DB_PATH", cfg.db_path)
    monkeypatch.setattr(server_module, "ECOSYSTEM_ROOT", CORPUS_DIR)

    # --- Session state reset ---
    server_module._session_surfaced.clear()
    server_module._session_queries.clear()

    # --- Singleton reset (close any open DB from a prior test) ---
    if server_module._db_conn is not None:
        server_module._db_conn.close()
    server_module._db_conn = None
    server_module._openai_client = None

    yield cfg

    # --- Teardown ---
    if server_module._db_conn is not None:
        server_module._db_conn.close()
    server_module._db_conn = None
    server_module._openai_client = None
    server_module._session_surfaced.clear()
    server_module._session_queries.clear()
    config_module._config = None


# ---------------------------------------------------------------------------
# Optional DB fixture — opt-in for tests that need a real connection
# ---------------------------------------------------------------------------

@pytest.fixture
def db(test_env, tmp_path):
    """Provide a fresh SQLite connection with the Sense schema initialised.

    Wires the connection into server._db_conn so server functions that call
    get_db() receive the test connection rather than opening a new one.

    Usage::

        def test_something(db):
            db.execute("SELECT COUNT(*) FROM chunks")
    """
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    server_module._init_db(conn)
    server_module._db_conn = conn

    yield conn

    conn.close()
    server_module._db_conn = None


# ---------------------------------------------------------------------------
# Fixture DB fixtures — opt-in for integration tests that use sense_test.db
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_db(monkeypatch):
    """Open a read-only connection to the pre-built fixture DB.

    Patches server.get_db() so search_chunks() and other DB-dependent
    functions read from the fixture data without touching the per-test
    tmp_path DB set up by test_env.

    Requires tests/fixtures/sense_test.db to exist (run
    ``python tests/generate_fixtures.py`` to build it).
    """
    conn = sqlite3.connect(f"file:{FIXTURE_DB_PATH}?mode=ro", uri=True)
    monkeypatch.setattr(server_module, "get_db", lambda: conn)
    yield conn
    conn.close()


@pytest.fixture
def query_embedding_lookup(fixture_db):
    """Return a dict mapping query_text -> pre-computed embedding (np.ndarray).

    Reads from the query_fixtures table in the fixture DB, which is
    populated by tests/generate_fixtures.py.
    """
    rows = fixture_db.execute(
        "SELECT query_text, embedding FROM query_fixtures"
    ).fetchall()
    return {
        row[0]: np.frombuffer(row[1], dtype=np.float32)
        for row in rows
    }
