"""Tests for the relevance feedback system."""

import sqlite3
import time

import pytest

import sense_mcp.server as server_module
import sense_mcp.session as session_module
from sense_mcp.feedback import (
    get_feedback_stats,
    init_feedback_table,
    load_relevance_weights,
    record_feedback,
)
from sense_mcp.session import SessionState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def feedback_db(db):
    """DB fixture with feedback table guaranteed (db fixture runs _init_db
    which now calls init_feedback_table)."""
    return db


# ---------------------------------------------------------------------------
# Feedback table basics
# ---------------------------------------------------------------------------


class TestFeedbackTable:
    def test_init_creates_table(self, feedback_db):
        tables = feedback_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'"
        ).fetchall()
        assert len(tables) == 1

    def test_record_feedback_inserts_row(self, feedback_db):
        record_feedback(
            feedback_db,
            query_text="test query",
            file_path="/some/file.md",
            label="useful",
            similarity=0.65,
            mode="explore",
            note="very relevant",
        )
        rows = feedback_db.execute("SELECT * FROM feedback").fetchall()
        assert len(rows) == 1
        assert rows[0][3] == "useful"  # label column

    def test_invalid_label_raises(self, feedback_db):
        with pytest.raises(ValueError, match="Invalid label"):
            record_feedback(
                feedback_db,
                query_text="q",
                file_path="/f.md",
                label="maybe",
            )

    def test_label_constraint_in_db(self, feedback_db):
        """Direct SQL insert with bad label should fail."""
        with pytest.raises(sqlite3.IntegrityError):
            feedback_db.execute(
                "INSERT INTO feedback (query_text, file_path, label, created_at) "
                "VALUES ('q', '/f', 'bad', '2026-01-01')"
            )


# ---------------------------------------------------------------------------
# Relevance weights
# ---------------------------------------------------------------------------


class TestRelevanceWeights:
    def test_no_feedback_returns_empty(self, feedback_db):
        weights = load_relevance_weights(feedback_db)
        assert weights == {}

    def test_useful_boosts_weight(self, feedback_db):
        for _ in range(3):
            record_feedback(feedback_db, "q", "/good.md", "useful")
        weights = load_relevance_weights(feedback_db)
        assert weights["/good.md"] > 1.0

    def test_noise_penalises_weight(self, feedback_db):
        for _ in range(3):
            record_feedback(feedback_db, "q", "/noisy.md", "noise")
        weights = load_relevance_weights(feedback_db)
        assert weights["/noisy.md"] < 1.0

    def test_mixed_feedback_near_neutral(self, feedback_db):
        # Use different query texts so latest-wins gives one useful, one noise
        for i in range(3):
            record_feedback(feedback_db, f"q_useful_{i}", "/mixed.md", "useful")
            record_feedback(feedback_db, f"q_noise_{i}", "/mixed.md", "noise")
        weights = load_relevance_weights(feedback_db)
        # With equal useful and noise across different queries, weight ~1.0
        assert abs(weights["/mixed.md"] - 1.0) < 0.01

    def test_latest_wins_correction_overrides_auto_label(self, feedback_db):
        # Auto-label says noise for query "q1"
        record_feedback(feedback_db, "q1", "/corrected.md", "noise", source="auto:hook")
        weights_before = load_relevance_weights(feedback_db)
        assert weights_before["/corrected.md"] < 1.0

        # Human correction says useful for same (file_path, query_text) pair
        record_feedback(feedback_db, "q1", "/corrected.md", "useful", source="corrected:mat")
        weights_after = load_relevance_weights(feedback_db)
        assert weights_after["/corrected.md"] > 1.0

        # Both rows still in table (append-only audit trail)
        count = feedback_db.execute(
            "SELECT COUNT(*) FROM feedback WHERE file_path = '/corrected.md'"
        ).fetchone()[0]
        assert count == 2

    def test_prior_dampens_single_signal(self, feedback_db):
        record_feedback(feedback_db, "q", "/one.md", "noise")
        weights = load_relevance_weights(feedback_db, prior=2.0)
        # Single noise with prior=2.0: 1.0 + 0.3 * (-1)/5 = 0.94
        assert weights["/one.md"] > 0.9
        assert weights["/one.md"] < 1.0

    def test_custom_boost_factor(self, feedback_db):
        for _ in range(5):
            record_feedback(feedback_db, "q", "/f.md", "noise")
        w_low = load_relevance_weights(feedback_db, boost_factor=0.1)
        w_high = load_relevance_weights(feedback_db, boost_factor=0.5)
        # Higher boost_factor = more extreme penalty
        assert w_high["/f.md"] < w_low["/f.md"]


# ---------------------------------------------------------------------------
# Feedback stats
# ---------------------------------------------------------------------------


class TestFeedbackStats:
    def test_empty_stats(self, feedback_db):
        stats = get_feedback_stats(feedback_db)
        assert stats["total"] == 0
        assert stats["by_label"] == {}

    def test_stats_with_mixed_feedback(self, feedback_db):
        record_feedback(feedback_db, "q1", "/a.md", "useful", mode="explore")
        record_feedback(feedback_db, "q2", "/a.md", "useful", mode="explore")
        record_feedback(feedback_db, "q3", "/b.md", "noise", mode="build")
        record_feedback(feedback_db, "q4", "/c.md", "noise", mode="build")
        record_feedback(feedback_db, "q5", "/c.md", "noise", mode="build")

        stats = get_feedback_stats(feedback_db)
        assert stats["total"] == 5
        assert stats["by_label"]["useful"] == 2
        assert stats["by_label"]["noise"] == 3
        assert stats["top_useful"][0] == ("/a.md", 2)
        assert stats["top_noisy"][0] == ("/c.md", 2)
        assert stats["by_mode"]["explore"]["useful"] == 2
        assert stats["by_mode"]["build"]["noise"] == 3


# ---------------------------------------------------------------------------
# Session last_results
# ---------------------------------------------------------------------------


class TestLastResults:
    def test_record_last_results(self):
        state = SessionState()
        results = [
            {"file_path": "/a.md", "similarity": 0.8, "score": 0.7},
            {"file_path": "/b.md", "similarity": 0.6, "score": 0.5},
        ]
        state.record_last_results(results, "test query")
        assert len(state.last_results) == 2
        assert state.last_results[0]["file_path"] == "/a.md"
        assert state.last_results[0]["query"] == "test query"
        assert state.last_results[0]["similarity"] == 0.8

    def test_last_results_round_trip(self, tmp_path):
        state_path = str(tmp_path / "state.json")
        state = SessionState()
        state.record_last_results(
            [{"file_path": "/x.md", "similarity": 0.5}], "q"
        )
        state.save(state_path)

        loaded = SessionState.load(state_path)
        assert len(loaded.last_results) == 1
        assert loaded.last_results[0]["file_path"] == "/x.md"

    def test_last_results_capped(self):
        state = SessionState()
        results = [
            {"file_path": f"/f{i}.md", "similarity": 0.5} for i in range(30)
        ]
        state.record_last_results(results, "q", cap=20)
        assert len(state.last_results) == 20


# ---------------------------------------------------------------------------
# Weight cache
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Source column and schema migration (SPEC-003)
# ---------------------------------------------------------------------------


class TestSourceColumn:
    def test_source_defaults_to_manual(self, feedback_db):
        record_feedback(feedback_db, "q", "/f.md", "useful")
        row = feedback_db.execute(
            "SELECT source FROM feedback WHERE file_path = '/f.md'"
        ).fetchone()
        assert row[0] == "manual"

    def test_auto_hook_source(self, feedback_db):
        record_feedback(feedback_db, "q", "/f.md", "useful", source="auto:hook")
        row = feedback_db.execute(
            "SELECT source FROM feedback WHERE file_path = '/f.md'"
        ).fetchone()
        assert row[0] == "auto:hook"

    def test_corrected_mat_source(self, feedback_db):
        record_feedback(feedback_db, "q", "/f.md", "noise", source="corrected:mat")
        row = feedback_db.execute(
            "SELECT source FROM feedback WHERE file_path = '/f.md'"
        ).fetchone()
        assert row[0] == "corrected:mat"

    def test_stats_include_source_breakdown(self, feedback_db):
        record_feedback(feedback_db, "q1", "/a.md", "useful", source="auto:hook")
        record_feedback(feedback_db, "q2", "/b.md", "noise", source="auto:hook")
        record_feedback(feedback_db, "q1", "/a.md", "noise", source="corrected:mat")
        stats = get_feedback_stats(feedback_db)
        assert stats["by_source"]["auto:hook"] == 2
        assert stats["by_source"]["corrected:mat"] == 1
        assert stats["correction_rate"] == 0.5  # 1 correction / 2 auto

    def test_schema_migration_adds_source_column(self, tmp_path):
        """Simulate a pre-SPEC-003 database without source column."""
        db_path = tmp_path / "old.db"
        conn = sqlite3.connect(str(db_path))
        # Create old schema without source column
        conn.executescript("""
            CREATE TABLE feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                file_path TEXT NOT NULL,
                label TEXT NOT NULL CHECK(label IN ('useful', 'noise')),
                similarity REAL,
                mode TEXT,
                note TEXT,
                created_at TEXT NOT NULL
            );
        """)
        conn.execute(
            "INSERT INTO feedback (query_text, file_path, label, created_at) "
            "VALUES ('q', '/old.md', 'useful', '2026-01-01')"
        )
        conn.commit()

        # Run migration
        init_feedback_table(conn)

        # Old row should have 'manual' default
        row = conn.execute("SELECT source FROM feedback").fetchone()
        assert row[0] == "manual"

        # New inserts should work with source
        record_feedback(conn, "q2", "/new.md", "noise", source="auto:hook")
        row = conn.execute(
            "SELECT source FROM feedback WHERE file_path = '/new.md'"
        ).fetchone()
        assert row[0] == "auto:hook"
        conn.close()


class TestWeightCache:
    def test_invalidate_forces_reload(self, feedback_db):
        # Prime the cache
        w1 = server_module._get_relevance_weights()
        assert w1 == {}

        # Add feedback
        record_feedback(feedback_db, "q", "/f.md", "noise")

        # Cache still returns old value
        w2 = server_module._get_relevance_weights()
        assert w2 == {}

        # Invalidate
        server_module._invalidate_weight_cache()
        w3 = server_module._get_relevance_weights()
        assert "/f.md" in w3
        assert w3["/f.md"] < 1.0
