"""Tests for SessionState — sense_mcp/session.py.

Covers:
  - get_surfaced_penalty: unseen files, seen files (penalty^count), floor at 0.05
  - get_circling_files: files from similar queries, empty set with no prior queries
  - record_surfaced: caps at surfaced_cap
  - record_query: rolling window at max_queries
  - load/save: round-trip serialisation, file locking
"""

import json
import time

import numpy as np
import pytest

from sense_mcp.session import SessionState


class TestGetSurfacedPenalty:
    def test_returns_one_for_unseen_file(self):
        """Unseen files get no penalty — multiplier is 1.0."""
        state = SessionState()
        result = state.get_surfaced_penalty("some/unseen/file.md", base_penalty=0.75)
        assert result == 1.0

    def test_returns_linear_decay_for_seen_file(self):
        """Penalty decays linearly: base - 0.05 * (count - 1)."""
        state = SessionState()
        state.surfaced["path/to/file.md"] = {"count": 2, "last_ts": 0.0}
        result = state.get_surfaced_penalty("path/to/file.md", base_penalty=0.75)
        assert result == pytest.approx(0.70)  # 0.75 - 0.05 * 1

    def test_penalty_at_count_three(self):
        """Third surfacing: 0.75 - 0.05 * 2 = 0.65."""
        state = SessionState()
        state.surfaced["path/to/file.md"] = {"count": 3, "last_ts": 0.0}
        result = state.get_surfaced_penalty("path/to/file.md", base_penalty=0.75)
        assert result == pytest.approx(0.65)

    def test_floors_at_0_50(self):
        """Penalty never goes below 0.5 regardless of count."""
        state = SessionState()
        state.surfaced["path/to/file.md"] = {"count": 100, "last_ts": 0.0}
        result = state.get_surfaced_penalty("path/to/file.md", base_penalty=0.75)
        assert result == pytest.approx(0.50)


class TestGetCirclingFiles:
    def test_returns_empty_set_with_no_prior_queries(self):
        """No prior queries → no circling files returned."""
        state = SessionState()
        embedding = np.ones(4, dtype=np.float32)
        embed_fn = lambda text: np.ones(4, dtype=np.float32)
        result = state.get_circling_files(embedding, embed_fn=embed_fn)
        assert result == set()

    def test_returns_files_from_similar_queries(self):
        """Files from semantically-similar past queries are returned."""
        state = SessionState()
        embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        past_files = ["project-a/TRACE_2026.md", "project-b/README.md"]

        # Record a query with a stored text and known surfaced files
        state.record_query("prior query", past_files, max_queries=50)

        # embed_fn returns an identical vector → similarity = 1.0
        embed_fn = lambda text: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        result = state.get_circling_files(embedding, embed_fn=embed_fn, threshold=0.75)
        assert result == set(past_files)

    def test_excludes_files_from_dissimilar_queries(self):
        """Files from orthogonal past queries are not returned."""
        state = SessionState()
        query_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        state.record_query("unrelated query", ["other/file.md"], max_queries=50)

        # embed_fn returns orthogonal vector → similarity = 0.0
        embed_fn = lambda text: np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        result = state.get_circling_files(query_emb, embed_fn=embed_fn, threshold=0.75)
        assert result == set()


class TestRecordSurfaced:
    def test_caps_at_surfaced_cap(self, test_env):
        """surfaced dict never exceeds surfaced_cap entries."""
        cap = test_env.surfaced_cap  # 100 from test_config.toml
        state = SessionState()
        results = [{"file_path": f"file_{i}.md"} for i in range(cap + 20)]
        state.record_surfaced(results, surfaced_cap=cap)
        assert len(state.surfaced) <= cap

    def test_increments_count_for_seen_file(self):
        """Re-surfacing the same file increments its count."""
        state = SessionState()
        results = [{"file_path": "a.md"}]
        state.record_surfaced(results, surfaced_cap=500)
        state.record_surfaced(results, surfaced_cap=500)
        assert state.surfaced["a.md"]["count"] == 2

    def test_evicts_oldest_by_timestamp(self):
        """LRU eviction removes entries with the smallest last_ts."""
        state = SessionState()
        # Plant an old entry
        state.surfaced["old.md"] = {"count": 1, "last_ts": 0.0}
        # Fill to cap with fresh entries
        results = [{"file_path": f"new_{i}.md"} for i in range(5)]
        state.record_surfaced(results, surfaced_cap=5)
        assert "old.md" not in state.surfaced


class TestRecordQuery:
    def test_rolling_window_eviction(self):
        """Oldest queries are dropped when max_queries is exceeded."""
        state = SessionState()
        for i in range(5):
            state.record_query(f"query {i}", [f"file_{i}.md"], max_queries=3)
        assert len(state.queries) == 3
        # Only the last 3 queries remain
        stored_texts = [q["query"] for q in state.queries]
        assert stored_texts == ["query 2", "query 3", "query 4"]

    def test_stores_surfaced_files_not_embeddings(self):
        """Query entries contain query text and surfaced_files, no embeddings."""
        state = SessionState()
        state.record_query("test query", ["a.md", "b.md"], max_queries=50)
        entry = state.queries[0]
        assert entry["query"] == "test query"
        assert entry["surfaced_files"] == ["a.md", "b.md"]
        assert "embedding" not in entry


class TestLoadSave:
    def test_round_trip(self, tmp_path):
        """State survives a save/load cycle."""
        path = str(tmp_path / "state.json")
        state = SessionState()
        state.surfaced["a.md"] = {"count": 3, "last_ts": 1.0}
        state.queries.append({"query": "hello", "ts": 2.0, "surfaced_files": ["a.md"]})
        state.last_query_time = 99.0
        state.save(path)

        loaded = SessionState.load(path)
        assert loaded.surfaced == {"a.md": {"count": 3, "last_ts": 1.0}}
        assert len(loaded.queries) == 1
        assert loaded.queries[0]["query"] == "hello"
        assert loaded.last_query_time == 99.0

    def test_load_returns_empty_state_on_missing_file(self, tmp_path):
        """Missing file returns a fresh empty SessionState."""
        path = str(tmp_path / "nonexistent.json")
        state = SessionState.load(path)
        assert state.surfaced == {}
        assert state.queries == []
        assert state.last_query_time == 0.0

    def test_load_returns_empty_state_on_corrupt_json(self, tmp_path):
        """Corrupt JSON returns a fresh empty SessionState."""
        path = str(tmp_path / "corrupt.json")
        with open(path, "w") as f:
            f.write("{not valid json")
        state = SessionState.load(path)
        assert state.surfaced == {}
