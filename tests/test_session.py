"""Tests for session tracking functions — REQ-006.

Covers:
  - get_surfaced_penalty: unseen files, seen files (penalty^count), floor at 0.05
  - detect_circling_topics: files from similar queries, empty set with no prior queries
  - record_surfaced: caps at surfaced_cap
"""

import numpy as np
import pytest

import sense_mcp.server as server_module
from sense_mcp.server import (
    detect_circling_topics,
    get_surfaced_penalty,
    record_query,
    record_surfaced,
)


class TestGetSurfacedPenalty:
    def test_returns_one_for_unseen_file(self):
        """Unseen files get no penalty — multiplier is 1.0."""
        result = get_surfaced_penalty("some/unseen/file.md", base_penalty=0.75)
        assert result == 1.0

    def test_returns_penalty_power_count_for_seen_file(self):
        """Penalty compounds multiplicatively: penalty^count."""
        server_module._session_surfaced["path/to/file.md"] = {"count": 2, "last_ts": 0.0}
        result = get_surfaced_penalty("path/to/file.md", base_penalty=0.75)
        assert result == pytest.approx(0.75 ** 2)

    def test_floors_at_0_05(self):
        """Penalty never goes below 0.05 regardless of count."""
        server_module._session_surfaced["path/to/file.md"] = {"count": 100, "last_ts": 0.0}
        result = get_surfaced_penalty("path/to/file.md", base_penalty=0.75)
        assert result == pytest.approx(0.05)


class TestDetectCirclingTopics:
    def test_returns_empty_set_with_no_prior_queries(self):
        """No prior queries → no circling topics returned."""
        embedding = np.ones(4, dtype=np.float32)
        result = detect_circling_topics(embedding)
        assert result == set()

    def test_returns_files_from_similar_queries(self):
        """Files from semantically-similar past queries are returned."""
        embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        past_files = ["project-a/TRACE_2026.md", "project-b/README.md"]
        record_query("prior query", embedding, past_files)
        result = detect_circling_topics(embedding, threshold=0.75)
        assert result == set(past_files)


class TestRecordSurfaced:
    def test_caps_at_surfaced_cap(self, test_env):
        """_session_surfaced never exceeds cfg.surfaced_cap entries."""
        cap = test_env.surfaced_cap  # 100 from test_config.toml
        results = [{"file_path": f"file_{i}.md"} for i in range(cap + 20)]
        record_surfaced(results)
        assert len(server_module._session_surfaced) <= cap
