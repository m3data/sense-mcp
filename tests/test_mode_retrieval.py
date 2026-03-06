"""Tests for search_chunks_contextual() — REQ-009.

Covers 7 scenarios:
1. No mode active → falls through to flat search (same file_paths as search_chunks)
2. mode='none' → bypasses auto-detection, metadata has mode=None
3. build mode → code results carry mode_multiplier=1.5
4. explore mode → cross-project results are flagged cross_project=True
5. cool-off mode → all mode_multiplier values are < 1.0
6. Metadata dict contains required fields (mode, diversity_profile, slots, session_queries)
7. session_queries count accumulates across multiple calls within the same process
"""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

import sense_mcp.server as server_module
from sense_mcp.server import search_chunks, search_chunks_contextual

FIXTURE_DB_PATH = Path(__file__).parent / "fixtures" / "sense_test.db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_embedding(conn: sqlite3.Connection, query_id: int) -> np.ndarray:
    """Return the pre-computed embedding for a query_fixtures row."""
    row = conn.execute(
        "SELECT embedding FROM query_fixtures WHERE id = ?", (query_id,)
    ).fetchone()
    assert row is not None, f"No query fixture with id={query_id}"
    return np.frombuffer(row[0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Scenario 1: No mode → falls through to flat search
# ---------------------------------------------------------------------------


def test_no_mode_falls_through_to_flat_search(fixture_db):
    """When no mode is active, contextual search returns same results as flat search."""
    # Query 1: entrainment/documentation — good general-purpose query
    emb = _load_embedding(fixture_db, 1)

    flat_results = search_chunks(emb, limit=5)
    ctx_results, meta = search_chunks_contextual(emb, "entrainment query", limit=5)

    assert meta["mode"] is None

    flat_paths = [r["file_path"] for r in flat_results]
    ctx_paths = [r["file_path"] for r in ctx_results]
    assert flat_paths == ctx_paths


# ---------------------------------------------------------------------------
# Scenario 2: mode='none' bypasses detection → flat results, mode=None in metadata
# ---------------------------------------------------------------------------


def test_mode_none_returns_flat_results(fixture_db):
    """Passing mode='none' forces flat search regardless of any active mode."""
    emb = _load_embedding(fixture_db, 2)  # Python/code query

    ctx_results, meta = search_chunks_contextual(
        emb, "rhythm synchronisation api", mode="none", limit=5
    )

    assert meta["mode"] is None
    # Verify results look like flat search output — no mode-specific fields
    for r in ctx_results:
        assert "mode_multiplier" not in r


# ---------------------------------------------------------------------------
# Scenario 3: build mode boosts code source_type (multiplier 1.5)
# ---------------------------------------------------------------------------


def test_build_mode_boosts_code(fixture_db):
    """In build mode, code chunks carry mode_multiplier=1.5."""
    # Query 2 is a code query (Python API) — likely to surface code chunks
    emb = _load_embedding(fixture_db, 2)

    results, meta = search_chunks_contextual(
        emb, "python api rhythm synchronisation", mode="build", limit=10
    )

    assert meta["mode"] == "build"

    code_results = [r for r in results if r["source_type"] == "code"]
    assert len(code_results) > 0, "Expected at least one code result in build mode"
    for r in code_results:
        assert r["mode_multiplier"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Scenario 4: explore mode enables cross-project retrieval (cross_project_boost=1.4)
# ---------------------------------------------------------------------------


def test_explore_mode_boosts_cross_project(fixture_db, test_env):
    """Explore mode's cross_project_boost is 1.4 and results span multiple projects."""
    # Verify the explore profile carries the expected cross_project_boost
    profile = test_env.mode_profiles["explore"]
    assert profile["cross_project_boost"] == pytest.approx(1.4)

    # With no project filter, explore mode's wide diversity profile (4, 3, 3) surfaces
    # results from across the corpus — multiple projects should appear
    emb = _load_embedding(fixture_db, 8)  # ecosystem overview query

    results, meta = search_chunks_contextual(
        emb, "ecosystem adaptive capacity", mode="explore", limit=10
    )

    assert meta["mode"] == "explore"

    projects = {r["project"] for r in results}
    assert len(projects) > 1, "Expected results from multiple projects in explore mode"


# ---------------------------------------------------------------------------
# Scenario 5: cool-off suppresses all source types (mode_multiplier < 1.0)
# ---------------------------------------------------------------------------


def test_cool_off_suppresses_all_source_types(fixture_db):
    """In cool-off mode, every result has mode_multiplier < 1.0."""
    emb = _load_embedding(fixture_db, 1)

    results, meta = search_chunks_contextual(
        emb, "entrainment query", mode="cool-off", limit=10
    )

    assert meta["mode"] == "cool-off"
    assert len(results) > 0, "Expected results even in cool-off mode"
    for r in results:
        assert r["mode_multiplier"] < 1.0, (
            f"Expected mode_multiplier < 1.0 in cool-off, got {r['mode_multiplier']} "
            f"for source_type={r['source_type']}"
        )


# ---------------------------------------------------------------------------
# Scenario 6: Metadata dict has correct fields
# ---------------------------------------------------------------------------


def test_metadata_has_required_fields(fixture_db):
    """Metadata dict contains mode, diversity_profile, slots, and session_queries."""
    emb = _load_embedding(fixture_db, 4)  # cooperative/documentation query

    _, meta = search_chunks_contextual(
        emb, "cooperative structures", mode="explore", limit=5
    )

    assert "mode" in meta
    assert "diversity_profile" in meta
    assert "slots" in meta
    assert "session_queries" in meta
    assert "circling_count" in meta
    assert "resurfaced_count" in meta

    # explore → wide diversity profile per test_config.toml
    assert meta["diversity_profile"] == "wide"
    assert meta["mode"] == "explore"
    # slots is a tuple/list of 3 ints matching [mode.diversity_slots.wide] = [4, 3, 3]
    assert list(meta["slots"]) == [4, 3, 3]


# ---------------------------------------------------------------------------
# Scenario 7: Session tracking accumulates across calls
# ---------------------------------------------------------------------------


def test_session_queries_accumulate_across_calls(fixture_db, monkeypatch):
    """session_queries count in metadata grows with each contextual search call."""
    # Stub embed_query so circling detection doesn't hit the OpenAI API.
    # A fixed zero vector ensures no past query ever exceeds the similarity
    # threshold, keeping the test focused on query-count accumulation.
    monkeypatch.setattr(server_module, "embed_query", lambda _: np.zeros(1536, dtype=np.float32))

    emb1 = _load_embedding(fixture_db, 1)
    emb2 = _load_embedding(fixture_db, 2)
    emb3 = _load_embedding(fixture_db, 3)

    _, meta1 = search_chunks_contextual(emb1, "query one", mode="build", limit=5)
    _, meta2 = search_chunks_contextual(emb2, "query two", mode="build", limit=5)
    _, meta3 = search_chunks_contextual(emb3, "query three", mode="build", limit=5)

    assert meta1["session_queries"] == 1
    assert meta2["session_queries"] == 2
    assert meta3["session_queries"] == 3


# ---------------------------------------------------------------------------
# Scenario 8: min_type_slots config defaults to 2
# ---------------------------------------------------------------------------


def test_min_type_slots_config_default(test_env):
    """The [mode] min_type_slots key defaults to 2."""
    assert test_env.min_type_slots == 2


# ---------------------------------------------------------------------------
# Scenario 9: Stratified pool ensures multiple source_types appear in results
# ---------------------------------------------------------------------------


def test_stratified_pool_includes_multiple_source_types(fixture_db, monkeypatch):
    """Stratified pool assembly produces results spanning multiple source_types.

    The fixture DB has 4 source_types (code, documentation, project_claude, trace).
    With explore mode (wide: [4,3,3]) and limit=10, the stratified pool (pool_size=80)
    guarantees per-type representation, so the final results should include chunks
    from at least 2 distinct source_types.
    """
    monkeypatch.setattr(server_module, "embed_query", lambda _: np.zeros(1536, dtype=np.float32))

    emb = _load_embedding(fixture_db, 1)

    results, meta = search_chunks_contextual(
        emb, "entrainment adaptive capacity", mode="explore", limit=10
    )

    assert meta["mode"] == "explore"
    assert len(results) > 0

    source_types_in_results = {r["source_type"] for r in results}
    assert len(source_types_in_results) >= 2, (
        f"Expected results from at least 2 source_types with stratified pool, "
        f"got: {source_types_in_results}"
    )
