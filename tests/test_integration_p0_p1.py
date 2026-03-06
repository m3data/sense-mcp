"""Integration tests: P0 and P1 fixes verified end-to-end.

Five scenarios covering session state sharing between MCP-style
(search_chunks_contextual) and hook-style (SessionState CRUD) callers.

No real OpenAI API calls — all embeddings are deterministic 2-D numpy vectors.
DB and session state are isolated per test via the conftest test_env autouse
fixture (tmp_path for DB path; tmp_path for session state file).
"""

import time
from datetime import datetime, timezone

import numpy as np
import pytest

import sense_mcp.server as server_module
from sense_mcp.hook import COOLDOWN_SECONDS, should_search
from sense_mcp.server import search_chunks_contextual
from sense_mcp.session import SessionState


# ---------------------------------------------------------------------------
# Embedding helpers
#
# 2-D normalised unit vectors — cosine similarity with _Q equals the
# first component of each vector (since _Q = [1, 0]).
#   sim(_Q, _HIGH) = 0.9
#   sim(_Q, _LOW)  = 0.3
# ---------------------------------------------------------------------------

_Q    = np.array([1.0,     0.0    ], dtype=np.float32)  # query vector
_HIGH = np.array([0.9,     0.43589], dtype=np.float32)  # sim ≈ 0.9 with _Q
_LOW  = np.array([0.3,     0.95394], dtype=np.float32)  # sim ≈ 0.3 with _Q


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

_TODAY    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
_NOW_ISO  = datetime.now(timezone.utc).isoformat()

_INSERT_SQL = """
    INSERT INTO chunks
        (file_path, file_hash, project, source_type, section, date,
         evergreen, content, token_count, embedding, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _insert_chunk(conn, file_path, project, source_type, emb_vec, content="test content"):
    """Insert a single chunk row with a controlled embedding vector."""
    conn.execute(
        _INSERT_SQL,
        (
            file_path, "deadbeef", project, source_type,
            None, _TODAY, 0, content, len(content.split()),
            emb_vec.tobytes(), _NOW_ISO,
        ),
    )


# ---------------------------------------------------------------------------
# Scenario 1: MCP search → shared state → hook penalty
# ---------------------------------------------------------------------------


def test_mcp_surfaced_files_recorded_in_shared_state(db, monkeypatch):
    """MCP search records surfaced files; hook-style load sees resurfacing penalty.

    Verifies P0: session state written by search_chunks_contextual is persisted
    to the shared state file and correctly read back by any caller.
    """
    # Stub embed_query so circling detection never hits the OpenAI API.
    # Zero vector → cosine similarity with any past query = 0 → no circling.
    monkeypatch.setattr(server_module, "embed_query", lambda _: np.zeros(2, dtype=np.float32))

    _insert_chunk(db, "/proj/file_a.md", "proj", "documentation", _HIGH)
    db.commit()

    # --- Simulate MCP search ---
    results, meta = search_chunks_contextual(
        _Q, "adaptive capacity query", mode="explore", limit=5
    )

    assert len(results) > 0
    assert any(r["file_path"] == "/proj/file_a.md" for r in results)
    assert meta["session_queries"] == 1

    # --- Hook-style load: shared state must reflect the MCP search ---
    state = SessionState.load()
    assert "/proj/file_a.md" in state.surfaced, (
        "Shared state should contain files surfaced by MCP search"
    )
    assert state.surfaced["/proj/file_a.md"]["count"] >= 1

    # --- Resurfacing penalty must apply for this file ---
    penalty = state.get_surfaced_penalty("/proj/file_a.md", base_penalty=0.75)
    assert penalty < 1.0, (
        "A file surfaced by MCP search should carry a resurfacing penalty < 1.0 "
        f"on subsequent hook lookups, got penalty={penalty}"
    )


# ---------------------------------------------------------------------------
# Scenario 2: Bidirectional state — hook records, MCP search sees it
# ---------------------------------------------------------------------------


def test_hook_surfaced_file_visible_to_mcp_search(db, monkeypatch):
    """State written by a hook call is visible to the next MCP search.

    Verifies P0 bidirectionality: SessionState is truly shared between the
    hook process and the MCP server process via the JSON state file.
    """
    monkeypatch.setattr(server_module, "embed_query", lambda _: np.zeros(2, dtype=np.float32))

    _insert_chunk(db, "/proj/file_b.md", "proj", "documentation", _HIGH)
    db.commit()

    # --- Simulate hook recording file_b as previously surfaced ---
    state = SessionState.load()
    state.record_surfaced([{"file_path": "/proj/file_b.md"}], surfaced_cap=500)
    state.save()

    # --- MCP search: only file in DB, must appear, should be marked resurfaced ---
    results, meta = search_chunks_contextual(
        _Q, "some query about adaptive systems", mode="explore", limit=5
    )

    assert len(results) > 0, "Expected at least one result (only one chunk in DB)"
    file_b = next(r for r in results if r["file_path"] == "/proj/file_b.md")
    assert file_b.get("resurfaced") is True, (
        "MCP search should mark file_b as resurfaced — hook wrote it to shared state first"
    )
    assert meta["resurfaced_count"] >= 1

    # Count should have incremented: hook wrote 1, MCP adds another 1
    final_state = SessionState.load()
    assert final_state.surfaced["/proj/file_b.md"]["count"] >= 2, (
        "Surfaced count should accumulate across hook and MCP calls"
    )


# ---------------------------------------------------------------------------
# Scenario 3: Circling detection across callers
# ---------------------------------------------------------------------------


def test_circling_detection_across_callers(db, monkeypatch):
    """A query recorded by hook triggers circling detection in the next MCP search.

    Verifies P1: get_circling_files() re-embeds past query texts on demand and
    correctly identifies files from semantically similar prior queries, regardless
    of which caller (hook or MCP) originally recorded the query.
    """
    # embed_query returns _Q so that embed_fn(past_query_text) == current query_embedding
    # → cosine similarity = 1.0 → past query is "similar" → circling fires
    monkeypatch.setattr(server_module, "embed_query", lambda _: _Q.copy())

    _insert_chunk(db, "/proj/circling.md", "proj", "documentation", _HIGH)
    db.commit()

    # --- Hook-style: record a prior query that surfaced circling.md ---
    state = SessionState.load()
    state.record_query(
        "adaptive systems prior query",
        ["/proj/circling.md"],
        max_queries=50,
    )
    state.save()

    # --- MCP search with semantically similar query ---
    results, meta = search_chunks_contextual(
        _Q, "adaptive systems current query", mode="explore", limit=5
    )

    assert meta["circling_count"] > 0, (
        "Circling detection should fire when current query is similar to a prior "
        "query recorded by the hook"
    )
    circ = next(
        (r for r in results if r["file_path"] == "/proj/circling.md"), None
    )
    assert circ is not None, "/proj/circling.md should appear in results"
    assert circ.get("circling") is True, (
        "Result should be flagged circling=True"
    )


# ---------------------------------------------------------------------------
# Scenario 4: Stratified pool — minority research type surfaces in think-with
# ---------------------------------------------------------------------------


def test_stratified_pool_surfaces_minority_research_type(db, monkeypatch, test_env):
    """Research chunks appear in think-with results despite being outnumbered by docs.

    Fixture: 20 documentation chunks (high similarity) + 3 research + 3 code (low
    similarity).  Without the stratified pool and mode multiplier, only documentation
    would dominate.  With think-with (research×1.5) + wide diversity [4,3,3],
    research chunks surface via the divergence slots.
    """
    monkeypatch.setattr(server_module, "embed_query", lambda _: np.zeros(2, dtype=np.float32))

    # 20 documentation chunks — high baseline similarity, will fill confirmation slots
    for i in range(20):
        _insert_chunk(
            db, f"/docs/doc_{i:02d}.md", "proj", "documentation",
            _HIGH, f"documentation content {i}",
        )

    # 3 research chunks — lower baseline similarity
    for i in range(3):
        _insert_chunk(
            db, f"/research/res_{i}.md", "proj", "research",
            _LOW, f"research content {i}",
        )

    # 3 code chunks — lower baseline similarity
    for i in range(3):
        _insert_chunk(
            db, f"/code/code_{i}.py", "proj", "code",
            _LOW, f"code content {i}",
        )

    db.commit()

    # Verify think-with profile has expected research multiplier
    profile = test_env.mode_profiles["think-with"]
    assert profile["source_type_multipliers"]["research"] == pytest.approx(1.5)
    assert profile["diversity_profile"] == "wide"

    results, meta = search_chunks_contextual(
        _Q, "adaptive systems thinking", mode="think-with", limit=10
    )

    assert meta["mode"] == "think-with"
    assert len(results) > 0

    types_in_results = {r["source_type"] for r in results}
    assert "research" in types_in_results, (
        "Research chunks should appear in think-with results despite low baseline "
        f"rank — mode multiplier (1.5) + divergence slots should surface them. "
        f"Got types: {types_in_results}"
    )


# ---------------------------------------------------------------------------
# Scenario 5: Cooldown gate uses shared state across callers
# ---------------------------------------------------------------------------


def test_cooldown_shared_across_callers(monkeypatch):
    """Hook cooldown gate rejects when shared state shows a recent query by any caller.

    Verifies P0: last_query_time written by one caller (MCP or hook) is visible
    to the hook's should_search gate, preventing redundant searches within the
    cooldown window.
    """
    # Simulate MCP server updating last_query_time in shared state
    state = SessionState.load()
    state.last_query_time = time.time()  # just now
    state.save()

    # Hook reads the same shared state
    loaded_state = SessionState.load()
    assert loaded_state.last_query_time > 0, (
        "last_query_time should have been persisted by the simulated MCP call"
    )
    assert time.time() - loaded_state.last_query_time < COOLDOWN_SECONDS, (
        "last_query_time should be recent (within cooldown window)"
    )

    # A substantive prompt (≥8 words, not a continuation, not a slash command)
    prompt = "How does adaptive capacity develop under systemic stress conditions"
    assert not should_search(prompt, loaded_state), (
        "Hook gate should return False when shared state shows a query was made "
        "within the cooldown window, regardless of which caller triggered it"
    )
