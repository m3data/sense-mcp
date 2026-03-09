"""Tests for contextual query construction (SPEC-001)."""

import time
import numpy as np
import pytest

from sense_mcp.hook import build_contextual_query, MIN_CONTEXT_QUERIES
from sense_mcp.session import SessionState


def _make_state(queries: list[dict]) -> SessionState:
    """Build a SessionState with pre-populated query history."""
    state = SessionState()
    state.queries = queries
    return state


def _make_query(text: str, age_seconds: int = 0) -> dict:
    """Build a query entry with controlled timestamp."""
    return {
        "query": text,
        "ts": time.time() - age_seconds,
        "surfaced_files": [],
    }


def _deterministic_embed(text: str) -> np.ndarray:
    """Deterministic pseudo-embedding for testing. Maps text to a stable vector."""
    rng = np.random.RandomState(hash(text) % (2**31))
    vec = rng.randn(64)
    return vec / np.linalg.norm(vec)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# TEST-003: Graceful fallback with insufficient context
# ---------------------------------------------------------------------------


class TestGracefulFallback:
    def test_no_prior_queries_returns_current(self):
        """With 0 prior queries, returns current embedding unchanged."""
        current = _deterministic_embed("test query")
        state = _make_state([])
        result = build_contextual_query(current, state, _deterministic_embed)
        np.testing.assert_array_equal(result, current)

    def test_one_prior_query_returns_current(self):
        """With 1 prior query (below MIN_CONTEXT_QUERIES), returns current unchanged."""
        current = _deterministic_embed("test query")
        state = _make_state([_make_query("prior query")])
        result = build_contextual_query(current, state, _deterministic_embed)
        np.testing.assert_array_equal(result, current)

    def test_two_prior_queries_triggers_blending(self):
        """With 2 prior queries (= MIN_CONTEXT_QUERIES), blending occurs."""
        current = _deterministic_embed("test query")
        state = _make_state([
            _make_query("first context"),
            _make_query("second context"),
        ])
        result = build_contextual_query(current, state, _deterministic_embed)
        # Should NOT be identical to current (blending happened)
        assert not np.allclose(result, current)


# ---------------------------------------------------------------------------
# TEST-002: Current message dominance
# ---------------------------------------------------------------------------


class TestCurrentMessageDominance:
    def test_composite_closer_to_current_than_any_context(self):
        """Composite embedding is always more similar to current message
        than to any individual context query."""
        current = _deterministic_embed("harness design architecture")
        queries = [
            _make_query("somatic ai safety refactor"),
            _make_query("hence spawn overnight loop"),
            _make_query("cooperative platform matching"),
        ]
        state = _make_state(queries)
        result = build_contextual_query(
            current, state, _deterministic_embed, max_context_weight=0.4
        )

        sim_to_current = _cosine(result, current)
        for q in queries:
            ctx_emb = _deterministic_embed(q["query"])
            sim_to_ctx = _cosine(result, ctx_emb)
            assert sim_to_current > sim_to_ctx, (
                f"Composite is closer to context '{q['query']}' "
                f"({sim_to_ctx:.4f}) than to current ({sim_to_current:.4f})"
            )

    def test_dominance_holds_with_many_context_queries(self):
        """Even with max window of context, current message still dominates."""
        current = _deterministic_embed("what was the query construct")
        queries = [_make_query(f"topic {i} about harness design") for i in range(10)]
        state = _make_state(queries)
        result = build_contextual_query(
            current, state, _deterministic_embed,
            context_window=10, max_context_weight=0.4,
        )

        sim_to_current = _cosine(result, current)
        # Current message contributes at least 60% weight, so similarity should be high
        assert sim_to_current > 0.8, f"Current similarity too low: {sim_to_current:.4f}"


# ---------------------------------------------------------------------------
# TEST-001: Composite improves topical relevance
# ---------------------------------------------------------------------------


class TestTopicalRelevance:
    def test_composite_incorporates_context_signal(self):
        """Composite embedding is closer to context queries than the raw query alone.
        Uses controlled geometry: current on axis 0, context on axis 1.
        Blending should pull composite toward axis 1."""
        current = np.zeros(64)
        current[0] = 1.0  # axis 0

        context_emb = np.zeros(64)
        context_emb[1] = 1.0  # axis 1

        def _fixed_embed(text: str) -> np.ndarray:
            return context_emb.copy()

        queries = [
            _make_query("ctx a"),
            _make_query("ctx b"),
            _make_query("ctx c"),
        ]
        state = _make_state(queries)

        composite = build_contextual_query(
            current, state, _fixed_embed, max_context_weight=0.4
        )

        # Raw query has zero similarity to context (orthogonal axes)
        sim_raw = _cosine(current, context_emb)
        # Composite should have nonzero similarity to context
        sim_composite = _cosine(composite, context_emb)
        assert sim_composite > sim_raw, (
            f"Composite ({sim_composite:.4f}) should be closer to context "
            f"than raw query ({sim_raw:.4f})"
        )
        # And composite should still be closer to current than to context
        sim_to_current = _cosine(composite, current)
        assert sim_to_current > sim_composite, (
            f"Composite should still be closer to current ({sim_to_current:.4f}) "
            f"than to context ({sim_composite:.4f})"
        )


# ---------------------------------------------------------------------------
# TEST-004: Session timeout
# ---------------------------------------------------------------------------


class TestSessionTimeout:
    def test_stale_queries_excluded(self):
        """Queries older than session_timeout are excluded from blending."""
        current = _deterministic_embed("current query")
        queries = [
            _make_query("old query 1", age_seconds=8000),
            _make_query("old query 2", age_seconds=7500),
            _make_query("old query 3", age_seconds=7000),
        ]
        state = _make_state(queries)
        result = build_contextual_query(
            current, state, _deterministic_embed,
            session_timeout=7200,  # 2 hours
        )
        # All queries older than timeout, so should fall back to current
        np.testing.assert_array_equal(result, current)

    def test_mix_of_fresh_and_stale(self):
        """Only fresh queries participate in blending."""
        current = _deterministic_embed("current query")
        queries = [
            _make_query("stale query", age_seconds=8000),
            _make_query("fresh query 1", age_seconds=100),
            _make_query("fresh query 2", age_seconds=50),
        ]
        state = _make_state(queries)
        result = build_contextual_query(
            current, state, _deterministic_embed,
            session_timeout=7200,
        )
        # Blending should occur (2 fresh queries >= MIN_CONTEXT_QUERIES)
        assert not np.allclose(result, current)


# ---------------------------------------------------------------------------
# TEST-005: Trajectory-aware decay
# ---------------------------------------------------------------------------


class TestTrajectoryAwareDecay:
    """Use controlled geometry: current along axis 0, context along axis 1.
    This ensures context pull has a consistent, measurable direction."""

    @staticmethod
    def _axis_embed(axis: int, dims: int = 64) -> np.ndarray:
        vec = np.zeros(dims)
        vec[axis] = 1.0
        return vec

    def _make_controlled_state(self):
        """Context queries point along different axes (orthogonal to current on axis 0).
        Varying axes means the weight distribution across queries matters —
        different decay factors produce geometrically distinct composites."""
        # Map query text to distinct axes
        _ctx_map = {"ctx 1": 1, "ctx 2": 2, "ctx 3": 3}

        def _ctx_embed(text: str) -> np.ndarray:
            axis = _ctx_map.get(text, 1)
            return self._axis_embed(axis)

        queries = [
            _make_query("ctx 1"),
            _make_query("ctx 2"),
            _make_query("ctx 3"),
        ]
        return _make_state(queries), _ctx_embed

    def test_converging_reduces_context_influence(self):
        """When converging, context has less influence (lower effective decay).
        Less context pull = composite stays closer to current (axis 0)."""
        current = self._axis_embed(0)
        state, ctx_embed = self._make_controlled_state()

        result_neutral = build_contextual_query(
            current, state, ctx_embed,
            trajectory_signal={"trend": "stable"},
        )
        result_converging = build_contextual_query(
            current, state, ctx_embed,
            trajectory_signal={"trend": "converging"},
        )

        # Converging should be closer to current (less context pull along axis 1)
        sim_neutral = _cosine(result_neutral, current)
        sim_converging = _cosine(result_converging, current)
        assert sim_converging > sim_neutral, (
            f"Converging ({sim_converging:.4f}) should be closer to current "
            f"than neutral ({sim_neutral:.4f})"
        )

    def test_diverging_increases_context_influence(self):
        """When diverging, context has more influence (higher effective decay).
        More context pull = composite moves further from current (axis 0)."""
        current = self._axis_embed(0)
        state, ctx_embed = self._make_controlled_state()

        result_neutral = build_contextual_query(
            current, state, ctx_embed,
            trajectory_signal={"trend": "stable"},
        )
        result_diverging = build_contextual_query(
            current, state, ctx_embed,
            trajectory_signal={"trend": "diverging"},
        )

        # Diverging should pull composite further from current (more context anchor)
        sim_neutral = _cosine(result_neutral, current)
        sim_diverging = _cosine(result_diverging, current)
        assert sim_diverging < sim_neutral, (
            f"Diverging ({sim_diverging:.4f}) should be further from current "
            f"than neutral ({sim_neutral:.4f})"
        )


# ---------------------------------------------------------------------------
# TEST-006: Output is normalised
# ---------------------------------------------------------------------------


class TestOutputNormalisation:
    def test_output_is_unit_vector(self):
        """Composite embedding is L2-normalised."""
        current = _deterministic_embed("test query")
        queries = [
            _make_query("context 1"),
            _make_query("context 2"),
        ]
        state = _make_state(queries)
        result = build_contextual_query(current, state, _deterministic_embed)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-6, f"Output norm {norm} is not unit length"
