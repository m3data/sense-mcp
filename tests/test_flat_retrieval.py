"""Integration tests for search_chunks() — REQ-008.

Loads the pre-built fixture DB (tests/fixtures/sense_test.db) via the
``fixture_db`` conftest fixture, which patches server.get_db() to return
the fixture connection.  Uses pre-computed query embeddings from the
``query_fixtures`` table via the ``query_embedding_lookup`` conftest
fixture so no live OpenAI calls are made.

Scenarios covered:
  1. Semantic relevance — a gold-standard query returns its expected
     document within the top 3 results.
  2. Project filter — passing project= returns only chunks from that project.
  3. source_type filter — passing source_type= returns only chunks of
     that type.
  4. Score ordering — results are ranked by composite score descending.
  5. Limit — the limit parameter caps the number of results returned.
"""

import pytest

from sense_mcp.server import search_chunks

# ---------------------------------------------------------------------------
# Query texts from GOLD_QUERIES in tests/generate_fixtures.py
# ---------------------------------------------------------------------------

# Scenario 1 & 4: documentation retrieval for project-a
_ENTRAINMENT_QUERY = "How does entrainment work in distributed sociotechnical systems?"
_ENTRAINMENT_EXPECTED_SUFFIX = "project-a/docs/overview.md"

# Scenario 2: project filter — all results should be from project-a
_PROJECT_FILTER_QUERY = _ENTRAINMENT_QUERY
_PROJECT_FILTER_VALUE = "project-a"

# Scenario 3: source_type filter — only trace chunks should be returned
_TRACE_QUERY = "MBA 915 teaching session observations and student engagement"
_SOURCE_TYPE_FILTER_VALUE = "trace"

# Scenario 5: limit — cooperative query has broad coverage
_LIMIT_QUERY = "Worker cooperative ownership structures and collective intelligence"


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestFlatRetrieval:
    """Integration tests for search_chunks() using the fixture DB."""

    def test_semantic_query_returns_expected_document_in_top_3(
        self, fixture_db, query_embedding_lookup
    ):
        """A gold-standard query returns its expected document within the top 3.

        Uses the entrainment query whose expected file is
        project-a/docs/overview.md.  Verifies at least one of the top-3
        result file_paths ends with that relative path.
        """
        emb = query_embedding_lookup[_ENTRAINMENT_QUERY]
        results = search_chunks(emb)

        assert results, "Expected non-empty results for the entrainment query"
        top_paths = [r["file_path"] for r in results[:3]]
        assert any(p.endswith(_ENTRAINMENT_EXPECTED_SUFFIX) for p in top_paths), (
            f"Expected '{_ENTRAINMENT_EXPECTED_SUFFIX}' in top-3 results.\n"
            f"Got: {top_paths}"
        )

    def test_project_filter_restricts_results(
        self, fixture_db, query_embedding_lookup
    ):
        """Passing project= returns only chunks belonging to that project.

        Runs the entrainment query filtered to project-a and asserts every
        returned result has project == 'project-a'.
        """
        emb = query_embedding_lookup[_PROJECT_FILTER_QUERY]
        results = search_chunks(emb, project=_PROJECT_FILTER_VALUE)

        assert results, f"Expected at least one result for project='{_PROJECT_FILTER_VALUE}'"
        for r in results:
            assert r["project"] == _PROJECT_FILTER_VALUE, (
                f"Expected project='{_PROJECT_FILTER_VALUE}', got '{r['project']}' "
                f"in {r['file_path']}"
            )

    def test_source_type_filter_restricts_results(
        self, fixture_db, query_embedding_lookup
    ):
        """Passing source_type= returns only chunks of that source type.

        Runs the trace-oriented query filtered to source_type='trace' and
        asserts every returned result has source_type == 'trace'.
        """
        emb = query_embedding_lookup[_TRACE_QUERY]
        results = search_chunks(emb, source_type=_SOURCE_TYPE_FILTER_VALUE)

        assert results, f"Expected at least one result for source_type='{_SOURCE_TYPE_FILTER_VALUE}'"
        for r in results:
            assert r["source_type"] == _SOURCE_TYPE_FILTER_VALUE, (
                f"Expected source_type='{_SOURCE_TYPE_FILTER_VALUE}', "
                f"got '{r['source_type']}' in {r['file_path']}"
            )

    def test_results_ordered_by_score_descending(
        self, fixture_db, query_embedding_lookup
    ):
        """Results are ranked by composite score in descending order.

        Iterates through consecutive pairs and asserts score[i] >= score[i+1].
        """
        emb = query_embedding_lookup[_ENTRAINMENT_QUERY]
        results = search_chunks(emb)

        assert len(results) >= 2, "Need at least 2 results to verify ordering"
        for i in range(len(results) - 1):
            assert results[i]["score"] >= results[i + 1]["score"], (
                f"Results not sorted: score[{i}]={results[i]['score']:.6f} < "
                f"score[{i + 1}]={results[i + 1]['score']:.6f}"
            )

    def test_limit_caps_results(self, fixture_db, query_embedding_lookup):
        """The limit parameter caps the number of results returned.

        Requests limit=3 and asserts at most 3 results are returned.
        """
        emb = query_embedding_lookup[_LIMIT_QUERY]
        results = search_chunks(emb, limit=3)

        assert len(results) <= 3, (
            f"Expected at most 3 results with limit=3, got {len(results)}"
        )
