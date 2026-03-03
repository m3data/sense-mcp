"""Tests for assemble_diverse_results() — REQ-005.

Covers:
  1. Results split into correct pools (confirmation / divergence / serendipity)
  2. Confirmation filled first from top-ranked candidates
  3. Divergence prefers different source_type or project
  4. Serendipity prefers unseen projects
  5. Results below min_similarity excluded
  6. Overflow fills remaining serendipity slots from leftovers
  7. slot_type correctly assigned on each result
  8. No duplicate file_paths across pools
"""

import pytest

from sense_mcp.server import assemble_diverse_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_result(
    file_path: str,
    similarity: float,
    source_type: str = "doc",
    project: str = "project-a",
    score: float | None = None,
) -> dict:
    return {
        "file_path": file_path,
        "similarity": similarity,
        "score": score if score is not None else similarity,
        "source_type": source_type,
        "project": project,
        "content": "x",
        "section": None,
        "date": None,
        "token_count": 10,
        "decay": 1.0,
    }


# ---------------------------------------------------------------------------
# 1. Results split into correct pools
# ---------------------------------------------------------------------------

def test_correct_pool_sizes():
    """Counts of each slot_type match requested slots."""
    results = [
        make_result("a/f1.md", 0.9, source_type="doc",   project="alpha"),
        make_result("a/f2.md", 0.8, source_type="doc",   project="alpha"),
        make_result("b/f3.md", 0.7, source_type="trace", project="beta"),
        make_result("c/f4.md", 0.6, source_type="code",  project="gamma"),
        make_result("d/f5.md", 0.5, source_type="doc",   project="delta"),
    ]
    out = assemble_diverse_results(results, slots=(2, 1, 1), current_project=None)
    counts = {t: 0 for t in ("confirmation", "divergence", "serendipity")}
    for r in out:
        counts[r["slot_type"]] += 1
    assert counts["confirmation"] == 2
    assert counts["divergence"] == 1
    assert counts["serendipity"] == 1


# ---------------------------------------------------------------------------
# 2. Confirmation filled first from top-ranked
# ---------------------------------------------------------------------------

def test_confirmation_uses_top_ranked():
    """Confirmation pool draws the highest-scoring candidates."""
    results = [
        make_result("a/high.md",  0.95, project="alpha"),
        make_result("a/mid.md",   0.70, project="alpha"),
        make_result("b/low.md",   0.50, project="beta"),
    ]
    out = assemble_diverse_results(results, slots=(2, 0, 0), current_project=None)
    confirm_paths = {r["file_path"] for r in out if r["slot_type"] == "confirmation"}
    assert "a/high.md" in confirm_paths
    assert "a/mid.md" in confirm_paths
    assert "b/low.md" not in confirm_paths


# ---------------------------------------------------------------------------
# 3. Divergence prefers different source_type or project
# ---------------------------------------------------------------------------

def test_divergence_prefers_different_source_type():
    """A result with a novel source_type is chosen for divergence."""
    results = [
        make_result("a/f1.md", 0.9, source_type="doc",   project="alpha"),
        make_result("a/f2.md", 0.8, source_type="doc",   project="alpha"),
        make_result("b/f3.md", 0.7, source_type="trace", project="alpha"),  # different source_type
        make_result("a/f4.md", 0.6, source_type="doc",   project="alpha"),
    ]
    out = assemble_diverse_results(results, slots=(2, 1, 0), current_project=None)
    divergent = [r for r in out if r["slot_type"] == "divergence"]
    assert len(divergent) == 1
    assert divergent[0]["file_path"] == "b/f3.md"


def test_divergence_prefers_different_project():
    """A result from a novel project is chosen for divergence."""
    results = [
        make_result("a/f1.md", 0.9, source_type="doc", project="alpha"),
        make_result("a/f2.md", 0.8, source_type="doc", project="alpha"),
        make_result("b/f3.md", 0.7, source_type="doc", project="beta"),   # different project
        make_result("a/f4.md", 0.6, source_type="doc", project="alpha"),
    ]
    out = assemble_diverse_results(results, slots=(2, 1, 0), current_project=None)
    divergent = [r for r in out if r["slot_type"] == "divergence"]
    assert len(divergent) == 1
    assert divergent[0]["file_path"] == "b/f3.md"


# ---------------------------------------------------------------------------
# 4. Serendipity prefers unseen projects
# ---------------------------------------------------------------------------

def test_serendipity_prefers_unseen_projects():
    """Serendipity slot picks a result from a project not yet seen."""
    results = [
        make_result("a/f1.md", 0.9, source_type="doc",  project="alpha"),
        make_result("b/f2.md", 0.8, source_type="trace", project="beta"),
        make_result("a/f3.md", 0.7, source_type="doc",  project="alpha"),
        make_result("c/f4.md", 0.6, source_type="doc",  project="gamma"),  # unseen project
        make_result("a/f5.md", 0.5, source_type="doc",  project="alpha"),
    ]
    out = assemble_diverse_results(results, slots=(1, 1, 1), current_project=None)
    serendip = [r for r in out if r["slot_type"] == "serendipity"]
    assert len(serendip) == 1
    assert serendip[0]["project"] not in {"alpha", "beta"}


def test_serendipity_excludes_current_project():
    """current_project is treated as already-seen for serendipity."""
    results = [
        make_result("a/f1.md", 0.9, source_type="doc", project="alpha"),
        make_result("b/f2.md", 0.8, source_type="doc", project="current"),
        make_result("c/f3.md", 0.7, source_type="doc", project="gamma"),
    ]
    out = assemble_diverse_results(
        results, slots=(1, 0, 1), current_project="current"
    )
    serendip = [r for r in out if r["slot_type"] == "serendipity"]
    assert len(serendip) == 1
    assert serendip[0]["project"] != "current"


# ---------------------------------------------------------------------------
# 5. Below min_similarity excluded
# ---------------------------------------------------------------------------

def test_below_min_similarity_excluded():
    """Results with similarity < min_similarity never appear in output."""
    results = [
        make_result("a/ok.md",   0.5, project="alpha"),
        make_result("b/low.md",  0.1, project="beta"),   # below default 0.20
        make_result("c/edge.md", 0.2, project="gamma"),  # exactly at threshold
    ]
    out = assemble_diverse_results(results, slots=(3, 0, 0), current_project=None)
    paths = {r["file_path"] for r in out}
    assert "b/low.md" not in paths
    assert "a/ok.md" in paths
    assert "c/edge.md" in paths


def test_custom_min_similarity():
    """Custom min_similarity threshold is respected."""
    results = [
        make_result("a/high.md", 0.8, project="alpha"),
        make_result("b/mid.md",  0.4, project="beta"),
        make_result("c/low.md",  0.3, project="gamma"),
    ]
    out = assemble_diverse_results(
        results, slots=(3, 0, 0), current_project=None, min_similarity=0.5
    )
    paths = {r["file_path"] for r in out}
    assert "a/high.md" in paths
    assert "b/mid.md" not in paths
    assert "c/low.md" not in paths


def test_all_below_min_similarity_returns_empty():
    """Empty list returned when no results meet the threshold."""
    results = [
        make_result("a/f1.md", 0.1, project="alpha"),
        make_result("b/f2.md", 0.15, project="beta"),
    ]
    out = assemble_diverse_results(results, slots=(2, 1, 1), current_project=None)
    assert out == []


# ---------------------------------------------------------------------------
# 6. Overflow fills remaining serendipity slots
# ---------------------------------------------------------------------------

def test_serendipity_overflow_fills_from_leftovers():
    """When no novel projects remain, serendipity fills from leftover viable results."""
    results = [
        make_result("a/f1.md", 0.9, source_type="doc",  project="alpha"),
        make_result("a/f2.md", 0.8, source_type="doc",  project="alpha"),
        make_result("a/f3.md", 0.7, source_type="doc",  project="alpha"),
        make_result("a/f4.md", 0.6, source_type="doc",  project="alpha"),
    ]
    # All results are from the same project — serendipity has to overflow
    out = assemble_diverse_results(results, slots=(1, 0, 2), current_project=None)
    serendip = [r for r in out if r["slot_type"] == "serendipity"]
    # Should still fill 2 serendipity slots from remaining viable results
    assert len(serendip) == 2


# ---------------------------------------------------------------------------
# 7. slot_type correctly assigned
# ---------------------------------------------------------------------------

def test_slot_type_assigned_to_all_results():
    """Every result in the output has a valid slot_type field."""
    results = [
        make_result("a/f1.md", 0.9, source_type="doc",   project="alpha"),
        make_result("b/f2.md", 0.8, source_type="trace", project="beta"),
        make_result("c/f3.md", 0.7, source_type="code",  project="gamma"),
        make_result("d/f4.md", 0.6, source_type="doc",   project="delta"),
    ]
    out = assemble_diverse_results(results, slots=(1, 1, 1), current_project=None)
    valid_types = {"confirmation", "divergence", "serendipity"}
    for r in out:
        assert "slot_type" in r
        assert r["slot_type"] in valid_types


def test_slot_type_values_match_expected_assignment():
    """Each pool has only its own slot_type value."""
    results = [
        make_result("a/f1.md", 0.95, source_type="doc",  project="alpha"),
        make_result("a/f2.md", 0.90, source_type="doc",  project="alpha"),
        make_result("b/f3.md", 0.80, source_type="trace", project="beta"),
        make_result("c/f4.md", 0.70, source_type="code",  project="gamma"),
    ]
    out = assemble_diverse_results(results, slots=(2, 1, 1), current_project=None)
    by_type = {t: [] for t in ("confirmation", "divergence", "serendipity")}
    for r in out:
        by_type[r["slot_type"]].append(r)
    assert len(by_type["confirmation"]) == 2
    assert len(by_type["divergence"]) == 1
    assert len(by_type["serendipity"]) == 1


# ---------------------------------------------------------------------------
# 8. No duplicate file_paths across pools
# ---------------------------------------------------------------------------

def test_no_duplicate_file_paths():
    """The same file_path never appears in more than one pool."""
    results = [
        make_result("a/f1.md", 0.9, source_type="doc",   project="alpha"),
        make_result("b/f2.md", 0.8, source_type="trace", project="beta"),
        make_result("c/f3.md", 0.7, source_type="code",  project="gamma"),
        make_result("d/f4.md", 0.6, source_type="doc",   project="delta"),
        make_result("e/f5.md", 0.5, source_type="doc",   project="epsilon"),
    ]
    out = assemble_diverse_results(results, slots=(2, 1, 1), current_project=None)
    paths = [r["file_path"] for r in out]
    assert len(paths) == len(set(paths))


def test_no_duplicates_when_only_a_few_viable():
    """No duplicates even when viable pool is smaller than requested slots."""
    results = [
        make_result("a/f1.md", 0.9, project="alpha"),
        make_result("b/f2.md", 0.8, project="beta"),
    ]
    out = assemble_diverse_results(results, slots=(2, 1, 1), current_project=None)
    paths = [r["file_path"] for r in out]
    assert len(paths) == len(set(paths))
