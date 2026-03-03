---
id: SPEC-001
title: Sense Retrieval Evaluation Suite
status: draft
version: 1.0
created: 2026-03-04
author: agent:kairos
research-connection: RQ2 (adaptive capacity conditions), RQ5 (sensing/stewarding)
---

# SPEC-001: Sense Retrieval Evaluation Suite

## Context

[[Sense]] is the ambient relevance engine for the EarthianLabs ecosystem. It indexes ~1,258 files across ~23 projects, chunks and embeds them, and surfaces relevant context into every AI conversation automatically. It runs as an MCP server and via a Claude Code hook.

There are currently zero tests. The codebase has been through one recursive adversarial audit (RAA-POC-001, confidence 0.77) which identified 10 issues. Several pure functions have grown complex enough that changes carry regression risk: temporal decay, diversity injection, mode-aware reranking, chunking logic.

This spec defines Layer 1 evals: retrieval quality and pure function correctness. Hook relevance evaluation (Layer 2) and mode-aware calibration (Layer 3) are out of scope.

## Research Connection

The eval suite connects to the research frame through two questions:

- **RQ2 (conditions for adaptive capacity):** The eval criteria encode assumptions about what "relevant" means in context. Making these assumptions explicit and testable is itself an act of making conditions visible.
- **RQ5 (sensing and stewarding):** Sense is infrastructure for sensing. Evals verify that the sensing infrastructure performs as intended. Unreliable sensing degrades the capacity it claims to support.

## Scope

### In Scope

- Unit tests for all pure functions (no API calls, no external DB)
- Integration tests against a fixture database with pre-computed embeddings
- Gold-standard query-result pairs for retrieval quality regression
- Test infrastructure: fixtures, conftest, pytest configuration

### Out of Scope

- Hook relevance evaluation (how useful are ambient results in context)
- Mode-aware reranking calibration (are the multiplier values optimal)
- Performance benchmarks (latency, throughput)
- End-to-end tests requiring a running MCP server

---

## Architecture Decision

### ADR-001: Pre-Computed Fixture Embeddings

**Decision:** Use pre-computed embeddings stored in a SQLite fixture database rather than mocking the OpenAI API.

**Context:** Tests need embedding vectors to exercise search functions. Three options:

1. **Mock OpenAI API** — Return random or zero vectors. Fast but tests can't verify retrieval quality because similarity scores are meaningless.
2. **Live API calls** — Real embeddings, real quality. Slow (~300ms per call), costs money, flaky in CI, non-deterministic.
3. **Pre-computed fixtures** — Embed a small test corpus once, store vectors in a fixture DB. Tests load the fixture DB. Deterministic, fast, free, and similarity scores are meaningful.

**Trade-off:** Fixture embeddings are frozen in time. If OpenAI changes the embedding model, fixtures need regeneration. Acceptable because: (a) model changes are rare and announced, (b) regeneration is a single script run, (c) the alternative (mocks) gives up retrieval quality testing entirely.

**Consequence:** A fixture generation script (`tests/generate_fixtures.py`) computes embeddings for the test corpus and writes them to `tests/fixtures/sense_test.db`. This script is run manually when the test corpus or embedding model changes. The fixture DB is committed to the repo.

---

## Purity Boundary Map

### Pure Core (no I/O, no shared state, deterministic)

| Function | Module | What it computes |
|----------|--------|-----------------|
| `compute_decay()` | server | Temporal decay multiplier from source type + date |
| `cosine_similarity()` | server | Cosine similarity between two vectors |
| `count_tokens()` | server | Token count for a string |
| `_truncate_for_embedding()` | server | Truncated text within token limit |
| `chunk_file()` | server | File content split into chunks |
| `_split_paragraphs()` | server | Paragraph-based text splitting |
| `extract_date()` | server | Date extracted from filename or content |
| `is_evergreen()` | server/config | Whether a source type has no decay |
| `assemble_diverse_results()` | server | Ranked results split into diversity slots |
| `get_surfaced_penalty()` | server | Resurfacing penalty multiplier |
| `detect_circling_topics()` | server | Files from semantically similar prior queries |
| `_deep_merge()` | config | Recursive dict merge |
| `_build_matcher()` | config | Classification rule compilation |
| `SenseConfig.classify_source_type()` | config | File classification via rules |

### Effectful Shell (I/O, external calls, state)

| Function | Module | Effects |
|----------|--------|---------|
| `embed_texts()` / `embed_query()` | server | OpenAI API call |
| `get_db()` / `_init_db()` | server | SQLite connection |
| `sync_corpus()` | server | Filesystem walk + DB write + API calls |
| `search_chunks()` | server | DB read + vector computation |
| `search_chunks_contextual()` | server | DB read + mode detection + session state mutation |
| `discover_corpus()` | server | Filesystem walk |
| `sense_search()` / `sense_sync()` / `sense_status()` | server | MCP tool handlers |
| `detect_current_mode()` | server | File read (mode-history.jsonl) |
| `record_surfaced()` / `record_query()` | server | In-memory state mutation |

### Boundary Contracts

| Type | Direction |
|------|-----------|
| `list[dict]` (chunk descriptors) | corpus discovery -> chunking -> embedding -> DB |
| `np.ndarray` (float32, 1536-dim) | embedding API -> DB storage -> search comparison |
| `list[dict]` (search results) | DB read -> scoring -> diversity assembly -> output formatting |
| `SenseConfig` | config file -> all modules |

### Dependency Rule

Dependencies point inward: effectful shell calls pure core. Pure core does not import from shell.

**Exception:** `compute_decay()` reads `cfg.get_half_life()` and `cfg.decay_floor` via the config singleton. For testing, use `reload_config()` with a test config or patch `cfg` directly.

---

## Requirements

### REQ-001: Pure Function Unit Tests — Temporal Decay

The test suite SHALL verify `compute_decay()` for the following scenarios:

1. Evergreen source types (no half-life) return 1.0 regardless of date
2. Content dated today returns 1.0
3. Content at exactly one half-life returns ~0.5
4. Content at two half-lives returns ~0.25
5. Very old content floors at `decay_floor` (0.1)
6. Missing date returns `decay_floor`
7. Invalid date string returns `decay_floor`
8. Future-dated content is clamped to now (returns 1.0, not >1.0)

Trace: TEST-001

### REQ-002: Pure Function Unit Tests — Chunking

The test suite SHALL verify `chunk_file()` and `_split_paragraphs()` for:

1. Files under 512 tokens return a single chunk
2. Markdown files with `## ` headers split on headers
3. Section headers are preserved in chunk content
4. Sections exceeding 1024 tokens are sub-split by paragraph
5. Paragraph merging respects max_tokens boundary
6. Empty/whitespace-only paragraphs are excluded
7. Non-markdown files fall through to paragraph splitting
8. Section metadata propagates through sub-splits

Trace: TEST-002

### REQ-003: Pure Function Unit Tests — Similarity and Scoring

The test suite SHALL verify:

1. `cosine_similarity()` returns 1.0 for identical vectors
2. `cosine_similarity()` returns 0.0 for orthogonal vectors
3. `cosine_similarity()` handles zero vectors without error (returns 0.0)
4. `count_tokens()` returns correct count for known strings
5. `_truncate_for_embedding()` truncates text exceeding token limit
6. `_truncate_for_embedding()` passes through text within limit unchanged

Trace: TEST-003

### REQ-004: Pure Function Unit Tests — Classification

The test suite SHALL verify `classify_source_type()` via config rules:

1. Files matching `^TRACE_` or `DEV_UPDATE` filename pattern classify as `trace`
2. Files named `CLAUDE.md` classify as `project_claude`
3. `.py` files classify as `code`
4. Files matching no rule classify as the default (`documentation`)
5. Custom rules in test config are applied correctly
6. Case insensitivity works as configured

Trace: TEST-004

### REQ-005: Pure Function Unit Tests — Diversity Injection

The test suite SHALL verify `assemble_diverse_results()` for:

1. Results are split into confirmation/divergence/serendipity pools
2. Confirmation slots are filled first from top-ranked results
3. Divergence slots prefer different source_type or project from confirmation set
4. Serendipity slots prefer projects not seen in confirmation or divergence
5. Results below `min_similarity` are excluded
6. When insufficient candidates exist for a pool, remaining slots are filled from overflow
7. `slot_type` field is correctly assigned to each result
8. Dedup by file_path across pools (no duplicate files in output)

Trace: TEST-005

### REQ-006: Pure Function Unit Tests — Session Tracking

The test suite SHALL verify:

1. `get_surfaced_penalty()` returns 1.0 for unseen files
2. `get_surfaced_penalty()` returns `penalty^count` for previously surfaced files
3. Penalty is floored at 0.05
4. `detect_circling_topics()` returns files from prior queries above similarity threshold
5. `detect_circling_topics()` returns empty set when no prior queries exist
6. `record_surfaced()` caps `_session_surfaced` at `surfaced_cap`

Trace: TEST-006

### REQ-007: Pure Function Unit Tests — Configuration

The test suite SHALL verify:

1. `_deep_merge()` recursively merges dicts with override winning at leaves
2. `SenseConfig` loads from a test TOML file
3. `reload_config()` replaces the global singleton
4. Classification rules compile and match correctly
5. `is_evergreen()` returns True for types not in `half_lives`
6. `get_half_life()` returns correct values from config

Trace: TEST-007

### REQ-008: Integration Tests — Flat Retrieval Quality

The test suite SHALL verify `search_chunks()` against a fixture database:

1. A query semantically related to a known fixture document returns that document in the top 3
2. Project filter restricts results to the specified project
3. Source type filter restricts results to the specified type
4. Results are ordered by score (similarity * decay) descending
5. Limit parameter caps the number of returned results

Trace: TEST-008

### REQ-009: Integration Tests — Mode-Aware Retrieval

The test suite SHALL verify `search_chunks_contextual()` against a fixture database:

1. With no mode active, falls through to flat search (identical results to `search_chunks`)
2. With mode="none", bypasses mode detection and returns flat results
3. In build mode, code source_type results are boosted (multiplier 1.5)
4. In explore mode, cross-project results are boosted (multiplier 1.4)
5. In cool-off mode, all source types are suppressed (multipliers < 1.0)
6. Metadata dict contains mode, diversity_profile, and slot counts
7. Session tracking accumulates across multiple calls within the same process

Trace: TEST-009

### REQ-010: Gold-Standard Retrieval Pairs

The test suite SHALL include a minimum of 8 gold-standard query-result pairs that verify retrieval quality:

1. Each pair specifies: query text, expected file(s) in top N, expected source_type
2. Pairs cover at least 4 different source_types (trace, documentation, code, research)
3. Pairs cover at least 3 different projects
4. At least 2 pairs test cross-project retrieval (query about topic X surfaces results from project Y)
5. Pairs are documented with rationale (why this query should surface this result)
6. Test asserts the expected file appears in top 3 results (not exact rank)

Trace: TEST-010

---

## Non-Functional Requirements

### NFR-001: Test Execution Speed

Pure function tests SHALL complete in under 2 seconds total. Integration tests (fixture DB) SHALL complete in under 5 seconds total. No network calls during test execution.

Trace: TEST-011

### NFR-002: Test Independence

Each test SHALL be independent and order-insensitive. Session state (`_session_surfaced`, `_session_queries`) SHALL be reset between tests that exercise session tracking.

Trace: TEST-012

### NFR-003: Fixture Reproducibility

The fixture generation script SHALL be deterministic given the same test corpus and embedding model. The fixture DB SHALL be committed to the repository so tests run without API keys.

Trace: TEST-013

---

## Test Specifications

### TEST-001 through TEST-007: Pure Function Tests

**File:** `tests/test_pure.py`

Tests for REQ-001 through REQ-007. No external dependencies. May require patching `cfg` (the config singleton) for decay and classification tests with controlled values.

### TEST-008 through TEST-009: Integration Tests

**File:** `tests/test_integration.py`

Tests for REQ-008 and REQ-009. Load fixture DB from `tests/fixtures/sense_test.db`. Patch `get_db()` to return a connection to the fixture DB. Patch `embed_query()` to return pre-computed query embeddings from the fixture.

### TEST-010: Gold-Standard Pairs

**File:** `tests/test_gold_standard.py`

Tests for REQ-010. Same fixture infrastructure as integration tests. Each test case is a parameterised tuple of (query, expected_files, expected_source_type, rationale).

### TEST-011 through TEST-013: Non-Functional

Verified by pytest timing output (NFR-001), test isolation fixtures (NFR-002), and fixture generation script idempotency (NFR-003).

---

## Test Corpus Design

The fixture database is built from a small, representative test corpus stored in `tests/fixtures/corpus/`. This corpus is synthetic (not a copy of the live ecosystem) to keep it small and stable.

**Minimum corpus:**

| File | Project | Source Type | Purpose |
|------|---------|-------------|---------|
| `project-a/CLAUDE.md` | project-a | project_claude | Classification + retrieval |
| `project-a/server.py` | project-a | code | Code retrieval |
| `project-a/docs/overview.md` | project-a | documentation | Doc retrieval |
| `project-b/TRACE_2026-01-15_session.md` | project-b | trace | Trace retrieval + decay |
| `project-b/TRACE_2026-03-01_recent.md` | project-b | trace | Recent trace (low decay) |
| `project-b/research/findings.md` | project-b | research | Research retrieval |
| `project-c/README.md` | project-c | documentation | Cross-project retrieval |
| `project-c/analysis.py` | project-c | code | Cross-project code |
| `root-file.md` | root | documentation | Root project handling |

Content in each fixture file should be thematically distinct so that semantic similarity tests are meaningful. Suggested themes: coherence/entrainment (project-a), teaching/pedagogy (project-b), cooperative economics (project-c).

---

## Fixture Generation

**Script:** `tests/generate_fixtures.py`

1. Read all files from `tests/fixtures/corpus/`
2. Chunk each file using `chunk_file()`
3. Embed all chunks via OpenAI API (one-time cost, ~$0.001)
4. Write chunks + embeddings to `tests/fixtures/sense_test.db`
5. Pre-compute and store query embeddings for all gold-standard queries
6. Store query embeddings in a separate table: `query_fixtures(id, query_text, embedding)`

**Regeneration trigger:** Run manually when test corpus content changes or embedding model is updated.

---

## File Structure

```
sense-mcp/
  specs/
    SPEC-001-retrieval-evals.md    # This spec
  tests/
    conftest.py                     # Shared fixtures, DB patching, session reset
    test_pure.py                    # REQ-001 through REQ-007
    test_integration.py             # REQ-008 through REQ-009
    test_gold_standard.py           # REQ-010
    generate_fixtures.py            # One-time fixture generation script
    fixtures/
      corpus/                       # Synthetic test corpus files
        project-a/
        project-b/
        project-c/
        root-file.md
      sense_test.db                 # Pre-computed fixture database
      test_config.toml              # Test-specific Sense config
```

---

## Implementation Notes

- Use `reload_config()` in test setup to load `test_config.toml` instead of the live config
- Session state globals (`_session_surfaced`, `_session_queries`) need explicit reset in fixtures
- `extract_date()` tests need temp files on disk (it reads `st_mtime`); use `tmp_path` fixture
- For decay tests, freeze time or compute expected values relative to test execution date
- Gold-standard pairs will need tuning after fixture generation. Start with obvious semantic matches, then refine based on actual similarity scores.
