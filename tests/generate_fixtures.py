"""
tests/generate_fixtures.py — Build the fixture database for Sense evaluation tests.

Reads all corpus files from tests/fixtures/corpus/, chunks and embeds them using
Sense's own pipeline, and writes to tests/fixtures/sense_test.db.

Also defines and embeds gold-standard evaluation queries, storing them in a
query_fixtures table alongside expected retrieval targets.

Usage:
    python tests/generate_fixtures.py

Requires OPENAI_API_KEY in the environment.
"""

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Locate fixture paths relative to this file
TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
CORPUS_DIR = FIXTURES_DIR / "corpus"
TEST_CONFIG_PATH = FIXTURES_DIR / "test_config.toml"
OUTPUT_DB_PATH = FIXTURES_DIR / "sense_test.db"

# Set env vars before importing sense_mcp so the module-level cfg singleton
# loads the test config and resolves root to the fixture corpus directory.
os.environ["SENSE_CONFIG"] = str(TEST_CONFIG_PATH)
os.environ["SENSE_ROOT"] = str(CORPUS_DIR)

import sense_mcp.config as config_module  # noqa: E402
from sense_mcp.config import reload_config  # noqa: E402
from sense_mcp.server import (  # noqa: E402
    _init_db,
    chunk_file,
    classify_source_type,
    embed_texts,
    extract_date,
    file_hash,
)


# ---------------------------------------------------------------------------
# Gold-standard query definitions
#
# Covers 4 source types: documentation, code, trace, project_claude
# Covers 4 projects:     project-a, project-b, project-c, root
#
# expected_files: paths relative to the corpus root (tests/fixtures/corpus/)
# ---------------------------------------------------------------------------

GOLD_QUERIES = [
    {
        "query_text": "How does entrainment work in distributed sociotechnical systems?",
        "expected_files": ["project-a/docs/overview.md"],
        "expected_source_type": "documentation",
        "rationale": (
            "Technical documentation about entrainment phenomena is the primary topic "
            "of project-a/docs/overview.md. Tests documentation retrieval for project-a."
        ),
    },
    {
        "query_text": "Python API for detecting rhythm synchronisation signatures in interaction data",
        "expected_files": ["project-a/server.py"],
        "expected_source_type": "code",
        "rationale": (
            "project-a/server.py implements the entrainment analysis pipeline. "
            "Tests code retrieval across project-a."
        ),
    },
    {
        "query_text": "MBA 915 teaching session observations and student engagement",
        "expected_files": [
            "project-b/TRACE_2026-01-15.md",
            "project-b/TRACE_2026-02-01.md",
        ],
        "expected_source_type": "trace",
        "rationale": (
            "Both TRACE files in project-b record MBA 915 weekly teaching sessions. "
            "Tests trace retrieval for project-b."
        ),
    },
    {
        "query_text": "Worker cooperative ownership structures and collective intelligence",
        "expected_files": ["project-c/README.md"],
        "expected_source_type": "documentation",
        "rationale": (
            "project-c/README.md is the main documentation on cooperative economics. "
            "Tests documentation retrieval for project-c."
        ),
    },
    {
        "query_text": "Project identity, research goals, and current focus for coherence and entrainment research",
        "expected_files": ["project-a/CLAUDE.md"],
        "expected_source_type": "project_claude",
        "rationale": (
            "project-a/CLAUDE.md describes the project identity and research focus. "
            "Tests project_claude source type retrieval for project-a."
        ),
    },
    {
        "query_text": "Relational conditions for learning, attunement as prerequisite for effective pedagogy",
        "expected_files": ["project-b/research/findings.md"],
        "expected_source_type": "documentation",
        "rationale": (
            "project-b/research/findings.md documents findings on relational attunement in pedagogy. "
            "Tests documentation retrieval for a subdirectory file in project-b."
        ),
    },
    {
        "query_text": "Computational tools for analysing cooperative membership and participation data",
        "expected_files": ["project-c/analysis.py"],
        "expected_source_type": "code",
        "rationale": (
            "project-c/analysis.py provides the computational analysis tools. "
            "Tests code retrieval for project-c."
        ),
    },
    {
        "query_text": "Overview of ecosystem connecting multiple research projects on adaptive capacity",
        "expected_files": ["root-file.md"],
        "expected_source_type": "documentation",
        "rationale": (
            "root-file.md is the top-level ecosystem overview. "
            "Tests documentation retrieval for a root-level (project=root) file."
        ),
    },
    {
        "query_text": "Session trace from January 2025 fixture",
        "expected_files": ["TRACE_2025-01-01_sample-session.md"],
        "expected_source_type": "trace",
        "rationale": (
            "TRACE_2025-01-01_sample-session.md is the root-level trace fixture. "
            "Tests trace retrieval for a root-level file (project=root)."
        ),
    },
    {
        "query_text": "Solidarity economics, mutual aid networks resilience under resource scarcity",
        "expected_files": ["project-c/README.md"],
        "expected_source_type": "documentation",
        "rationale": (
            "Solidarity economics and resilience is a key section in project-c/README.md. "
            "Tests retrieval of a specific thematic section within project-c documentation."
        ),
    },
    {
        "query_text": "Root CLAUDE.md project configuration and instructions",
        "expected_files": ["CLAUDE.md"],
        "expected_source_type": "project_claude",
        "rationale": (
            "The root-level CLAUDE.md is classified as project_claude. "
            "Tests project_claude retrieval for the root project."
        ),
    },
    {
        "query_text": "Elinor Ostrom commons governance design principles for shared resources",
        "expected_files": ["project-c/README.md"],
        "expected_source_type": "documentation",
        "rationale": (
            "project-c/README.md includes a section on Ostrom's commons governance principles. "
            "Tests retrieval of academic concept coverage within a documentation file."
        ),
    },
]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def create_fixture_db(db_path: Path) -> sqlite3.Connection:
    """Create (or overwrite) the fixture DB with production + query_fixtures schemas."""
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    _init_db(conn)  # production schema: chunks + sync_meta
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS query_fixtures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            expected_files TEXT NOT NULL,
            expected_source_type TEXT NOT NULL,
            rationale TEXT NOT NULL
        );
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Corpus ingestion
# ---------------------------------------------------------------------------

def discover_and_chunk(cfg) -> list[dict]:
    """Walk the corpus, read and chunk all eligible files.

    Returns a list of chunk record dicts ready for embedding and DB insertion.
    """
    extensions = cfg.extensions
    excluded_dirs = cfg.excluded_dirs
    root = cfg.root

    chunk_records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in extensions:
            continue

        skip = False
        for part in path.relative_to(root).parts:
            if part in excluded_dirs:
                skip = True
                break
        if skip:
            continue

        source_type = classify_source_type(path)

        rel = path.relative_to(root)
        parts = rel.parts
        project = "root" if len(parts) <= 1 else parts[0]

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"  WARNING: could not read {path}: {exc}", file=sys.stderr)
            continue

        date = extract_date(path, content)
        fhash = file_hash(path)
        chunks = chunk_file(path, content)
        evergreen = cfg.is_evergreen(source_type)

        for chunk in chunks:
            if not chunk["content"].strip():
                continue
            chunk_records.append({
                "file_path": str(path),
                "file_hash": fhash,
                "project": project,
                "source_type": source_type,
                "section": chunk["section"],
                "date": date,
                "evergreen": 1 if evergreen else 0,
                "content": chunk["content"],
                "token_count": chunk["token_count"],
            })

    return chunk_records


def insert_chunks(conn: sqlite3.Connection, records: list[dict], embeddings) -> int:
    """Insert chunk records with their embeddings. Returns number inserted."""
    now = datetime.now(timezone.utc).isoformat()
    for chunk, emb in zip(records, embeddings):
        conn.execute(
            """INSERT INTO chunks
               (file_path, file_hash, project, source_type, section, date,
                evergreen, content, token_count, embedding, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                chunk["file_path"], chunk["file_hash"], chunk["project"],
                chunk["source_type"], chunk["section"], chunk["date"],
                chunk["evergreen"], chunk["content"], chunk["token_count"],
                emb.tobytes(), now,
            ),
        )
    conn.commit()
    return len(records)


def insert_queries(conn: sqlite3.Connection, queries: list[dict], embeddings) -> int:
    """Insert query fixtures with their embeddings. Returns number inserted."""
    for q, emb in zip(queries, embeddings):
        conn.execute(
            """INSERT INTO query_fixtures
               (query_text, embedding, expected_files, expected_source_type, rationale)
               VALUES (?, ?, ?, ?, ?)""",
            (
                q["query_text"],
                emb.tobytes(),
                json.dumps(q["expected_files"]),
                q["expected_source_type"],
                q["rationale"],
            ),
        )
    conn.commit()
    return len(queries)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Sense Fixture Generator ===")
    print(f"Corpus : {CORPUS_DIR}")
    print(f"Config : {TEST_CONFIG_PATH}")
    print(f"Output : {OUTPUT_DB_PATH}")
    print()

    # Reload config so our env vars are honoured (in case the module singleton
    # was already set by a prior import).
    config_module._config = None
    cfg = reload_config(TEST_CONFIG_PATH)
    # Ensure root is pinned to the fixture corpus regardless of config resolution.
    cfg.root = CORPUS_DIR

    # Step 1: discover + chunk
    print("Discovering and chunking corpus files...")
    chunk_records = discover_and_chunk(cfg)
    print(f"  {len(chunk_records)} chunks from corpus")

    # Step 2: initialise DB
    print("Initialising fixture DB...")
    conn = create_fixture_db(OUTPUT_DB_PATH)

    # Step 3: embed chunks
    print("Embedding corpus chunks (calling OpenAI)...")
    chunk_texts = [c["content"] for c in chunk_records]
    chunk_embeddings = embed_texts(chunk_texts)
    print(f"  {len(chunk_embeddings)} chunk embeddings received")

    # Step 4: insert chunks
    n_chunks = insert_chunks(conn, chunk_records, chunk_embeddings)
    print(f"  {n_chunks} chunks inserted into chunks table")

    # Step 5: embed queries
    print(f"Embedding {len(GOLD_QUERIES)} gold-standard queries (calling OpenAI)...")
    query_texts = [q["query_text"] for q in GOLD_QUERIES]
    query_embeddings = embed_texts(query_texts)
    print(f"  {len(query_embeddings)} query embeddings received")

    # Step 6: insert queries
    n_queries = insert_queries(conn, GOLD_QUERIES, query_embeddings)
    print(f"  {n_queries} queries inserted into query_fixtures table")

    # Summary
    print()
    print("=== Summary ===")

    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"Chunks indexed : {total_chunks}")

    by_type = conn.execute(
        "SELECT source_type, COUNT(*) FROM chunks GROUP BY source_type ORDER BY source_type"
    ).fetchall()
    print("  By source type:")
    for stype, count in by_type:
        print(f"    {stype}: {count}")

    by_project = conn.execute(
        "SELECT project, COUNT(*) FROM chunks GROUP BY project ORDER BY project"
    ).fetchall()
    print("  By project:")
    for proj, count in by_project:
        print(f"    {proj}: {count}")

    total_queries = conn.execute("SELECT COUNT(*) FROM query_fixtures").fetchone()[0]
    print(f"Query fixtures : {total_queries}")

    source_types_covered = sorted({q["expected_source_type"] for q in GOLD_QUERIES})
    projects_covered = set()
    for q in GOLD_QUERIES:
        for f in q["expected_files"]:
            parts = Path(f).parts
            projects_covered.add(parts[0] if len(parts) > 1 else "root")
    print(f"  Source types covered : {source_types_covered} ({len(source_types_covered)})")
    print(f"  Projects covered     : {sorted(projects_covered)} ({len(projects_covered)})")

    conn.close()
    print()
    print(f"Done. Fixture DB written to {OUTPUT_DB_PATH}")


if __name__ == "__main__":
    main()
