"""Sense — Semantic Retrieval MCP Server.

Indexes a markdown-heavy project ecosystem and makes cross-project connections
queryable by semantic similarity with temporal decay. Supports mode-aware
retrieval when paired with Vibe Harness.

Configuration is loaded from sense.toml (see sense.example.toml).
"""

import hashlib
import json
import math
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tiktoken
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

from config import get_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

cfg = get_config()

load_dotenv(cfg.env_file)

OPENAI_API_KEY = os.getenv(cfg.api_key_env)

# Backward-compatible aliases (used by the auto-query hook)
ECOSYSTEM_ROOT = cfg.root
DB_PATH = cfg.db_path

# Tokeniser for counting
_enc = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Lazy-init singletons
# ---------------------------------------------------------------------------

_openai_client = None
_db_conn = None


def get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(f"Missing {cfg.api_key_env} — set it in {cfg.env_file} or your environment")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def get_db() -> sqlite3.Connection:
    global _db_conn
    if _db_conn is None:
        _db_conn = sqlite3.connect(str(DB_PATH))
        _db_conn.execute("PRAGMA journal_mode=WAL")
        _init_db(_db_conn)
    return _db_conn


def _init_db(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            project TEXT NOT NULL,
            source_type TEXT NOT NULL,
            section TEXT,
            date TEXT,
            evergreen INTEGER NOT NULL DEFAULT 0,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
        CREATE INDEX IF NOT EXISTS idx_chunks_project ON chunks(project);
        CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(source_type);

        CREATE TABLE IF NOT EXISTS sync_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Corpus discovery
# ---------------------------------------------------------------------------

def classify_source_type(path: Path) -> str:
    """Classify a file into a source type using config-driven rules."""
    return cfg.classify_source_type(path)


def classify_project(path: Path) -> str:
    """Extract project name from file path."""
    rel = path.relative_to(cfg.root)
    parts = rel.parts
    if len(parts) <= 1:
        return "root"
    return parts[0]


def extract_date(path: Path, content: str) -> str | None:
    """Try to extract a date from filename or content."""
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    if date_match:
        return date_match.group(1)
    date_match = re.search(r"date:\s*(\d{4}-\d{2}-\d{2})", content[:500])
    if date_match:
        return date_match.group(1)
    try:
        mtime = path.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d")
    except OSError:
        return None


def is_evergreen(source_type: str) -> bool:
    return cfg.is_evergreen(source_type)


def detect_current_mode() -> str | None:
    """Read the current Vibe Harness mode from mode-history.jsonl.

    Returns the most recent mode name, or None if unavailable.
    Graceful — never raises.
    """
    try:
        history_path = cfg.mode_history_path
        if not history_path.exists():
            return None
        with open(history_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 512))
            tail = f.read().decode("utf-8", errors="replace")
        lines = [ln for ln in tail.strip().splitlines() if ln.strip()]
        if not lines:
            return None
        entry = json.loads(lines[-1])
        mode = entry.get("to_mode")
        if mode and mode in cfg.mode_profiles:
            return mode
        return None
    except Exception:
        return None


def discover_corpus() -> list[dict]:
    """Walk the corpus root and return a list of file descriptors."""
    root = cfg.root
    extensions = cfg.extensions
    excluded_dirs = cfg.excluded_dirs
    excluded_paths = cfg.excluded_paths

    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in extensions:
            continue
        if path in excluded_paths:
            continue

        skip = False
        for part in path.relative_to(root).parts:
            if part in excluded_dirs:
                skip = True
                break
        if skip:
            continue

        source_type = classify_source_type(path)
        project = classify_project(path)

        files.append({
            "path": path,
            "source_type": source_type,
            "project": project,
            "evergreen": is_evergreen(source_type),
        })
    return files


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def chunk_file(path: Path, content: str) -> list[dict]:
    """Split a file into chunks. Returns list of {section, content, token_count}."""
    total_tokens = count_tokens(content)

    if total_tokens <= 512:
        return [{"section": None, "content": content, "token_count": total_tokens}]

    if path.suffix in (".md", ".txt", ".rst"):
        sections = re.split(r"(?m)^## ", content)
        if len(sections) > 1:
            chunks = []
            if sections[0].strip():
                intro = sections[0].strip()
                tc = count_tokens(intro)
                if tc > 0:
                    chunks.append({"section": "intro", "content": intro, "token_count": tc})
            for section in sections[1:]:
                lines = section.split("\n", 1)
                header = lines[0].strip()
                body = lines[1].strip() if len(lines) > 1 else ""
                if not body:
                    continue
                full = f"## {header}\n\n{body}"
                tc = count_tokens(full)
                if tc > 1024:
                    for sub in _split_paragraphs(full, max_tokens=512, section=header):
                        chunks.append(sub)
                elif tc > 0:
                    chunks.append({"section": header, "content": full, "token_count": tc})
            if chunks:
                return chunks

    return _split_paragraphs(content, max_tokens=512)


def _split_paragraphs(text: str, max_tokens: int = 512, section: str | None = None) -> list[dict]:
    """Split text by double-newlines, merge small paragraphs, split large ones.

    Args:
        section: Optional section header to propagate to sub-chunks (preserves
                 metadata when large ## sections are sub-split).
    """
    paragraphs = re.split(r"\n\n+", text)
    chunks = []
    current = ""
    current_tokens = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        pt = count_tokens(para)

        if pt > max_tokens:
            if current.strip():
                chunks.append({"section": section, "content": current.strip(), "token_count": current_tokens})
                current = ""
                current_tokens = 0
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf = ""
            buf_t = 0
            for sent in sentences:
                st = count_tokens(sent)
                if buf_t + st > max_tokens and buf:
                    chunks.append({"section": section, "content": buf.strip(), "token_count": buf_t})
                    buf = sent
                    buf_t = st
                else:
                    buf = (buf + " " + sent).strip()
                    buf_t += st
            if buf.strip():
                chunks.append({"section": section, "content": buf.strip(), "token_count": buf_t})
        elif current_tokens + pt > max_tokens:
            if current.strip():
                chunks.append({"section": section, "content": current.strip(), "token_count": current_tokens})
            current = para
            current_tokens = pt
        else:
            current = (current + "\n\n" + para).strip()
            current_tokens += pt

    if current.strip():
        chunks.append({"section": section, "content": current.strip(), "token_count": current_tokens})

    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _truncate_for_embedding(text: str) -> str:
    """Truncate text to fit within embedding model token limit."""
    tokens = _enc.encode(text)
    if len(tokens) <= cfg.max_input_tokens:
        return text
    return _enc.decode(tokens[:cfg.max_input_tokens])


def embed_texts(texts: list[str]) -> list[np.ndarray]:
    """Batch embed texts via OpenAI API. Returns list of numpy arrays."""
    if not texts:
        return []
    client = get_openai()
    texts = [_truncate_for_embedding(t) if t.strip() else " " for t in texts]
    all_embeddings = []
    batch_size = cfg.batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=cfg.embedding_model, input=batch)
        for item in resp.data:
            all_embeddings.append(np.array(item.embedding, dtype=np.float32))
    return all_embeddings


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string."""
    return embed_texts([text])[0]


# ---------------------------------------------------------------------------
# Temporal decay
# ---------------------------------------------------------------------------

def compute_decay(source_type: str, date_str: str | None) -> float:
    """Compute temporal decay multiplier for a chunk."""
    half_life = cfg.get_half_life(source_type)
    if half_life is None:
        return 1.0  # Evergreen

    if not date_str:
        return cfg.decay_floor  # No date = assume old

    try:
        doc_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return cfg.decay_floor

    now = datetime.now(timezone.utc)
    # Clamp to now — future-dated frontmatter treated as "just published",
    # not as a persistent 1.0 override that never decays.
    doc_date = min(doc_date, now)
    age_days = (now - doc_date).days
    if age_days <= 0:
        return 1.0

    decay = math.pow(0.5, age_days / half_life)
    return max(decay, cfg.decay_floor)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


# ---------------------------------------------------------------------------
# Session trajectory tracking (MCP-scoped, in-memory)
#
# These globals only accumulate state within the long-running MCP server
# process. The auto-query hook (sense-auto-query.py) imports this module
# in a fresh process per invocation, so hook-based searches always start
# with empty state — session tracking is structurally inoperative for
# ambient (hook-based) queries. The hook maintains its own lightweight
# file-based dedup in /tmp/sense-auto-query-{session_id}.json.
# ---------------------------------------------------------------------------

# {file_path: {"count": int, "last_ts": float}}
_session_surfaced: dict[str, dict] = {}

# [{"query": str, "embedding": np.ndarray, "ts": float, "surfaced_files": list[str]}]
_session_queries: list[dict] = []


def record_surfaced(results: list[dict]) -> None:
    """Log which files were surfaced in a result set."""
    now = time.time()
    for r in results:
        fp = r["file_path"]
        if fp in _session_surfaced:
            _session_surfaced[fp]["count"] += 1
            _session_surfaced[fp]["last_ts"] = now
        else:
            _session_surfaced[fp] = {"count": 1, "last_ts": now}

    cap = cfg.surfaced_cap
    if len(_session_surfaced) > cap:
        by_ts = sorted(_session_surfaced.items(), key=lambda kv: kv[1]["last_ts"])
        to_remove = len(_session_surfaced) - cap
        for key, _ in by_ts[:to_remove]:
            del _session_surfaced[key]


def record_query(query: str, embedding: np.ndarray, surfaced_files: list[str]) -> None:
    """Log a query embedding for circling detection."""
    _session_queries.append({
        "query": query,
        "embedding": embedding,
        "ts": time.time(),
        "surfaced_files": surfaced_files,
    })
    max_q = cfg.max_queries
    if len(_session_queries) > max_q:
        _session_queries.pop(0)


def get_surfaced_penalty(file_path: str, base_penalty: float) -> float:
    """Return a multiplier that decays with each resurfacing.

    penalty^count, floored at 0.05 so nothing vanishes entirely.
    """
    entry = _session_surfaced.get(file_path)
    if not entry:
        return 1.0
    return max(base_penalty ** entry["count"], 0.05)


def detect_circling_topics(embedding: np.ndarray, threshold: float = 0.75) -> set[str]:
    """Find files from past queries that are semantically similar to this one.

    Returns file paths that appeared in results of similar prior queries —
    these are circling topics (salience signal, worth boosting).
    """
    circling_files: set[str] = set()
    for past in _session_queries:
        sim = cosine_similarity(embedding, past["embedding"])
        if sim >= threshold:
            circling_files.update(past["surfaced_files"])
    return circling_files


def search_chunks(
    query_embedding: np.ndarray,
    project: str | None = None,
    source_type: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Search all chunks by cosine similarity with temporal decay."""
    conn = get_db()

    sql = "SELECT id, file_path, project, source_type, section, date, evergreen, content, token_count, embedding FROM chunks"
    conditions = []
    params = []
    if project:
        conditions.append("project = ?")
        params.append(project)
    if source_type:
        conditions.append("source_type = ?")
        params.append(source_type)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    rows = conn.execute(sql, params).fetchall()

    results = []
    for row in rows:
        (chunk_id, file_path, proj, stype, section, date, evergreen, content, token_count, emb_blob) = row
        stored_emb = np.frombuffer(emb_blob, dtype=np.float32)
        sim = cosine_similarity(query_embedding, stored_emb)
        decay = compute_decay(stype, date)
        score = sim * decay

        results.append({
            "score": score,
            "similarity": sim,
            "decay": decay,
            "file_path": file_path,
            "project": proj,
            "source_type": stype,
            "section": section,
            "date": date,
            "content": content,
            "token_count": token_count,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]


# ---------------------------------------------------------------------------
# Diversity injection
# ---------------------------------------------------------------------------

def assemble_diverse_results(
    all_results: list[dict],
    slots: tuple[int, int, int],
    current_project: str | None,
    min_similarity: float = 0.20,
) -> list[dict]:
    """Split ranked results into confirmation / divergence / serendipity pools.

    slots: (confirmation_count, divergence_count, serendipity_count)
    Each result gets a 'slot_type' field.
    """
    n_confirm, n_diverge, n_serendip = slots

    viable = [r for r in all_results if r["similarity"] >= min_similarity]
    if not viable:
        return []

    if n_diverge == 0 and n_serendip == 0:
        for r in viable[:n_confirm]:
            r["slot_type"] = "confirmation"
        return viable[:n_confirm]

    confirmation = []
    used_paths = set()
    for r in viable:
        if len(confirmation) >= n_confirm:
            break
        if r["file_path"] in used_paths:
            continue
        confirmation.append(r)
        used_paths.add(r["file_path"])

    confirm_types = {r["source_type"] for r in confirmation}
    confirm_projects = {r["project"] for r in confirmation}

    divergence = []
    for r in viable:
        if r["file_path"] in used_paths:
            continue
        if len(divergence) >= n_diverge:
            break
        is_divergent = (
            r["source_type"] not in confirm_types
            or r["project"] not in confirm_projects
        )
        if is_divergent:
            divergence.append(r)
            used_paths.add(r["file_path"])

    all_seen_projects = confirm_projects | {r["project"] for r in divergence}
    if current_project:
        all_seen_projects.add(current_project)
    serendipity = []
    for r in viable:
        if r["file_path"] in used_paths:
            continue
        if len(serendipity) >= n_serendip:
            break
        if r["project"] not in all_seen_projects:
            serendipity.append(r)
            used_paths.add(r["file_path"])

    if len(serendipity) < n_serendip:
        for r in viable:
            if r["file_path"] in used_paths:
                continue
            if len(serendipity) >= n_serendip:
                break
            serendipity.append(r)
            used_paths.add(r["file_path"])

    for r in confirmation:
        r["slot_type"] = "confirmation"
    for r in divergence:
        r["slot_type"] = "divergence"
    for r in serendipity:
        r["slot_type"] = "serendipity"

    return confirmation + divergence + serendipity


# ---------------------------------------------------------------------------
# Contextual search pipeline (mode-aware)
# ---------------------------------------------------------------------------

def search_chunks_contextual(
    query_embedding: np.ndarray,
    query_text: str,
    project: str | None = None,
    source_type: str | None = None,
    mode: str | None = None,
    limit: int = 10,
) -> tuple[list[dict], dict]:
    """Mode-aware search with session tracking and diversity injection.

    Returns (results, metadata) where metadata contains mode info and indicators.
    Falls through to plain search_chunks when no mode is active.

    Note: session tracking (resurfacing penalty, circling detection) only
    accumulates state within the MCP server process. Hook-based callers
    that import this module directly get fresh state per invocation.
    """
    mode_profiles = cfg.mode_profiles
    diversity_slots = cfg.diversity_slots

    # Explicit "none" bypasses auto-detection → flat search
    if mode and mode.lower() in ("none", "flat"):
        results = search_chunks(query_embedding, project, source_type, limit)
        return results, {"mode": None}

    if not mode:
        mode = detect_current_mode()

    if not mode or mode not in mode_profiles:
        results = search_chunks(query_embedding, project, source_type, limit)
        return results, {"mode": None}

    profile = mode_profiles[mode]
    slots = diversity_slots[profile["diversity_profile"]]

    pool_size = limit * 5
    candidates = search_chunks(query_embedding, project, None, pool_size)

    if source_type:
        for r in candidates:
            if r["source_type"] == source_type:
                r["score"] *= 1.2

    multipliers = profile["source_type_multipliers"]
    for r in candidates:
        mult = multipliers.get(r["source_type"], 1.0)
        r["score"] *= mult
        r["mode_multiplier"] = mult

    cross_boost = profile["cross_project_boost"]
    for r in candidates:
        if project and r["project"] != project:
            r["score"] *= cross_boost
            r["cross_project"] = True
        else:
            r["cross_project"] = False

    base_penalty = profile["already_surfaced_penalty"]
    for r in candidates:
        penalty = get_surfaced_penalty(r["file_path"], base_penalty)
        if penalty < 1.0:
            r["score"] *= penalty
            r["resurfaced"] = True
            r["resurface_penalty"] = penalty
        else:
            r["resurfaced"] = False

    circling_files = detect_circling_topics(query_embedding)
    for r in candidates:
        if r["file_path"] in circling_files:
            r["score"] *= 1.3
            r["circling"] = True
        else:
            r["circling"] = False

    candidates.sort(key=lambda x: x["score"], reverse=True)

    results = assemble_diverse_results(candidates, slots, project)
    results = results[:limit]

    surfaced_files = [r["file_path"] for r in results]
    record_surfaced(results)
    record_query(query_text, query_embedding, surfaced_files)

    metadata = {
        "mode": mode,
        "diversity_profile": profile["diversity_profile"],
        "slots": slots,
        "circling_count": sum(1 for r in results if r.get("circling")),
        "resurfaced_count": sum(1 for r in results if r.get("resurfaced")),
        "session_queries": len(_session_queries),
    }

    return results, metadata


# ---------------------------------------------------------------------------
# Sync logic
# ---------------------------------------------------------------------------

def file_hash(path: Path) -> str:
    """SHA-256 of file contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sync_corpus() -> dict:
    """Discover corpus, embed new/changed files, remove deleted ones. Returns stats."""
    conn = get_db()
    corpus = discover_corpus()

    existing = {}
    for row in conn.execute("SELECT DISTINCT file_path, file_hash FROM chunks").fetchall():
        existing[row[0]] = row[1]

    current_paths = set()
    to_index = []
    unchanged = 0

    for desc in corpus:
        path = desc["path"]
        path_str = str(path)
        current_paths.add(path_str)

        try:
            h = file_hash(path)
        except OSError:
            continue

        if path_str in existing and existing[path_str] == h:
            unchanged += 1
            continue

        to_index.append((desc, h))

    removed_paths = set(existing.keys()) - current_paths
    changed_paths = {str(desc["path"]) for desc, _ in to_index}
    paths_to_delete = removed_paths | changed_paths

    deleted_count = 0
    for path_str in paths_to_delete:
        conn.execute("DELETE FROM chunks WHERE file_path = ?", (path_str,))
        deleted_count += 1
    if paths_to_delete:
        conn.commit()

    new_chunks = []
    for desc, h in to_index:
        path = desc["path"]
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        date = extract_date(path, content)
        chunks = chunk_file(path, content)

        for chunk in chunks:
            new_chunks.append({
                "file_path": str(path),
                "file_hash": h,
                "project": desc["project"],
                "source_type": desc["source_type"],
                "section": chunk["section"],
                "date": date,
                "evergreen": 1 if desc["evergreen"] else 0,
                "content": chunk["content"],
                "token_count": chunk["token_count"],
            })

    new_chunks = [c for c in new_chunks if c["content"].strip()]

    embedded_count = 0
    if new_chunks:
        texts = [c["content"] for c in new_chunks]
        embeddings = embed_texts(texts)
        now = datetime.now(timezone.utc).isoformat()

        for chunk, emb in zip(new_chunks, embeddings):
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
                )
            )
        conn.commit()
        embedded_count = len(new_chunks)

    conn.execute(
        "INSERT OR REPLACE INTO sync_meta (key, value) VALUES (?, ?)",
        ("last_sync", datetime.now(timezone.utc).isoformat())
    )
    conn.commit()

    return {
        "files_discovered": len(corpus),
        "files_unchanged": unchanged,
        "files_indexed": len(to_index),
        "chunks_embedded": embedded_count,
        "files_removed": len(removed_paths),
        "files_updated": len(changed_paths),
    }


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP(cfg.server_name)


@mcp.tool()
def sense_search(
    query: str,
    project: str = "",
    source_type: str = "",
    limit: int = 10,
    mode: str = "",
) -> str:
    """Search the indexed ecosystem by semantic similarity.

    Finds relevant content across traces, documentation, project CLAUDE.md
    files, reference material, and research notes. Results are ranked by
    cosine similarity with temporal decay (recent content scores higher).

    Args:
        query: Natural language search query.
        project: Optional project filter (e.g. 'somatic-ai-safety', 'teaching').
        source_type: Optional type filter: trace, documentation, project_claude, reference, research, teaching, code. In flat mode (no Vibe Harness), this is a hard SQL filter — only matching types returned. In mode-aware mode, this becomes a soft 1.2x score boost — matching types are preferred but other types can still appear in divergence/serendipity slots.
        limit: Max results to return (default 10).
        mode: Optional Vibe Harness mode override (explore, build, think-with, ship, cool-off, none). Auto-detected if omitted. Use 'none' to force flat cosine search regardless of Vibe Harness state.
    """
    try:
        conn = get_db()
        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if total == 0:
            return "Index is empty. Run sense_sync first to build the index."

        query_emb = embed_query(query)
        results, meta = search_chunks_contextual(
            query_emb,
            query_text=query,
            project=project or None,
            source_type=source_type or None,
            mode=mode or None,
            limit=limit,
        )

        if not results:
            return f"No results found for: {query}"

        root_str = str(cfg.root)
        active_mode = meta.get("mode")
        if active_mode:
            lines = [
                f"Found {len(results)} result(s) for: \"{query}\"",
                f"Mode: {active_mode} | Diversity: {meta['diversity_profile']} | Session queries: {meta['session_queries']}",
                "",
            ]
        else:
            lines = [f"Found {len(results)} result(s) for: \"{query}\"", ""]

        for i, r in enumerate(results, 1):
            rel_path = r["file_path"].replace(root_str + "/", "")
            section_label = f" > {r['section']}" if r["section"] else ""
            decay_label = f" (decay: {r['decay']:.2f})" if r["decay"] < 1.0 else ""

            slot_label = ""
            if r.get("slot_type"):
                slot_label = f" [{r['slot_type']}]"

            indicators = []
            if r.get("resurfaced"):
                indicators.append(f"resurfaced:{r.get('resurface_penalty', 0):.2f}")
            if r.get("circling"):
                indicators.append("circling")
            if r.get("cross_project"):
                indicators.append("cross-project")
            indicator_str = f" | {', '.join(indicators)}" if indicators else ""

            lines.append(f"### {i}. [{r['project']}] {rel_path}{section_label}{slot_label}")
            lines.append(f"Score: {r['score']:.4f} | Similarity: {r['similarity']:.4f}{decay_label} | Type: {r['source_type']} | Date: {r['date'] or 'unknown'}{indicator_str}")
            lines.append("")

            preview = r["content"][:300]
            if len(r["content"]) > 300:
                preview += "..."
            lines.append(preview)
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"Error searching: {e}"


@mcp.tool()
def sense_sync() -> str:
    """Rebuild or update the Sense search index.

    Walks the configured corpus root, discovers files, chunks them,
    and embeds new/changed content. Uses file hashing for change detection
    so unchanged files are skipped. Safe to run repeatedly.
    """
    try:
        stats = sync_corpus()
        lines = [
            "Sense sync complete.",
            "",
            f"Files discovered: {stats['files_discovered']}",
            f"Files unchanged (skipped): {stats['files_unchanged']}",
            f"Files indexed (new/updated): {stats['files_indexed']}",
            f"Chunks embedded: {stats['chunks_embedded']}",
            f"Files removed from index: {stats['files_removed']}",
            f"Files re-indexed (content changed): {stats['files_updated']}",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Error during sync: {e}"


@mcp.tool()
def sense_status() -> str:
    """Show current Sense index statistics.

    Returns chunk counts by project and source type, total token count,
    and last sync time.
    """
    try:
        conn = get_db()
        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

        if total == 0:
            return "Index is empty. Run sense_sync to build it."

        total_tokens = conn.execute("SELECT SUM(token_count) FROM chunks").fetchone()[0] or 0

        project_rows = conn.execute(
            "SELECT project, COUNT(*), SUM(token_count) FROM chunks GROUP BY project ORDER BY COUNT(*) DESC"
        ).fetchall()

        type_rows = conn.execute(
            "SELECT source_type, COUNT(*), SUM(token_count) FROM chunks GROUP BY source_type ORDER BY COUNT(*) DESC"
        ).fetchall()

        last_sync = conn.execute(
            "SELECT value FROM sync_meta WHERE key = 'last_sync'"
        ).fetchone()
        last_sync_str = last_sync[0] if last_sync else "never"

        lines = [
            "Sense Index Status",
            "",
            f"Total chunks: {total}",
            f"Total tokens: {total_tokens:,}",
            f"Last sync: {last_sync_str}",
            "",
            "**By project:**",
        ]
        for proj, count, tokens in project_rows:
            lines.append(f"  {proj}: {count} chunks ({tokens:,} tokens)")

        lines.append("")
        lines.append("**By source type:**")
        for stype, count, tokens in type_rows:
            half_life = cfg.get_half_life(stype)
            decay_info = f"half-life {half_life}d" if half_life else "evergreen"
            lines.append(f"  {stype}: {count} chunks ({tokens:,} tokens) [{decay_info}]")

        return "\n".join(lines)
    except Exception as e:
        return f"Error getting status: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
