"""Sense — Ambient Relevance for AI Conversations.

Indexes a project ecosystem and injects relevant context into every
conversation automatically. Weights by recency (temporal decay), adapts
to working mode (Vibe Harness integration), and structures results for
productive connections, not just nearest-match retrieval.

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

from .config import get_config
from .feedback import init_feedback_table, load_relevance_weights, record_feedback, get_feedback_stats
from .session import SessionState
from .trajectory import TrajectoryComputer

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
    init_feedback_table(conn)


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

# ---------------------------------------------------------------------------
# Relevance weight cache (feedback-driven)
# ---------------------------------------------------------------------------

_relevance_weights: dict[str, float] = {}
_weights_loaded_at: float = 0.0


def _get_relevance_weights() -> dict[str, float]:
    """Return cached per-file relevance weights, refreshing if stale."""
    global _relevance_weights, _weights_loaded_at
    ttl = cfg.feedback_weight_cache_ttl
    now = time.time()
    if now - _weights_loaded_at < ttl:
        return _relevance_weights
    try:
        _relevance_weights = load_relevance_weights(
            get_db(),
            boost_factor=cfg.feedback_boost_factor,
            prior=cfg.feedback_prior,
        )
        _weights_loaded_at = now
    except Exception:
        pass  # graceful — use stale cache or empty dict
    return _relevance_weights


def _invalidate_weight_cache() -> None:
    """Force next search to reload weights (call after recording feedback)."""
    global _weights_loaded_at
    _weights_loaded_at = 0.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


# ---------------------------------------------------------------------------
# Session trajectory tracking (shared state)
#
# search_chunks_contextual loads SessionState from /tmp/sense-session-state.json
# at entry and saves it before return. Both the MCP server and the hook
# (hook.py) share state through this file, protected by fcntl.flock.
# ---------------------------------------------------------------------------


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

    rel_weights = _get_relevance_weights()

    results = []
    for row in rows:
        (chunk_id, file_path, proj, stype, section, date, evergreen, content, token_count, emb_blob) = row
        stored_emb = np.frombuffer(emb_blob, dtype=np.float32)
        sim = cosine_similarity(query_embedding, stored_emb)
        decay = compute_decay(stype, date)
        rel_w = rel_weights.get(file_path, 1.0)
        ep_w = cfg.get_epistemic_weight(file_path)
        score = sim * decay * rel_w * ep_w

        results.append({
            "score": score,
            "similarity": sim,
            "decay": decay,
            "relevance_weight": rel_w,
            "file_path": file_path,
            "project": proj,
            "source_type": stype,
            "section": section,
            "date": date,
            "content": content,
            "token_count": token_count,
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate by file — keep highest-scoring chunk per file
    seen_files: set[str] = set()
    deduped: list[dict] = []
    for r in results:
        if r["file_path"] not in seen_files:
            seen_files.add(r["file_path"])
            deduped.append(r)
            if len(deduped) >= limit:
                break

    return deduped


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

    Session state is loaded from shared storage at entry and saved before return,
    so both the MCP server and hook invocations accumulate a shared trajectory.
    """
    session = SessionState.load()

    # --- Trajectory computation (semantic climate) ---
    traj = TrajectoryComputer.load()
    traj.add_embedding(query_embedding)
    traj_signal = traj.compute_signal()
    traj.save()
    session.trajectory_signal = traj_signal

    mode_profiles = cfg.mode_profiles
    diversity_slots = cfg.diversity_slots

    # Explicit "none" bypasses auto-detection → flat search
    if mode and mode.lower() in ("none", "flat"):
        results = search_chunks(query_embedding, project, source_type, limit)
        return results, {"mode": None, "trajectory": traj_signal}

    if not mode:
        mode = detect_current_mode()

    if not mode or mode not in mode_profiles:
        results = search_chunks(query_embedding, project, source_type, limit)
        return results, {"mode": None, "trajectory": traj_signal}

    profile = mode_profiles[mode]
    diversity_profile_name = profile["diversity_profile"]

    # Anti-entrainment: if trajectory is converging, widen diversity
    # regardless of mode profile. This is the core relevance realisation
    # mechanism — when the conversation narrows, surface productive dissonance.
    if traj_signal.get("trend") == "converging" and diversity_profile_name == "narrow":
        diversity_profile_name = "wide"

    slots = diversity_slots[diversity_profile_name]

    pool_size = limit * 8
    raw_candidates = search_chunks(query_embedding, project, None, pool_size)

    # --- Stratified pool assembly ---
    # Group by source_type (each list is already score-sorted from search_chunks)
    by_type: dict[str, list[dict]] = {}
    for r in raw_candidates:
        stype = r["source_type"]
        if stype not in by_type:
            by_type[stype] = []
        by_type[stype].append(r)

    num_active_types = len(by_type)
    if num_active_types == 0:
        candidates = []
    else:
        per_type_quota = max(pool_size // num_active_types, cfg.min_type_slots)
        stratified: list[dict] = []
        used_ids: set[int] = set()
        for type_candidates in by_type.values():
            take = min(len(type_candidates), per_type_quota)
            for r in type_candidates[:take]:
                stratified.append(r)
                used_ids.add(id(r))
        # Fill remaining slots from leftover candidates (raw_candidates is
        # already sorted by score descending, so order is preserved)
        remaining = [r for r in raw_candidates if id(r) not in used_ids]
        fill_slots = pool_size - len(stratified)
        candidates = stratified + remaining[:max(0, fill_slots)]

    # --- Additive bias model (SPEC-004 Phase 1) ---
    # Contextual signals are summed and added with a small weight (α),
    # bounding how much mode/session state can shift ranking relative to
    # the base relevance score (cosine × decay × feedback weight).
    #
    # Signal design: each signal is normalised to roughly [-1, +1] range.
    # α controls total contextual influence (default 0.05 = max ±5% shift
    # on a score of 1.0, more on lower scores but still bounded).

    ALPHA = 0.05  # contextual influence weight

    multipliers = profile["source_type_multipliers"]
    cross_boost = profile["cross_project_boost"]
    base_penalty = profile["already_surfaced_penalty"]
    circling_files = session.get_circling_files(query_embedding, embed_fn=embed_query)

    for r in candidates:
        bias_sum = 0.0

        # Source type preference: multiplier > 1.0 = positive, < 1.0 = negative
        mult = multipliers.get(r["source_type"], 1.0)
        type_signal = mult - 1.0  # e.g. 1.5 → +0.5, 0.6 → -0.4
        bias_sum += type_signal
        r["mode_multiplier"] = mult

        # Explicit source_type filter boost (small)
        if source_type and r["source_type"] == source_type:
            bias_sum += 0.2

        # Cross-project signal
        if project and r["project"] != project:
            cross_signal = cross_boost - 1.0  # e.g. 1.3 → +0.3
            bias_sum += cross_signal
            r["cross_project"] = True
        else:
            r["cross_project"] = False

        # Resurfacing signal (now uses decaying penalty, not exponential)
        penalty = session.get_surfaced_penalty(r["file_path"], base_penalty)
        if penalty < 1.0:
            resurface_signal = penalty - 1.0  # e.g. 0.75 → -0.25
            bias_sum += resurface_signal
            r["resurfaced"] = True
            r["resurface_penalty"] = penalty
        else:
            r["resurfaced"] = False

        # Circling signal (topic recurrence)
        if r["file_path"] in circling_files:
            bias_sum += 0.3
            r["circling"] = True
        else:
            r["circling"] = False

        # Apply additive bias: score = base_score + α × Σ(signals)
        r["bias_sum"] = round(bias_sum, 4)
        r["bias_contribution"] = round(ALPHA * bias_sum, 4)
        r["score"] = r["score"] + ALPHA * bias_sum

    candidates.sort(key=lambda x: x["score"], reverse=True)

    results = assemble_diverse_results(candidates, slots, project)
    results = results[:limit]

    surfaced_files = [r["file_path"] for r in results]
    session.record_surfaced(results, cfg.surfaced_cap)
    session.record_query(query_text, surfaced_files, cfg.max_queries)
    session.record_last_results(results, query_text)
    session.save()

    metadata = {
        "mode": mode,
        "diversity_profile": diversity_profile_name,
        "slots": slots,
        "circling_count": sum(1 for r in results if r.get("circling")),
        "resurfaced_count": sum(1 for r in results if r.get("resurfaced")),
        "session_queries": len(session.queries),
        "trajectory": traj_signal,
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
        source_type: Optional type filter: trace, mistake, documentation, project_claude, reference, research, teaching, code. In flat mode (no Vibe Harness), this is a hard SQL filter — only matching types returned. In mode-aware mode, this becomes a soft 1.2x score boost — matching types are preferred but other types can still appear in divergence/serendipity slots.
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
        traj_info = meta.get("trajectory", {})
        traj_label = ""
        if traj_info.get("trend") and traj_info["trend"] != "insufficient_data":
            dk = traj_info.get("delta_kappa")
            dk_str = f" dk={dk:.3f}" if dk is not None else ""
            traj_label = f" | Trajectory: {traj_info['trend']}{dk_str} ({traj_info.get('turn_count', 0)} turns)"

        if active_mode:
            lines = [
                f"Found {len(results)} result(s) for: \"{query}\"",
                f"Mode: {active_mode} | Diversity: {meta['diversity_profile']} | Session queries: {meta['session_queries']}{traj_label}",
                "",
            ]
        else:
            header = f"Found {len(results)} result(s) for: \"{query}\""
            if traj_label:
                header += traj_label
            lines = [header, ""]

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


@mcp.tool()
def sense_feedback(
    label: str,
    result_index: int = 0,
    file_path: str = "",
    note: str = "",
) -> str:
    """Record relevance feedback for a surfaced result.

    Labels a previously surfaced file as 'useful' or 'noise' so Sense
    can learn to distinguish signal from noise over time. Feedback adjusts
    per-file relevance weights applied during search scoring.

    Args:
        label: 'useful' or 'noise'.
        result_index: 1-based index from the most recent sense_search results.
            Use this OR file_path, not both.
        file_path: Full file path of the result. Use this when result_index
            is not available.
        note: Optional note explaining why (helps pattern analysis later).
    """
    if label not in ("useful", "noise"):
        return f"Invalid label: {label!r}. Must be 'useful' or 'noise'."

    session = SessionState.load()
    query_text = ""
    similarity = None

    if result_index > 0:
        if not session.last_results:
            return "No recent search results in session state. Use file_path instead."
        if result_index > len(session.last_results):
            return f"Index {result_index} out of range (last search had {len(session.last_results)} results)."
        entry = session.last_results[result_index - 1]
        file_path = entry["file_path"]
        query_text = entry.get("query", "")
        similarity = entry.get("similarity")
    elif not file_path:
        return "Provide either result_index or file_path."

    mode = detect_current_mode()

    try:
        record_feedback(
            get_db(),
            query_text=query_text,
            file_path=file_path,
            label=label,
            similarity=similarity,
            mode=mode,
            note=note or None,
        )
        _invalidate_weight_cache()

        root_str = str(cfg.root)
        rel_path = file_path.replace(root_str + "/", "")
        return f"Recorded: {rel_path} = {label}" + (f" ({note})" if note else "")
    except Exception as e:
        return f"Error recording feedback: {e}"


@mcp.tool()
def sense_feedback_stats() -> str:
    """Show what Sense has learned from relevance feedback.

    Summarises accumulated feedback: counts by label, most useful and
    noisiest files, current relevance weights, and feedback by mode.
    """
    try:
        stats = get_feedback_stats(get_db())

        if stats["total"] == 0:
            return "No feedback recorded yet. Use sense_feedback to label results."

        root_str = str(cfg.root)

        def rel(p):
            return p.replace(root_str + "/", "")

        lines = [
            f"Total feedback: {stats['total']}",
            f"  useful: {stats['by_label'].get('useful', 0)}",
            f"  noise: {stats['by_label'].get('noise', 0)}",
            "",
        ]

        if stats["top_useful"]:
            lines.append("**Most useful files:**")
            for fp, cnt in stats["top_useful"]:
                lines.append(f"  {rel(fp)}: {cnt}x useful")
            lines.append("")

        if stats["top_noisy"]:
            lines.append("**Noisiest files:**")
            for fp, cnt in stats["top_noisy"]:
                lines.append(f"  {rel(fp)}: {cnt}x noise")
            lines.append("")

        extremes = stats.get("weight_extremes", {})
        if extremes.get("most_penalised"):
            lines.append("**Current weight adjustments:**")
            for fp, w in extremes["most_penalised"]:
                if w < 1.0:
                    lines.append(f"  {rel(fp)}: {w:.3f} (penalised)")
            for fp, w in extremes.get("most_boosted", []):
                if w > 1.0:
                    lines.append(f"  {rel(fp)}: {w:.3f} (boosted)")
            lines.append("")

        if stats["by_mode"]:
            lines.append("**By mode:**")
            for mode_name, labels in stats["by_mode"].items():
                parts = [f"{l}: {c}" for l, c in labels.items()]
                lines.append(f"  {mode_name}: {', '.join(parts)}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error getting feedback stats: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
