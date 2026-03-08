#!/usr/bin/env python3
"""Sense auto-query hook — ambient knowledge surfacing for Claude Code.

Fires on UserPromptSubmit. Embeds the user's prompt, searches the Sense
index for relevant prior work, and injects results as context before
Claude processes the message.

Bypasses MCP transport — imports Sense search logic directly and opens
SQLite in read-only mode to coexist with the running MCP server.

Always exits 0. Never blocks user input.

Entry point for `sense-mcp-hook` console script (pyproject.toml).
"""

import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

from sense_mcp.session import SessionState

# ---------------------------------------------------------------------------
# Gate conditions
# ---------------------------------------------------------------------------

CONTINUATION_PATTERN = re.compile(
    r"^(yes|no|ok|go|sure|do it|fire it|continue|proceed|thanks|thank you|"
    r"looks good|lgtm|ship it|sounds good|perfect|great|nice|done|yep|yup|"
    r"nah|nope|got it|roger|ack|k|y|n)[\s.!?]*$",
    re.IGNORECASE,
)

MIN_WORDS = 8
COOLDOWN_SECONDS = 60


# ---------------------------------------------------------------------------
# Search parameters
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLD = 0.35
MAX_RESULTS = 3
PREVIEW_CHARS = 200
SURFACED_PENALTY = 0.5
MAX_EMBED_WORDS = 100  # truncate long prompts for embedding


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

def should_search(prompt_text: str, session_state: SessionState) -> bool:
    """Apply gate conditions. Returns True if search should proceed."""
    text = prompt_text.strip()

    # Slash command
    if text.startswith("/"):
        return False

    # Continuation signal
    if CONTINUATION_PATTERN.match(text):
        return False

    # Too short (whitespace split, not tiktoken — saves ~200ms cold start)
    if len(text.split()) < MIN_WORDS:
        return False

    # Cooldown
    if time.time() - session_state.last_query_time < COOLDOWN_SECONDS:
        return False

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Read hook input from stdin
    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return

    # Extract prompt text — handle both dict and string formats
    prompt = input_data.get("prompt", {})
    if isinstance(prompt, dict):
        prompt_text = prompt.get("content", "")
    elif isinstance(prompt, str):
        prompt_text = prompt
    else:
        prompt_text = str(prompt)

    if not prompt_text:
        return

    # Load shared session state and check gates
    state = SessionState.load()
    if not should_search(prompt_text, state):
        return

    # -----------------------------------------------------------------------
    # Past gates — import Sense server (deferred to avoid slow import on
    # short-circuit paths like "yes" or "/commit")
    # -----------------------------------------------------------------------

    from sense_mcp import server as sense_server

    # Pre-seed a read-only DB connection to avoid WAL conflicts with the
    # running MCP server. Must happen BEFORE any search call.
    db_path = str(sense_server.DB_PATH)
    if not os.path.exists(db_path):
        return

    ro_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    sense_server._db_conn = ro_conn

    try:
        # Check index has content
        total = ro_conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if total == 0:
            return

        # Embed the prompt (truncate long prompts to ~100 words)
        words = prompt_text.split()
        search_text = " ".join(words[:MAX_EMBED_WORDS])
        query_emb = sense_server.embed_query(search_text)

        # Search with mode-awareness (mode=None -> auto-detect from Vibe Harness)
        results, _meta = sense_server.search_chunks_contextual(
            query_emb,
            query_text=search_text,
            mode=None,
            limit=5,
        )

        if not results:
            state.last_query_time = time.time()
            state.save()
            return

        # Load relevance weights from feedback (read-only is fine)
        from sense_mcp.feedback import load_relevance_weights
        try:
            rel_weights = load_relevance_weights(
                ro_conn,
                boost_factor=cfg.feedback_boost_factor,
                prior=cfg.feedback_prior,
            )
        except Exception:
            rel_weights = {}

        # Filter and deduplicate
        seen_files = set()
        filtered = []

        for r in results:
            # Similarity threshold
            if r["similarity"] < SIMILARITY_THRESHOLD:
                continue

            # Deduplicate by file (keep highest-scoring chunk per file)
            fp = r["file_path"]
            if fp in seen_files:
                continue
            seen_files.add(fp)

            # Apply relevance feedback weight
            rel_w = rel_weights.get(fp, 1.0)
            if rel_w != 1.0:
                r["score"] *= rel_w

            # De-weight previously surfaced files using count-based penalty
            penalty = state.get_surfaced_penalty(fp, SURFACED_PENALTY)
            if penalty < 1.0:
                r["score"] *= penalty
                # Drop if penalty pushes below a usable score
                if r["score"] < SIMILARITY_THRESHOLD * 0.5:
                    continue

            filtered.append(r)
            if len(filtered) >= MAX_RESULTS:
                break

        # Update session state (even if no results — records cooldown)
        cfg = sense_server.cfg
        state.last_query_time = time.time()
        state.record_surfaced(filtered, surfaced_cap=cfg.surfaced_cap)
        state.record_query(
            search_text,
            [r["file_path"] for r in filtered],
            max_queries=cfg.max_queries,
        )
        state.save()

        if not filtered:
            return

        # Format output for injection
        eco_str = str(sense_server.ECOSYSTEM_ROOT)
        lines = ["<sense-context>", "Prior work that may be relevant:", ""]

        for i, r in enumerate(filtered, 1):
            rel_path = r["file_path"].replace(eco_str + "/", "")
            section = f" > {r['section']}" if r.get("section") else ""
            preview = r["content"][:PREVIEW_CHARS]
            if len(r["content"]) > PREVIEW_CHARS:
                preview += "..."

            lines.append(f"{i}. [{r['project']}] {rel_path}{section}")
            lines.append(
                f"   Score: {r['score']:.2f} | Type: {r['source_type']}"
                f" | Date: {r.get('date', 'unknown')}"
            )
            lines.append(f"   {preview}")
            lines.append("")

        lines.append("Use or ignore as appropriate.")
        lines.append("</sense-context>")

        print("\n".join(lines))

    finally:
        ro_conn.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass  # Never block user input
    sys.exit(0)
