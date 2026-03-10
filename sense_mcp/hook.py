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
MIN_CONTEXT_QUERIES = 2  # need at least this many prior queries to blend


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

def build_contextual_query(
    current_emb: "np.ndarray",
    session_state: SessionState,
    embed_fn,
    context_window: int = 5,
    decay_factor: float = 0.5,
    max_context_weight: float = 0.4,
    session_timeout: int = 7200,
    trajectory_signal: dict | None = None,
) -> tuple["np.ndarray", dict]:
    """Blend current message embedding with recent query context.

    Returns (embedding, metadata) where metadata describes the blend for
    dashboard visibility. The embedding is L2-normalised and preserves the
    current message as dominant (>= 1 - max_context_weight of the blend).

    Falls back to (current_emb, empty metadata) when insufficient context.
    """
    import numpy as np

    no_context = {"blended": False, "reason": "insufficient_context"}

    # Gate: need enough prior queries
    if len(session_state.queries) < MIN_CONTEXT_QUERIES:
        return current_emb, no_context

    # Filter to queries within session timeout
    now = time.time()
    recent_queries = [
        q for q in session_state.queries
        if now - q.get("ts", 0) < session_timeout
    ]

    if len(recent_queries) < MIN_CONTEXT_QUERIES:
        return current_emb, no_context

    # Take last N queries
    context_queries = recent_queries[-context_window:]

    # Compute raw weights (most recent = index N-1 gets highest weight)
    n = len(context_queries)
    raw_weights = [decay_factor ** (n - 1 - i) for i in range(n)]

    # Adjust the cap based on trajectory signal — this changes the TOTAL
    # amount of context influence, not just its distribution
    effective_cap = max_context_weight
    cap_reason = None
    if trajectory_signal:
        trend = trajectory_signal.get("trend")
        if trend == "converging":
            effective_cap *= 0.5  # less total context, let current message break out
            cap_reason = "converging: cap halved to let current message break out"
        elif trend == "diverging":
            effective_cap = min(effective_cap * 1.5, 0.6)  # more anchoring, hard ceiling at 0.6
            cap_reason = "diverging: cap increased for more anchoring"

    # Cap total context weight to enforce current message dominance
    context_sum = sum(raw_weights)
    if context_sum > effective_cap:
        scale = effective_cap / context_sum
        weights = [w * scale for w in raw_weights]
    else:
        weights = raw_weights

    # Re-embed context queries and blend
    composite = current_emb.copy().astype(np.float64)
    for weight, query_entry in zip(weights, context_queries):
        ctx_emb = embed_fn(query_entry["query"])
        composite += weight * ctx_emb

    # L2 normalise
    norm = np.linalg.norm(composite)
    if norm > 0:
        composite = composite / norm

    # Build metadata for dashboard
    context_meta = {
        "blended": True,
        "effective_cap": round(effective_cap, 3),
        "cap_reason": cap_reason,
        "context_queries": [
            {
                "query": cq["query"][:120],
                "weight": round(w, 4),
                "ts": cq.get("ts", 0),
            }
            for w, cq in zip(weights, context_queries)
        ],
    }

    return composite, context_meta


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
    from sense_mcp.feedback import record_feedback
    from sense_mcp.trajectory import TrajectoryComputer

    # Open a read-write connection for feedback auto-labelling (SPEC-003 REQ-006).
    # SQLite WAL mode handles concurrent writers with the MCP server.
    db_path = str(sense_server.DB_PATH)
    if not os.path.exists(db_path):
        return

    rw_conn = sqlite3.connect(db_path)
    rw_conn.execute("PRAGMA journal_mode=WAL")
    sense_server._db_conn = rw_conn

    try:
        # Check index has content
        total = rw_conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if total == 0:
            return

        # Embed the prompt (truncate long prompts to ~100 words)
        words = prompt_text.split()
        search_text = " ".join(words[:MAX_EMBED_WORDS])
        query_emb = sense_server.embed_query(search_text)

        # Update trajectory (adds embedding, computes signal)
        traj = TrajectoryComputer.load()
        traj.add_embedding(query_emb)
        traj_signal = traj.compute_signal()
        traj.save()

        # Build contextual query by blending with recent conversation
        cfg = sense_server.cfg
        search_emb, context_meta = build_contextual_query(
            current_emb=query_emb,
            session_state=state,
            embed_fn=sense_server.embed_query,
            context_window=cfg.hook_context_window,
            decay_factor=cfg.hook_context_decay,
            max_context_weight=cfg.hook_max_context_weight,
            session_timeout=cfg.hook_context_session_timeout,
            trajectory_signal=traj_signal,
        )

        # Search with mode-awareness (mode=None -> auto-detect from Vibe Harness)
        results, _meta = sense_server.search_chunks_contextual(
            search_emb,
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
                rw_conn,
                boost_factor=cfg.feedback_boost_factor,
                prior=cfg.feedback_prior,
            )
        except Exception:
            rel_weights = {}

        # Anti-entrainment: when trajectory is converging, boost cross-project
        # and divergent results to surface productive dissonance.
        is_converging = traj_signal.get("trend") == "converging"

        # Filter and deduplicate
        seen_files = set()
        filtered = []
        noise_candidates = []  # candidates above floor but filtered out
        noise_floor = SIMILARITY_THRESHOLD * 0.8  # REQ-006: below this = don't label

        for r in results:
            # Similarity threshold
            if r["similarity"] < SIMILARITY_THRESHOLD:
                # Track as noise candidate if above the labelling floor
                if r["similarity"] >= noise_floor:
                    noise_candidates.append(r)
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
                    noise_candidates.append(r)
                    continue

            # Anti-entrainment boost: when converging, prefer cross-project
            # and divergent source types to break the convergence trap
            if is_converging and r.get("cross_project"):
                r["score"] *= 1.3

            filtered.append(r)
            if len(filtered) >= MAX_RESULTS:
                break

        # Update session state (even if no results — records cooldown)
        state.last_query_time = time.time()
        state.trajectory_signal = traj_signal
        state.record_surfaced(filtered, surfaced_cap=cfg.surfaced_cap)
        state.record_query(
            search_text,
            [r["file_path"] for r in filtered],
            max_queries=cfg.max_queries,
            surfaced_results=[
                {
                    "file_path": r["file_path"],
                    "section": r.get("section", ""),
                    "snippet": r.get("content", "")[:PREVIEW_CHARS],
                    "source_type": r.get("source_type", ""),
                    "score": round(r.get("score", 0.0), 2),
                    "project": r.get("project", ""),
                }
                for r in filtered
            ],
            context_meta=context_meta,
            trajectory=traj_signal,
        )
        state.save()

        # --- Auto-labelling (SPEC-003 REQ-006) ---
        # Results that passed all gates -> useful
        # Results filtered out above noise floor -> noise
        mode_str = sense_server.detect_current_mode()
        try:
            for r in filtered:
                record_feedback(
                    rw_conn,
                    query_text=search_text,
                    file_path=r["file_path"],
                    label="useful",
                    similarity=r.get("similarity"),
                    mode=mode_str,
                    source="auto:hook",
                )
            for r in noise_candidates:
                record_feedback(
                    rw_conn,
                    query_text=search_text,
                    file_path=r["file_path"],
                    label="noise",
                    similarity=r.get("similarity"),
                    mode=mode_str,
                    source="auto:hook",
                )
        except Exception:
            pass  # auto-labelling is non-critical — never block

        if not filtered:
            return

        # Format output for injection
        eco_str = str(sense_server.ECOSYSTEM_ROOT)
        lines = ["<sense-context>", "Prior work that may be relevant:", ""]

        # Add trajectory annotation when signal is meaningful
        trend = traj_signal.get("trend", "insufficient_data")
        if trend != "insufficient_data":
            dk = traj_signal.get("delta_kappa")
            dk_str = f" (dk={dk:.3f})" if dk is not None else ""
            if trend == "converging":
                lines.append(f"[Trajectory: {trend}{dk_str} -- surfacing for productive dissonance]")
            else:
                lines.append(f"[Trajectory: {trend}{dk_str}]")
            lines.append("")

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
        rw_conn.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass  # Never block user input
    sys.exit(0)
