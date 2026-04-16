"""Sense session state — persisted session tracking for ambient relevance.

Holds surfaced file history and query history across the MCP server process
and hook invocations. State is serialised to a JSON file and protected with
fcntl.flock for safe concurrent access between hook and MCP server.

Design constraints:
  - No imports from server.py or config.py at module level (avoids circular
    imports — this module is shared by both callers).
  - Config values (surfaced_cap, max_queries) are passed as parameters.
  - Embeddings are NOT stored (not JSON-serialisable). get_circling_files()
    re-embeds past query texts on demand via a caller-supplied embed_fn.
  - numpy is imported lazily (inside methods that need it) so that importing
    this module for the hook cooldown gate does not incur numpy's import cost.
"""

import fcntl
import json
import time
from dataclasses import dataclass, field
from typing import Callable

STATE_PATH = "/tmp/sense-session-state.json"


def _extract_bias_fields(r: dict) -> dict:
    """Extract the bias breakdown fields from a result dict."""
    return {
        "score": round(r.get("score", 0.0), 4),
        "bias_sum": r.get("bias_sum", 0.0),
        "bias_contribution": r.get("bias_contribution", 0.0),
        "mode_multiplier": r.get("mode_multiplier", 1.0),
        "resurfaced": r.get("resurfaced", False),
        "resurface_penalty": r.get("resurface_penalty"),
        "circling": r.get("circling", False),
        "cross_project": r.get("cross_project", False),
    }


def format_surfaced_result(r: dict, snippet_len: int = 200) -> dict:
    """Build a serialisable result dict for session state and dashboard.

    Shared by server.py and hook.py to keep the surfaced_results schema
    consistent across both search paths.
    """
    return {
        "file_path": r["file_path"],
        "section": r.get("section", ""),
        "snippet": r.get("content", "")[:snippet_len],
        "source_type": r.get("source_type", ""),
        "project": r.get("project", ""),
        **_extract_bias_fields(r),
    }


@dataclass
class SessionState:
    """Serialisable session state for ambient relevance tracking.

    surfaced: {file_path: {"count": int, "last_ts": float}}
    queries:  [{"query": str, "ts": float, "surfaced_files": [str]}]
    last_query_time: float — used by hook cooldown gate
    """

    surfaced: dict = field(default_factory=dict)
    queries: list = field(default_factory=list)
    last_query_time: float = 0.0
    last_results: list = field(default_factory=list)  # [{file_path, query, similarity}]
    trajectory_signal: dict = field(default_factory=dict)  # {delta_kappa, trend, turn_count, curvature_std}

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | None = None) -> "SessionState":
        """Load state from JSON file with shared read lock.

        Returns an empty SessionState on any read/parse error.
        """
        if path is None:
            path = STATE_PATH
        try:
            with open(path, "r") as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            state = cls()
            state.surfaced = data.get("surfaced", {})
            state.queries = data.get("queries", [])
            state.last_query_time = float(data.get("last_query_time", 0.0))
            state.last_results = data.get("last_results", [])
            state.trajectory_signal = data.get("trajectory_signal", {})
            return state
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return cls()

    def save(self, path: str | None = None) -> None:
        """Persist state to JSON file with exclusive write lock. Silent on error."""
        if path is None:
            path = STATE_PATH
        try:
            with open(path, "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    json.dump(
                        {
                            "surfaced": self.surfaced,
                            "queries": self.queries,
                            "last_query_time": self.last_query_time,
                            "last_results": self.last_results,
                            "trajectory_signal": self.trajectory_signal,
                        },
                        f,
                    )
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        except OSError:
            pass  # non-critical

    # -----------------------------------------------------------------------
    # Mutation
    # -----------------------------------------------------------------------

    def record_surfaced(self, results: list[dict], surfaced_cap: int) -> None:
        """Update surfaced counts and evict LRU entries beyond cap."""
        now = time.time()
        for r in results:
            fp = r["file_path"]
            if fp in self.surfaced:
                self.surfaced[fp]["count"] += 1
                self.surfaced[fp]["last_ts"] = now
            else:
                self.surfaced[fp] = {"count": 1, "last_ts": now}

        if len(self.surfaced) > surfaced_cap:
            by_ts = sorted(self.surfaced.items(), key=lambda kv: kv[1]["last_ts"])
            to_remove = len(self.surfaced) - surfaced_cap
            for key, _ in by_ts[:to_remove]:
                del self.surfaced[key]

    def record_last_results(
        self, results: list[dict], query_text: str, cap: int = 20
    ) -> None:
        """Store the most recent search results for index-based feedback."""
        self.last_results = [
            {
                "file_path": r["file_path"],
                "query": query_text,
                "similarity": r.get("similarity", 0.0),
                **_extract_bias_fields(r),
            }
            for r in results[:cap]
        ]

    def record_query(
        self,
        query_text: str,
        surfaced_files: list[str],
        max_queries: int,
        surfaced_results: list[dict] | None = None,
        context_meta: dict | None = None,
        trajectory: dict | None = None,
    ) -> None:
        """Append query to history with rolling-window eviction.

        surfaced_results: rich result data for dashboard (file_path, section,
            snippet, source_type, score, project).
        context_meta: contextual query construction metadata (blended, weights,
            cap, prior queries used).
        trajectory: trajectory signal at query time (trend, delta_kappa, turn_count).
        Kept alongside surfaced_files for backward compat with get_circling_files().
        """
        entry: dict = {
            "query": query_text,
            "ts": time.time(),
            "surfaced_files": surfaced_files,
        }
        if surfaced_results:
            entry["surfaced_results"] = surfaced_results
        if context_meta:
            entry["context_meta"] = context_meta
        if trajectory:
            entry["trajectory"] = trajectory
        self.queries.append(entry)
        if len(self.queries) > max_queries:
            self.queries = self.queries[-max_queries:]

    # -----------------------------------------------------------------------
    # Scoring
    # -----------------------------------------------------------------------

    def get_surfaced_penalty(self, file_path: str, base_penalty: float) -> float:
        """Return a decaying penalty based on surfacing count.

        Uses a linear penalty capped at a floor, replacing the old exponential
        model (base_penalty^count) which compounded destructively over long
        sessions. See SPEC-004 for the diagnostic.

        Penalty = max(base_penalty - 0.05 * (count - 1), 0.5)

        Examples with base_penalty=0.75:
          1× surfaced: 0.75 (25% reduction)
          2× surfaced: 0.70 (30%)
          3× surfaced: 0.65 (35%)
          6× surfaced: 0.50 (50% floor — never worse)

        Returns 1.0 if unseen.
        """
        entry = self.surfaced.get(file_path)
        if not entry:
            return 1.0
        count = entry["count"]
        # Linear decay from base_penalty, floored at 0.5
        penalty = base_penalty - 0.05 * (count - 1)
        return max(penalty, 0.5)

    def get_circling_files(
        self,
        query_embedding: "np.ndarray",
        embed_fn: Callable[[str], "np.ndarray"],
        threshold: float = 0.75,
    ) -> set[str]:
        """Return file paths from prior queries semantically similar to query_embedding.

        Re-embeds past query texts on demand (embeddings are not stored in state).
        Similarity is cosine; threshold is inclusive.
        """
        import numpy as np  # lazy import — session.py has no top-level numpy dep

        if not self.queries:
            return set()

        circling: set[str] = set()
        for past in self.queries:
            past_emb = embed_fn(past["query"])
            sim = _cosine_similarity(query_embedding, past_emb)
            if sim >= threshold:
                circling.update(past["surfaced_files"])
        return circling


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: "np.ndarray", b: "np.ndarray") -> float:
    import numpy as np  # lazy import

    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0:
        return 0.0
    return dot / norm
