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
            }
            for r in results[:cap]
        ]

    def record_query(
        self,
        query_text: str,
        surfaced_files: list[str],
        max_queries: int,
    ) -> None:
        """Append query to history with rolling-window eviction."""
        self.queries.append(
            {
                "query": query_text,
                "ts": time.time(),
                "surfaced_files": surfaced_files,
            }
        )
        if len(self.queries) > max_queries:
            self.queries = self.queries[-max_queries:]

    # -----------------------------------------------------------------------
    # Scoring
    # -----------------------------------------------------------------------

    def get_surfaced_penalty(self, file_path: str, base_penalty: float) -> float:
        """Return base_penalty^count, floored at 0.05. Returns 1.0 if unseen."""
        entry = self.surfaced.get(file_path)
        if not entry:
            return 1.0
        return max(base_penalty ** entry["count"], 0.05)

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
