"""Semantic climate trajectory computation for Sense.

Tracks the semantic trajectory of a conversation by accumulating query
embeddings and computing local curvature (delta-kappa) using discrete
Frenet-Serret geometry. The curvature signal indicates whether the
conversation is converging (straightening — potential entrainment trap)
or diverging (bending — exploring new territory).

The curvature computation is inlined from semantic-climate-phase-space
core_metrics to avoid a runtime dependency on that package. The algorithm
is: kappa(t) = ||a_perp|| / ||v||^2, where v and a are velocity and
acceleration in embedding space.

Storage: embeddings are persisted as a numpy binary file alongside the
session state. File-locked for safe concurrent access between hook and
MCP server (same pattern as session.py).

Design constraints:
  - No imports from server.py or config.py at module level
  - numpy is the only dependency (already required by Sense)
  - Embedding buffer is size-capped (default 50 turns)
  - Graceful on all errors — never blocks user input
"""

import fcntl
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

TRAJECTORY_PATH = Path("/tmp/sense-trajectory-embeddings.npy")
TRAJECTORY_HISTORY_PATH = Path("/tmp/sense-trajectory-history.jsonl")

# Minimum embeddings needed for curvature computation
# (4 points -> 3 velocities -> 2 accelerations -> 2 curvature values)
MIN_EMBEDDINGS_FOR_CURVATURE = 4

# Rolling window for trend detection (compare recent vs older curvatures)
TREND_WINDOW = 5


def _compute_local_curvatures(embeddings: np.ndarray) -> np.ndarray:
    """Compute local curvature at each interior point using Frenet-Serret.

    For a trajectory e(t), curvature at point t is:
        kappa(t) = ||a_perp|| / ||v||^2

    where v = e(t+1) - e(t) is velocity, a = v(t+1) - v(t) is acceleration,
    and a_perp is the component of acceleration perpendicular to velocity.

    This is the same algorithm as SemanticComplexityAnalyzer._compute_local_curvatures
    in semantic-climate-phase-space/src/core_metrics.py, inlined here to avoid
    a package dependency.

    Args:
        embeddings: (n, d) array of embedding vectors

    Returns:
        Array of local curvatures at points 0 to n-3
    """
    n = len(embeddings)
    if n < MIN_EMBEDDINGS_FOR_CURVATURE:
        return np.array([])

    velocities = np.diff(embeddings, axis=0)  # (n-1, d)
    accelerations = np.diff(velocities, axis=0)  # (n-2, d)

    curvatures = []
    for i in range(len(accelerations)):
        v = velocities[i]
        a = accelerations[i]

        v_norm = np.linalg.norm(v)
        if v_norm < 1e-10:
            curvatures.append(0.0)
            continue

        v_hat = v / v_norm
        a_parallel = np.dot(a, v_hat) * v_hat
        a_perp = a - a_parallel
        kappa = np.linalg.norm(a_perp) / (v_norm ** 2)
        curvatures.append(kappa)

    return np.array(curvatures)


def _detect_trend(curvatures: np.ndarray, window: int = TREND_WINDOW) -> str:
    """Detect curvature trend from recent history.

    Compares mean curvature in the most recent `window` values against
    the preceding `window` values. A significant decrease indicates
    convergence (trajectory straightening); an increase indicates
    divergence (trajectory bending into new territory).

    Args:
        curvatures: Array of local curvature values
        window: Number of recent values to compare

    Returns:
        'converging', 'diverging', or 'stable'
    """
    if len(curvatures) < window * 2:
        # Not enough history for trend — use absolute level
        mean_k = float(np.mean(curvatures))
        if mean_k < 0.05:
            return "converging"
        elif mean_k > 0.15:
            return "diverging"
        return "stable"

    recent = curvatures[-window:]
    older = curvatures[-window * 2:-window]

    recent_mean = float(np.mean(recent))
    older_mean = float(np.mean(older))

    # Relative change threshold
    if older_mean > 1e-10:
        change = (recent_mean - older_mean) / older_mean
    else:
        change = recent_mean  # older was ~zero, any increase is divergence

    if change < -0.2:
        return "converging"
    elif change > 0.2:
        return "diverging"
    return "stable"


class TrajectoryComputer:
    """Accumulates query embeddings and computes trajectory signal.

    Usage:
        tc = TrajectoryComputer.load()
        tc.add_embedding(query_emb)
        signal = tc.compute_signal()
        tc.save()

    The signal dict is written to SessionState.trajectory_signal by the
    caller (server.py or hook.py).
    """

    def __init__(self, max_embeddings: int = 50):
        self.max_embeddings = max_embeddings
        self.embeddings: list[np.ndarray] = []

    @classmethod
    def load(cls, path: Path | None = None, max_embeddings: int = 50) -> "TrajectoryComputer":
        """Load embedding history from disk. Returns empty on any error."""
        if path is None:
            path = TRAJECTORY_PATH
        tc = cls(max_embeddings=max_embeddings)
        try:
            with open(path, "rb") as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    data = np.load(f, allow_pickle=False)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
            # data is (n, d) array
            if data.ndim == 2 and len(data) > 0:
                tc.embeddings = list(data[-max_embeddings:])
        except (FileNotFoundError, OSError, ValueError):
            pass
        return tc

    def save(self, path: Path | None = None) -> None:
        """Persist embedding history to disk. Silent on error."""
        if path is None:
            path = TRAJECTORY_PATH
        try:
            if not self.embeddings:
                return
            arr = np.stack(self.embeddings[-self.max_embeddings:])
            with open(path, "wb") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    np.save(f, arr)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        except OSError:
            pass

    def add_embedding(self, embedding: np.ndarray) -> None:
        """Append a query embedding to the trajectory buffer.

        If the new embedding has a different dimensionality than existing
        embeddings, the buffer is reset. This handles model changes or
        mixed-source embeddings gracefully.
        """
        if self.embeddings and self.embeddings[0].shape != embedding.shape:
            self.embeddings = []  # Dimension mismatch — reset
        self.embeddings.append(embedding)
        if len(self.embeddings) > self.max_embeddings:
            self.embeddings = self.embeddings[-self.max_embeddings:]

    def compute_signal(self) -> dict:
        """Compute trajectory signal from accumulated embeddings.

        Returns:
            dict: {
                'delta_kappa': float or None,   # mean local curvature
                'trend': str,                   # 'converging'|'diverging'|'stable'|'insufficient_data'
                'turn_count': int,              # number of embeddings in buffer
                'curvature_std': float or None, # std of local curvatures
            }
        """
        n = len(self.embeddings)
        insufficient = {
            "delta_kappa": None,
            "trend": "insufficient_data",
            "turn_count": n,
            "curvature_std": None,
        }

        if n < MIN_EMBEDDINGS_FOR_CURVATURE:
            return insufficient

        try:
            embeddings_arr = np.stack(self.embeddings)
        except ValueError:
            # Mixed shapes — shouldn't happen after add_embedding guard, but be safe
            return insufficient

        curvatures = _compute_local_curvatures(embeddings_arr)

        if len(curvatures) == 0:
            return insufficient

        mean_kappa = float(np.mean(curvatures))
        std_kappa = float(np.std(curvatures))
        trend = _detect_trend(curvatures)

        signal = {
            "delta_kappa": mean_kappa,
            "trend": trend,
            "turn_count": n,
            "curvature_std": std_kappa,
        }

        # Append to trajectory history JSONL for dashboard consumption
        _append_trajectory_history(signal)

        return signal

    def clear(self) -> None:
        """Reset trajectory buffer."""
        self.embeddings = []


def _append_trajectory_history(
    signal: dict, path: Path | None = None
) -> None:
    """Append a trajectory signal entry to the JSONL history file.

    One line per hook invocation (~50 bytes). Dashboard reads this directly.
    Silent on all errors — never blocks user input.
    """
    if path is None:
        path = TRAJECTORY_HISTORY_PATH
    try:
        entry = {
            "ts": time.time(),
            "trend": signal.get("trend", "insufficient_data"),
            "delta_kappa": signal.get("delta_kappa"),
            "turn_count": signal.get("turn_count", 0),
        }
        with open(path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(json.dumps(entry) + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except OSError:
        pass
