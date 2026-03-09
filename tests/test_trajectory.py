"""Tests for sense_mcp.trajectory — semantic climate trajectory computation."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sense_mcp.trajectory import (
    MIN_EMBEDDINGS_FOR_CURVATURE,
    TrajectoryComputer,
    _compute_local_curvatures,
    _detect_trend,
)


# ---------------------------------------------------------------------------
# _compute_local_curvatures
# ---------------------------------------------------------------------------


class TestComputeLocalCurvatures:
    def test_too_few_points(self):
        """Need at least 4 points for curvature."""
        emb = np.random.randn(3, 10)
        result = _compute_local_curvatures(emb)
        assert len(result) == 0

    def test_exactly_four_points(self):
        """4 points -> 3 velocities -> 2 accelerations -> 2 curvatures."""
        emb = np.random.randn(4, 10)
        result = _compute_local_curvatures(emb)
        assert len(result) == 2

    def test_linear_trajectory_zero_curvature(self):
        """A straight line has zero curvature."""
        # Linear trajectory in 10D
        t = np.linspace(0, 1, 10)
        direction = np.random.randn(10)
        direction /= np.linalg.norm(direction)
        emb = np.outer(t, direction)
        result = _compute_local_curvatures(emb)
        assert len(result) > 0
        # All curvatures should be ~0
        assert np.allclose(result, 0.0, atol=1e-8)

    def test_curved_trajectory_nonzero_curvature(self):
        """A curved trajectory has nonzero curvature."""
        # Circular arc in 2D embedded in 10D
        t = np.linspace(0, np.pi, 20)
        emb = np.zeros((20, 10))
        emb[:, 0] = np.cos(t)
        emb[:, 1] = np.sin(t)
        result = _compute_local_curvatures(emb)
        assert len(result) > 0
        assert np.mean(result) > 0.1  # Definitely curved

    def test_stationary_points_handled(self):
        """Stationary points (zero velocity) get curvature 0."""
        emb = np.zeros((6, 10))  # All identical
        result = _compute_local_curvatures(emb)
        assert len(result) > 0
        assert np.allclose(result, 0.0)

    def test_output_length(self):
        """n points -> n-2 curvature values (from n-1 velocities, n-2 accelerations)."""
        for n in [5, 10, 20]:
            emb = np.random.randn(n, 8)
            result = _compute_local_curvatures(emb)
            assert len(result) == n - 2


# ---------------------------------------------------------------------------
# _detect_trend
# ---------------------------------------------------------------------------


class TestDetectTrend:
    def test_insufficient_data_low_curvature(self):
        """Short history with low curvature = converging."""
        curvatures = np.array([0.01, 0.02, 0.01])
        assert _detect_trend(curvatures) == "converging"

    def test_insufficient_data_high_curvature(self):
        """Short history with high curvature = diverging."""
        curvatures = np.array([0.3, 0.4, 0.35])
        assert _detect_trend(curvatures) == "diverging"

    def test_insufficient_data_medium_curvature(self):
        """Short history with medium curvature = stable."""
        curvatures = np.array([0.08, 0.09, 0.07])
        assert _detect_trend(curvatures) == "stable"

    def test_decreasing_curvature_is_converging(self):
        """Curvature trending down = converging."""
        # Older values high, recent values low
        older = np.full(5, 0.3)
        recent = np.full(5, 0.1)
        curvatures = np.concatenate([older, recent])
        assert _detect_trend(curvatures) == "converging"

    def test_increasing_curvature_is_diverging(self):
        """Curvature trending up = diverging."""
        older = np.full(5, 0.1)
        recent = np.full(5, 0.3)
        curvatures = np.concatenate([older, recent])
        assert _detect_trend(curvatures) == "diverging"

    def test_stable_curvature(self):
        """Flat curvature = stable."""
        curvatures = np.full(10, 0.1)
        assert _detect_trend(curvatures) == "stable"


# ---------------------------------------------------------------------------
# TrajectoryComputer
# ---------------------------------------------------------------------------


class TestTrajectoryComputer:
    def test_empty_signal(self):
        """Empty trajectory returns insufficient_data."""
        tc = TrajectoryComputer()
        signal = tc.compute_signal()
        assert signal["trend"] == "insufficient_data"
        assert signal["delta_kappa"] is None
        assert signal["turn_count"] == 0

    def test_few_embeddings_insufficient(self):
        """Fewer than MIN_EMBEDDINGS_FOR_CURVATURE returns insufficient_data."""
        tc = TrajectoryComputer()
        for _ in range(MIN_EMBEDDINGS_FOR_CURVATURE - 1):
            tc.add_embedding(np.random.randn(16))
        signal = tc.compute_signal()
        assert signal["trend"] == "insufficient_data"
        assert signal["turn_count"] == MIN_EMBEDDINGS_FOR_CURVATURE - 1

    def test_enough_embeddings_produces_signal(self):
        """With enough embeddings, a signal is produced."""
        tc = TrajectoryComputer()
        for _ in range(10):
            tc.add_embedding(np.random.randn(16))
        signal = tc.compute_signal()
        assert signal["trend"] in ("converging", "diverging", "stable")
        assert signal["delta_kappa"] is not None
        assert signal["turn_count"] == 10
        assert signal["curvature_std"] is not None

    def test_max_embeddings_cap(self):
        """Buffer doesn't grow beyond max_embeddings."""
        tc = TrajectoryComputer(max_embeddings=10)
        for _ in range(20):
            tc.add_embedding(np.random.randn(8))
        assert len(tc.embeddings) == 10

    def test_linear_trajectory_converging(self):
        """Linear trajectory should be detected as converging."""
        tc = TrajectoryComputer()
        direction = np.random.randn(16)
        direction /= np.linalg.norm(direction)
        for i in range(15):
            tc.add_embedding(direction * i * 0.1)
        signal = tc.compute_signal()
        # Linear = zero curvature = converging
        assert signal["delta_kappa"] is not None
        assert signal["delta_kappa"] < 0.01

    def test_save_load_roundtrip(self):
        """Embeddings survive save/load cycle."""
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = Path(f.name)

        try:
            tc = TrajectoryComputer()
            for _ in range(5):
                tc.add_embedding(np.random.randn(16))
            original_signal = tc.compute_signal()
            tc.save(path)

            tc2 = TrajectoryComputer.load(path)
            assert len(tc2.embeddings) == 5
            loaded_signal = tc2.compute_signal()
            assert loaded_signal["turn_count"] == original_signal["turn_count"]
            if original_signal["delta_kappa"] is not None:
                assert abs(loaded_signal["delta_kappa"] - original_signal["delta_kappa"]) < 1e-6
        finally:
            path.unlink(missing_ok=True)

    def test_load_missing_file(self):
        """Loading from nonexistent file returns empty trajectory."""
        tc = TrajectoryComputer.load(Path("/tmp/nonexistent-sense-test.npy"))
        assert len(tc.embeddings) == 0

    def test_clear(self):
        """Clear resets the buffer."""
        tc = TrajectoryComputer()
        for _ in range(5):
            tc.add_embedding(np.random.randn(8))
        tc.clear()
        assert len(tc.embeddings) == 0
        assert tc.compute_signal()["trend"] == "insufficient_data"

    def test_save_empty_is_noop(self):
        """Saving empty trajectory doesn't create a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.npy"
            tc = TrajectoryComputer()
            tc.save(path)
            assert not path.exists()


# ---------------------------------------------------------------------------
# SessionState integration
# ---------------------------------------------------------------------------


class TestSessionStateTrajectory:
    def test_trajectory_signal_persists(self):
        """trajectory_signal field round-trips through save/load."""
        from sense_mcp.session import SessionState

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            state = SessionState()
            state.trajectory_signal = {
                "delta_kappa": 0.123,
                "trend": "converging",
                "turn_count": 7,
                "curvature_std": 0.05,
            }
            state.save(path)

            loaded = SessionState.load(path)
            assert loaded.trajectory_signal["trend"] == "converging"
            assert abs(loaded.trajectory_signal["delta_kappa"] - 0.123) < 1e-6
            assert loaded.trajectory_signal["turn_count"] == 7
        finally:
            Path(path).unlink(missing_ok=True)

    def test_trajectory_signal_default_empty(self):
        """Default trajectory_signal is empty dict."""
        from sense_mcp.session import SessionState
        state = SessionState()
        assert state.trajectory_signal == {}

    def test_backward_compatible_load(self):
        """Loading state without trajectory_signal field works."""
        from sense_mcp.session import SessionState

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            # Old format without trajectory_signal
            json.dump({"surfaced": {}, "queries": [], "last_query_time": 0.0, "last_results": []}, f)
            path = f.name

        try:
            loaded = SessionState.load(path)
            assert loaded.trajectory_signal == {}
        finally:
            Path(path).unlink(missing_ok=True)
