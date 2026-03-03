"""Tests for compute_decay() — REQ-001.

Covers all 8 scenarios:
  1. Evergreen source type returns 1.0
  2. Today's date returns 1.0
  3. One half-life ago returns ~0.5 (tolerance ±0.05)
  4. Two half-lives ago returns ~0.25 (tolerance ±0.05)
  5. Very old content floors at decay_floor
  6. Missing date returns decay_floor
  7. Invalid date returns decay_floor
  8. Future date clamped to now returns 1.0
"""

from datetime import datetime, timedelta, timezone

from sense_mcp.server import compute_decay


def _date_str(delta_days: int) -> str:
    """YYYY-MM-DD for a date delta_days relative to today (UTC)."""
    dt = datetime.now(timezone.utc) + timedelta(days=delta_days)
    return dt.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Scenario 1: Evergreen — source type not in half_lives
# ---------------------------------------------------------------------------

def test_evergreen_returns_1(test_env):
    """Source types without a configured half-life return 1.0 (no decay)."""
    cfg = test_env
    assert cfg.get_half_life("project_claude") is None, (
        "test_config.toml must not list project_claude in [decay.half_lives]"
    )
    result = compute_decay("project_claude", "2020-01-01")
    assert result == 1.0


# ---------------------------------------------------------------------------
# Scenario 2: Today returns 1.0
# ---------------------------------------------------------------------------

def test_today_returns_1():
    """A document dated today has zero age — decay multiplier is 1.0."""
    result = compute_decay("trace", _date_str(0))
    assert result == 1.0


# ---------------------------------------------------------------------------
# Scenario 3: One half-life ago returns ~0.5
# ---------------------------------------------------------------------------

def test_one_half_life_returns_half(test_env):
    """A document aged exactly one half-life returns approximately 0.5."""
    cfg = test_env
    half_life = cfg.get_half_life("trace")
    assert half_life is not None
    result = compute_decay("trace", _date_str(-half_life))
    assert abs(result - 0.5) <= 0.05


# ---------------------------------------------------------------------------
# Scenario 4: Two half-lives ago returns ~0.25
# ---------------------------------------------------------------------------

def test_two_half_lives_returns_quarter(test_env):
    """A document aged two half-lives returns approximately 0.25."""
    cfg = test_env
    half_life = cfg.get_half_life("trace")
    assert half_life is not None
    result = compute_decay("trace", _date_str(-2 * half_life))
    assert abs(result - 0.25) <= 0.05


# ---------------------------------------------------------------------------
# Scenario 5: Very old content floors at decay_floor
# ---------------------------------------------------------------------------

def test_very_old_floors_at_decay_floor(test_env):
    """A very old document is floored at decay_floor, not driven below it."""
    cfg = test_env
    result = compute_decay("trace", "2000-01-01")
    assert result == cfg.decay_floor


# ---------------------------------------------------------------------------
# Scenario 6: Missing date returns decay_floor
# ---------------------------------------------------------------------------

def test_missing_date_returns_floor(test_env):
    """None date_str treats the document as undated and returns decay_floor."""
    cfg = test_env
    result = compute_decay("trace", None)
    assert result == cfg.decay_floor


# ---------------------------------------------------------------------------
# Scenario 7: Invalid date returns decay_floor
# ---------------------------------------------------------------------------

def test_invalid_date_returns_floor(test_env):
    """An unparseable date string returns decay_floor."""
    cfg = test_env
    result = compute_decay("trace", "not-a-date")
    assert result == cfg.decay_floor


# ---------------------------------------------------------------------------
# Scenario 8: Future date clamped to now returns 1.0
# ---------------------------------------------------------------------------

def test_future_date_clamped_to_now():
    """A future-dated document is clamped to now — age is 0 → decay is 1.0."""
    result = compute_decay("trace", _date_str(30))
    assert result == 1.0
