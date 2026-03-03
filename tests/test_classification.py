"""Tests for classify_source_type() and related config functions.

Covers:
  REQ-004 — source type classification via config-driven rules
  REQ-007 — config helper functions (_deep_merge, reload_config,
             is_evergreen, get_half_life)

All classification tests run against test_config.toml which defines:
  trace        — filename matches ^TRACE_|DEV_UPDATE (case-insensitive)
  project_claude — filename matches CLAUDE.md$ (case-insensitive)
  code         — extension in [.py]
  default      — "documentation"

Decay half-lives in test_config.toml:
  trace=30, documentation=90, code=90  (teaching/research/project_claude absent → evergreen)
"""

from pathlib import Path

import sense_mcp.config as config_module
from sense_mcp.config import SenseConfig, _deep_merge, reload_config
from tests.conftest import CORPUS_DIR, TEST_CONFIG_PATH


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------

class TestDeepMerge:
    def test_non_overlapping_keys_preserved(self):
        base = {"a": 1, "b": {"x": 10}}
        override = {"c": 3}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": {"x": 10}, "c": 3}

    def test_leaf_value_overridden(self):
        base = {"a": 1}
        override = {"a": 99}
        result = _deep_merge(base, override)
        assert result["a"] == 99

    def test_nested_dict_merged_recursively(self):
        base = {"decay": {"floor": 0.1, "half_lives": {"trace": 30}}}
        override = {"decay": {"half_lives": {"trace": 14, "code": 60}}}
        result = _deep_merge(base, override)
        assert result["decay"]["floor"] == 0.1          # preserved
        assert result["decay"]["half_lives"]["trace"] == 14   # overridden
        assert result["decay"]["half_lives"]["code"] == 60    # added

    def test_base_not_mutated(self):
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        _deep_merge(base, override)
        assert base["a"]["b"] == 1

    def test_list_replaced_not_merged(self):
        base = {"exts": [".md", ".txt"]}
        override = {"exts": [".py"]}
        result = _deep_merge(base, override)
        assert result["exts"] == [".py"]


# ---------------------------------------------------------------------------
# reload_config
# ---------------------------------------------------------------------------

class TestReloadConfig:
    def test_reload_creates_new_instance(self, monkeypatch):
        monkeypatch.setattr(config_module, "_config", None)
        cfg1 = reload_config(TEST_CONFIG_PATH)
        cfg2 = reload_config(TEST_CONFIG_PATH)
        assert cfg1 is not cfg2

    def test_reload_updates_global_singleton(self, monkeypatch):
        monkeypatch.setattr(config_module, "_config", None)
        cfg = reload_config(TEST_CONFIG_PATH)
        assert config_module._config is cfg

    def test_reload_replaces_existing_singleton(self, monkeypatch):
        monkeypatch.setattr(config_module, "_config", None)
        old = reload_config(TEST_CONFIG_PATH)
        new = reload_config(TEST_CONFIG_PATH)
        assert config_module._config is new
        assert config_module._config is not old


# ---------------------------------------------------------------------------
# classify_source_type — using test_config.toml rules via test_env fixture
# ---------------------------------------------------------------------------

class TestClassifySourceType:
    def test_trace_prefix_classifies_as_trace(self, test_env):
        path = CORPUS_DIR / "TRACE_2025-01-01_sample-session.md"
        assert test_env.classify_source_type(path) == "trace"

    def test_trace_in_subdirectory_classifies_as_trace(self, test_env):
        path = CORPUS_DIR / "project-b" / "TRACE_2026-01-15.md"
        assert test_env.classify_source_type(path) == "trace"

    def test_dev_update_prefix_classifies_as_trace(self, test_env):
        # DEV_UPDATE is part of the trace filename pattern in test_config.toml
        path = CORPUS_DIR / "DEV_UPDATE_2025-03-01.md"
        assert test_env.classify_source_type(path) == "trace"

    def test_claude_md_classifies_as_project_claude(self, test_env):
        path = CORPUS_DIR / "CLAUDE.md"
        assert test_env.classify_source_type(path) == "project_claude"

    def test_nested_claude_md_classifies_as_project_claude(self, test_env):
        path = CORPUS_DIR / "project-a" / "CLAUDE.md"
        assert test_env.classify_source_type(path) == "project_claude"

    def test_claude_md_match_is_case_insensitive(self, test_env):
        # Rule has case_insensitive=true; lowercase variant should still match
        path = CORPUS_DIR / "claude.md"
        assert test_env.classify_source_type(path) == "project_claude"

    def test_py_extension_classifies_as_code(self, test_env):
        path = CORPUS_DIR / "sample_module.py"
        assert test_env.classify_source_type(path) == "code"

    def test_nested_py_classifies_as_code(self, test_env):
        path = CORPUS_DIR / "project-a" / "server.py"
        assert test_env.classify_source_type(path) == "code"

    def test_unmatched_md_classifies_as_default(self, test_env):
        path = CORPUS_DIR / "README.md"
        assert test_env.classify_source_type(path) == "documentation"

    def test_unmatched_root_file_classifies_as_default(self, test_env):
        path = CORPUS_DIR / "root-file.md"
        assert test_env.classify_source_type(path) == "documentation"

    def test_first_rule_wins_over_later_rules(self, test_env):
        # TRACE_foo.py matches both the trace filename rule and the code
        # extension rule; trace rule is listed first → must return "trace"
        path = CORPUS_DIR / "TRACE_foo.py"
        assert test_env.classify_source_type(path) == "trace"


# ---------------------------------------------------------------------------
# is_evergreen / get_half_life
# ---------------------------------------------------------------------------

class TestHalfLifeHelpers:
    """Tests use test_config.toml half_lives: trace=30, documentation=90, code=90.
    Types absent from half_lives (teaching, research, project_claude, reference)
    are evergreen.
    """

    def test_trace_not_evergreen(self, test_env):
        assert test_env.is_evergreen("trace") is False

    def test_documentation_not_evergreen(self, test_env):
        assert test_env.is_evergreen("documentation") is False

    def test_code_not_evergreen(self, test_env):
        assert test_env.is_evergreen("code") is False

    def test_teaching_has_default_half_life(self, test_env):
        # teaching=60 is in _DEFAULTS; test_config doesn't override it,
        # so it survives the deep merge — teaching is NOT evergreen
        assert test_env.is_evergreen("teaching") is False
        assert test_env.get_half_life("teaching") == 60

    def test_research_is_evergreen(self, test_env):
        assert test_env.is_evergreen("research") is True

    def test_project_claude_is_evergreen(self, test_env):
        assert test_env.is_evergreen("project_claude") is True

    def test_reference_is_evergreen(self, test_env):
        assert test_env.is_evergreen("reference") is True

    def test_get_half_life_trace(self, test_env):
        assert test_env.get_half_life("trace") == 30

    def test_get_half_life_documentation(self, test_env):
        assert test_env.get_half_life("documentation") == 90

    def test_get_half_life_code(self, test_env):
        assert test_env.get_half_life("code") == 90

    def test_get_half_life_returns_none_for_evergreen(self, test_env):
        assert test_env.get_half_life("teaching") is None
        assert test_env.get_half_life("research") is None
        assert test_env.get_half_life("unknown_type") is None
