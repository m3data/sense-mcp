"""Sense configuration — loads sense.toml and provides typed access.

Resolution order:
  1. SENSE_CONFIG env var (absolute path to TOML file)
  2. sense.toml in repo root (parent of sense_mcp/ package dir)
  3. sense.toml in current working directory
  4. Built-in defaults (minimal, functional for any markdown project)

Root path resolution:
  1. SENSE_ROOT env var
  2. [corpus] root in config (absolute or relative to config file dir)
  3. Parent of the repo root (sense-mcp/../)
"""

import os
import re
import tomllib
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults — enough to work with any markdown project out of the box
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "server": {
        "name": "Sense",
    },
    "embedding": {
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "max_input_tokens": 8000,
        "batch_size": 256,
    },
    "corpus": {
        "root": "..",
        "database": "sense.db",
        "api_key_env": "OPENAI_API_KEY",
        "env_file": ".env",
        "extensions": [".md", ".txt", ".rst", ".py"],
        "excluded_dirs": [
            "node_modules", "_site", ".venv", "venv", "__pycache__",
            ".git", ".tox", "dist", "build", "egg-info",
        ],
        "excluded_paths": [],
    },
    "decay": {
        "floor": 0.1,
        "half_lives": {
            "trace": 30,
            "teaching": 60,
            "documentation": 90,
            "code": 90,
        },
    },
    "session": {
        "surfaced_cap": 500,
        "max_queries": 50,
    },
    "mode": {
        "history_path": "~/.vibe-harness/mode-history.jsonl",
        "min_type_slots": 2,
        "diversity_slots": {
            "wide": [4, 3, 3],
            "narrow": [7, 2, 1],
            "suppress": [3, 0, 0],
        },
        "profiles": {
            "explore": {
                "source_type_multipliers": {
                    "code": 0.7, "research": 1.3, "documentation": 1.0,
                    "trace": 1.0, "teaching": 1.0, "project_claude": 1.0,
                    "reference": 1.0,
                },
                "cross_project_boost": 1.4,
                "already_surfaced_penalty": 0.75,
                "diversity_profile": "wide",
            },
            "build": {
                "source_type_multipliers": {
                    "code": 1.5, "research": 0.7, "documentation": 1.3,
                    "trace": 1.0, "teaching": 0.8, "project_claude": 1.2,
                    "reference": 1.0,
                },
                "cross_project_boost": 0.8,
                "already_surfaced_penalty": 0.80,
                "diversity_profile": "narrow",
            },
            "think-with": {
                "source_type_multipliers": {
                    "code": 0.6, "research": 1.5, "documentation": 1.0,
                    "trace": 1.0, "teaching": 1.1, "project_claude": 1.0,
                    "reference": 1.2,
                },
                "cross_project_boost": 1.3,
                "already_surfaced_penalty": 0.75,
                "diversity_profile": "wide",
            },
            "ship": {
                "source_type_multipliers": {
                    "code": 1.4, "research": 0.5, "documentation": 1.2,
                    "trace": 1.0, "teaching": 0.6, "project_claude": 1.1,
                    "reference": 1.0,
                },
                "cross_project_boost": 0.7,
                "already_surfaced_penalty": 0.85,
                "diversity_profile": "narrow",
            },
            "cool-off": {
                "source_type_multipliers": {
                    "code": 0.2, "research": 0.5, "documentation": 0.3,
                    "trace": 0.3, "teaching": 0.4, "project_claude": 0.3,
                    "reference": 0.5,
                },
                "cross_project_boost": 1.0,
                "already_surfaced_penalty": 0.65,
                "diversity_profile": "suppress",
            },
        },
    },
    "classification": {
        "default": "documentation",
        "rules": [
            {"type": "trace", "filename": r"^TRACE_|DEV_UPDATE", "case_insensitive": True},
            {"type": "project_claude", "filename": r"CLAUDE\.md$", "case_insensitive": True},
            {"type": "code", "extension": [".py"]},
        ],
    },
}


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins for leaf values."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# Config object
# ---------------------------------------------------------------------------

class SenseConfig:
    """Typed access to Sense configuration."""

    def __init__(self, config_path: Path | None = None):
        self._raw = dict(_DEFAULTS)
        # Default config_dir: repo root (parent of sense_mcp/ package).
        # For editable installs this is the sense-mcp/ checkout.
        # For uvx/pip installs this lands in site-packages (harmless —
        # paths fall through to SENSE_ROOT or CWD).
        self._config_dir = Path(__file__).resolve().parent.parent

        if config_path is None:
            config_path = self._find_config()

        if config_path and config_path.exists():
            with open(config_path, "rb") as f:
                user_config = tomllib.load(f)
            self._raw = _deep_merge(self._raw, user_config)
            self._config_dir = config_path.parent

        self._resolve_paths()
        self._compile_classification_rules()

    def _find_config(self) -> Path | None:
        """Find config file via env var or conventional location."""
        env_path = os.environ.get("SENSE_CONFIG")
        if env_path:
            return Path(env_path).resolve()

        # Repo root (parent of sense_mcp/ package dir) — works for
        # editable installs and local dev
        repo_root = Path(__file__).resolve().parent.parent / "sense.toml"
        if repo_root.exists():
            return repo_root

        # CWD fallback — works for uvx installs where the user has a
        # sense.toml in their project root
        cwd = Path.cwd() / "sense.toml"
        if cwd.exists():
            return cwd

        return None

    def _resolve_paths(self):
        """Resolve root, database, env_file to absolute paths."""
        corpus = self._raw["corpus"]

        # Root path
        env_root = os.environ.get("SENSE_ROOT")
        if env_root:
            self.root = Path(env_root).resolve()
        else:
            root_str = corpus.get("root", "..")
            root_path = Path(root_str)
            if root_path.is_absolute():
                self.root = root_path
            else:
                self.root = (self._config_dir / root_path).resolve()

        # Database path
        db_str = corpus.get("database", "sense.db")
        db_path = Path(db_str)
        if db_path.is_absolute():
            self.db_path = db_path
        else:
            self.db_path = (self._config_dir / db_path).resolve()

        # Env file (relative to corpus root)
        env_file_str = corpus.get("env_file", ".env")
        env_path = Path(env_file_str)
        if env_path.is_absolute():
            self.env_file = env_path
        else:
            self.env_file = self.root / env_path

        # Mode history path
        history_str = self._raw["mode"]["history_path"]
        self.mode_history_path = Path(os.path.expanduser(history_str))

    def _compile_classification_rules(self):
        """Pre-compile classification rules into matchers."""
        self._classification_rules = []
        raw_rules = self._raw["classification"].get("rules", [])

        for rule in raw_rules:
            source_type = rule["type"]
            matcher = _build_matcher(rule)
            self._classification_rules.append((source_type, matcher))

        self._classification_default = self._raw["classification"].get(
            "default", "documentation"
        )

    # --- Property accessors ---

    @property
    def server_name(self) -> str:
        return self._raw["server"]["name"]

    @property
    def embedding_model(self) -> str:
        return self._raw["embedding"]["model"]

    @property
    def embedding_dims(self) -> int:
        return self._raw["embedding"]["dimensions"]

    @property
    def max_input_tokens(self) -> int:
        return self._raw["embedding"]["max_input_tokens"]

    @property
    def batch_size(self) -> int:
        return self._raw["embedding"]["batch_size"]

    @property
    def api_key_env(self) -> str:
        return self._raw["corpus"]["api_key_env"]

    @property
    def extensions(self) -> set[str]:
        return set(self._raw["corpus"]["extensions"])

    @property
    def excluded_dirs(self) -> set[str]:
        return set(self._raw["corpus"]["excluded_dirs"])

    @property
    def excluded_paths(self) -> set[Path]:
        raw = self._raw["corpus"].get("excluded_paths", [])
        return {self.root / p for p in raw}

    @property
    def half_lives(self) -> dict[str, int | None]:
        """Map source_type → half-life in days. Missing = evergreen (None)."""
        return self._raw["decay"]["half_lives"]

    @property
    def decay_floor(self) -> float:
        return self._raw["decay"]["floor"]

    @property
    def surfaced_cap(self) -> int:
        return self._raw["session"]["surfaced_cap"]

    @property
    def max_queries(self) -> int:
        return self._raw["session"]["max_queries"]

    @property
    def mode_profiles(self) -> dict:
        return self._raw["mode"]["profiles"]

    @property
    def min_type_slots(self) -> int:
        return self._raw["mode"]["min_type_slots"]

    @property
    def diversity_slots(self) -> dict[str, tuple[int, int, int]]:
        raw = self._raw["mode"]["diversity_slots"]
        return {k: tuple(v) for k, v in raw.items()}

    def classify_source_type(self, path: Path) -> str:
        """Classify a file using config-driven rules. First match wins."""
        rel = str(path.relative_to(self.root))
        for source_type, matcher in self._classification_rules:
            if matcher(path, rel):
                return source_type
        return self._classification_default

    def is_evergreen(self, source_type: str) -> bool:
        return source_type not in self.half_lives

    def get_half_life(self, source_type: str) -> int | None:
        return self.half_lives.get(source_type)


# ---------------------------------------------------------------------------
# Classification rule matchers
# ---------------------------------------------------------------------------

def _build_matcher(rule: dict):
    """Build a callable (path, rel_str) -> bool from a rule dict."""
    matchers = []

    if "filename" in rule:
        flags = re.IGNORECASE if rule.get("case_insensitive") else 0
        pattern = re.compile(rule["filename"], flags)
        matchers.append(lambda p, r, _pat=pattern: bool(_pat.search(p.name)))

    if "path_contains" in rule:
        needle = rule["path_contains"]
        ci = rule.get("case_insensitive", False)
        if ci:
            needle_lower = needle.lower()
            matchers.append(lambda p, r, _n=needle_lower: _n in r.lower())
        else:
            matchers.append(lambda p, r, _n=needle: _n in r)

    if "path_segment" in rule:
        # Match against directory segments (avoids substring collisions)
        segments = rule["path_segment"]
        if isinstance(segments, str):
            segments = [segments]
        ci = rule.get("case_insensitive", True)  # default CI for segments

        def _segment_match(p, r, _segs=segments, _ci=ci):
            check = r.lower() if _ci else r
            for seg in _segs:
                seg_check = seg.lower() if _ci else seg
                needle = seg_check + "/"
                if check.startswith(needle) or ("/" + needle) in check:
                    return True
            return False

        matchers.append(_segment_match)

    if "extension" in rule:
        exts = rule["extension"]
        if isinstance(exts, str):
            exts = [exts]
        ext_set = set(exts)
        matchers.append(lambda p, r, _e=ext_set: p.suffix in _e)

    if not matchers:
        # Rule with no matcher never fires
        return lambda p, r: False

    # All matchers in a rule must pass (AND logic)
    # But typically each rule has exactly one matcher
    def _combined(p, r, _ms=matchers):
        return all(m(p, r) for m in _ms)

    return _combined


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_config: SenseConfig | None = None


def get_config() -> SenseConfig:
    """Get or create the global config singleton."""
    global _config
    if _config is None:
        _config = SenseConfig()
    return _config


def reload_config(config_path: Path | None = None) -> SenseConfig:
    """Force reload config (useful for testing)."""
    global _config
    _config = SenseConfig(config_path)
    return _config
