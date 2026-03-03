# TRACE_2026-03-03_package-restructure-uvx-install

## Constraint Acknowledgement

This trace is written under the constraints defined in EarthianLabs root CLAUDE.md.

- **Irreversibility constraints respected:** YES
- **Metabolic governors stressed:** LOW
- **Any overrides?** None. Session was focused structural work with a clear stopping point.

## Local Context

sense-mcp was previously a flat collection of top-level modules (`server.py`, `config.py`) alongside a README and pyproject.toml. This structure blocked `uv build` with "Unable to determine which files to ship inside the wheel" — hatchling could not identify a package boundary. The session resolved this blocker by converting sense-mcp into a proper Python package, making it installable via `uvx` or `uv tool install` without requiring a local clone.

This is load-bearing infrastructure for the broader research mission: Sense (ambient relevance injection) becoming accessible to collaborators outside the EarthianLabs ecosystem requires a clean, one-command install path.

Prior state:
- `server.py` at repo root
- `config.py` at repo root
- `.claude/hooks/sense-auto-query.py` using a `sys.path.insert` hack to import server.py

Post-session state:
- `sense_mcp/` Python package with `__init__.py`, `server.py`, `config.py`, `hook.py`
- Hook logic extracted into `sense_mcp/hook.py` — importable cleanly
- Ecosystem hook at `.claude/hooks/sense-auto-query.py` updated to `from sense_mcp import server as sense_server`
- `pyproject.toml` entry points updated: `sense-mcp = "sense_mcp.server:mcp.run"` and `sense-mcp-hook = "sense_mcp.hook:main"`
- `uv build` verified passing; wheel ships all 4 package files
- Version bumped 0.1.0 → 0.1.1; tagged and released at https://github.com/m3data/sense-mcp/releases/tag/v0.1.1

## Decisions Made

- **Package name `sense_mcp` (underscore)** — Python convention, used in pyproject.toml and __init__.py. Reversible in principle, but now embedded in a tagged release and any downstream installs. Treat as soft-irreversible.
- **Config path resolution: repo root (parent.parent) then CWD fallback** (reversible) — `_find_config()` and `_config_dir` updated in `sense_mcp/config.py` to work correctly for both editable installs (where config lives two levels up from the package dir) and uvx installs (where CWD fallback is appropriate). Can be tuned further without breaking the API.
- **Hook extracted into package as `sense_mcp/hook.py`** (reversible) — Both the package version and the ecosystem hook file at `.claude/hooks/sense-auto-query.py` exist. The ecosystem hook is the live one; the package version makes the hook shippable to external users. No functional duplication issue currently.
- **Version bump 0.1.0 → 0.1.1** (irreversible — tagged and pushed) — Chose a patch bump rather than minor because this is a structural refactor with no new user-facing features. The restructure is a packaging fix, not a capability addition.
- **README quick start rewritten for uvx/uv tool install** (reversible) — Manual/development install moved to a secondary section. Aligns README to the install path most collaborators will use.

## Compression Summary

sense-mcp is now a proper Python package at `sense_mcp/`. The blocker was flat module layout; the fix was creating the package directory and fixing import paths. The ecosystem hook at `.claude/hooks/sense-auto-query.py` was also updated as part of this but lives in the EarthianLabs root repo and needs its own commit there.

The running MCP server process is still using the old module path — it needs a restart for the import change to take effect. Kill with `pgrep -f "sense-mcp/server.py"` (or the new path `sense_mcp/server.py`).

Key file locations:
- Package: `/Users/m3untold/Code/EarthianLabs/sense-mcp/sense_mcp/`
- Entry point: `sense_mcp/server.py` — `mcp.run`
- Hook entry point: `sense_mcp/hook.py` — `main`
- Config: `sense_mcp/config.py` — `get_config()`
- Ecosystem hook: `/Users/m3untold/Code/EarthianLabs/.claude/hooks/sense-auto-query.py`
- Release: https://github.com/m3data/sense-mcp/releases/tag/v0.1.1

Research connections: RQ2 (conditions that cultivate adaptive capacity — installable infrastructure lowers the barrier for collaborators to adopt Sense), RQ5 (sensing and supporting — Sense is now accessible beyond the local ecosystem without a local clone).

## Residue / Open Tensions

- **Ecosystem hook needs a separate EarthianLabs root commit.** `.claude/hooks/sense-auto-query.py` was updated this session but lives outside the sense-mcp repo. It references `from sense_mcp import server as sense_server` which requires sense-mcp to be installed in the active Python environment. This is a latent fragility — if the hook runs in an environment where sense-mcp isn't installed, it will fail silently or with an import error.
- **PyPI publishing deferred.** The package is on GitHub but not PyPI. Until published, `uvx --from git+https://github.com/m3data/sense-mcp.git sense-mcp` is the install path. The README should reflect this accurately — worth verifying.
- **Hook cold-start latency is unresolved.** `uvx` is too slow for a hook invocation context (cold-start ~2–3 seconds). The README notes this and recommends `uv tool install` instead, but the friction is real for new users who try `uvx` first. No fix in scope; named here for continuity.
- **skills/ files not updated.** `sense-mcp/skills/` (if present) may still reference old flat module paths or old install instructions. Marked out of scope for this session but worth a sweep before any major collaborator onboarding.
- **MCP server restart required.** The live sense-mcp MCP process is not yet running the restructured package. Restart is a manual step — easy to forget, creates confusion if queries appear to work but the old code is actually running.
