# Changelog

All notable changes to Sense are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.4.2] - 2026-05-09

### Changed

- **Build/ship mode reframe** — multipliers rewritten to match what these modes actually deliver in the corpus. Build-in-Sense and ship-in-Sense are now "orient me to the code's context" (specs, prior decisions, conventions, mistakes), not "show me code". Filesystem tools (Read/Grep/Glob) cover code itself; Sense provides the surrounding context.
  - **Build**: `code` 1.5 → 0.8 (corpus-blocked: only ~3% of indexed chunks). `project_claude` 1.2 → 1.4. `reference` 1.0 → 1.2. `trace` 1.0 → 1.2. `research` 0.7 → 1.0 (data showed it earning its place at 18.8% of useful retrievals). `cross_project_boost` 0.8 → 0.9 (slight widening for adjacent prior decisions).
  - **Ship**: `code` 1.4 → 0.8 (same corpus reasoning). `project_claude` 1.1 → 1.4. `trace` 1.0 → 1.3 (release work hinges on what broke last time). `reference` 1.0 → 1.2. Stays narrow.
- **New `balanced` diversity profile** `[5, 3, 2]` — between `narrow` and `wide`. Build now uses `balanced`: anchored on the current thread but with deliberate adjacency for prior decisions and cross-project patterns.

### Why

Internal consistency check on 7,545 feedback rows (2026-05-09) showed build/ship multipliers were partially working against their own spec. Code multipliers (1.5×, 1.4×) couldn't deliver because the indexed corpus is only 3.3% code (Python only — Rust/Swift/JS/TS are not in the extension whitelist). Build/ship modes were *effectively* surfacing documentation, project_claude, research, and traces — sensible "orient me" content, but not what the spec claimed they did. The fix realigns spec with what's actually useful, rather than tuning labels against an instrument working against its own design intent.

The cross-project code-pattern gap (the original case for build's code boost) is real but separate. Decision deferred: widen the corpus to include Rust/Swift/JS as a scoped experiment after this reframe lands.

### Connects to

`DESIGN_DIRECTION_relevance-realisation.md` — Tier 1 mode-aware Sense spec is updated to reflect the reframe. Original build description ("boost code and technical documentation") was specced from intuition; revised description matches empirical behaviour.

## [0.4.1] - 2026-04-16

### Added

- **Bias breakdown in Companion dashboard** — each result now shows net bias contribution (positive/negative) and active signal badges: `seen` (resurfacing penalty with value), `circling` (topic recurrence), `x-proj` (cross-project), and mode weight multiplier.
- **Educational tooltips** — hover any score, source type, signal badge, or bias value for a plain-language explanation of what it means and how it affects ranking.
- **Signal legend** — inline guide above the query timeline anchors badge meanings without requiring hover.
- **Bias data pipeline** — `format_surfaced_result()` and `_extract_bias_fields()` shared helpers in `session.py` ensure both server and hook paths write consistent bias fields to session state. Previously, bias signals were computed during ranking but discarded before reaching the dashboard.

### Changed

- Signal badge labels: `resurf` → `seen`, `cross` → `x-proj` — more self-explaining at a glance.
- Trajectory label: `dk=` → `drift:` — lower barrier for new users, with delta-kappa explanation on hover.
- Feedback empty state copy now tells users what they can do, not what the system does internally.
- `__init__.py` version synced (was stuck at 0.1.1 since package restructure).

## [0.4.0] - 2026-04-16

### Changed

- **Additive bias model** — mode-aware scoring replaced from multiplicative to additive. Contextual signals (source type preferences, cross-project, resurfacing, circling) are summed and weighted by a small alpha (0.05), bounding how much mode can shift ranking relative to base relevance. Prevents the failure mode where stacked multipliers buried highly relevant results behind irrelevant content.
- **Resurfacing penalty** — replaced exponential compounding (`0.75^N`, floored at 0.05) with linear decay (`0.75 - 0.05*(N-1)`, floored at 0.50). A file surfaced 6 times retains 50% of its score, not 5%.
- **Adaptive diversity slots** — when top results cluster above 0.55 similarity, confirmation slots expand (up to 70%) at the expense of divergence/serendipity. Diversity injection only displaces relevant results when the result set is genuinely ambiguous.
- **Trajectory-aware resurfacing** — when conversation trajectory is diverging, resurfacing penalty is halved (recurrence = coherent anchoring). When converging, penalty applies normally (recurrence = circling).
- **Anti-entrainment for wide profiles** — when converging in already-wide modes (think-with, explore), slot distribution shifts toward more serendipity. Previously only narrow→wide transition worked.

### Added

- **Epistemic weight** — configurable prior on content importance. Foundational documents (research frames, primary publications, reference material) get a base multiplier applied alongside cosine similarity and temporal decay. Configured in `sense.toml` under `[epistemic_weight]` using the same matcher syntax as classification rules.
- **File deduplication** in base search — prevents same file occupying multiple result slots.
- **Retrieval eval suite** — 14 ground-truth queries across modes with precision@k tracking, ranking violation detection, and before/after comparison. Run via `python evals/eval_runner.py`.
- **`specs/` classification rule** — SPEC files are now classified as `research` (evergreen).

### Fixed

- Hook penalty logic aligned with server — same additive model and linear penalty curve.

### Eval results

Baseline (v0.3.0): 25% mean recall, 1/14 perfect queries.
After all changes: 78.6% mean recall, 10/14 perfect queries.
Think-with mode: 28.6% → 92.9%.

## [0.3.0] - 2026-03-10

Trajectory tracking, relevance feedback, companion dashboard.

## [0.2.0] - 2026-03-07

Shared session state, stratified pool sampling.

## [0.1.1] - 2026-02-28

Package restructure, uvx install support.

## [0.1.0] - 2026-02-20

Initial release. Semantic search with temporal decay, mode-aware retrieval, auto-query hook.
