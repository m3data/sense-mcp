# Changelog

All notable changes to Sense are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/).

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
