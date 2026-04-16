# Sense

![Repo Status](https://img.shields.io/badge/REPO_STATUS-Active_Research-blue?style=for-the-badge&labelColor=8b5e3c&color=e5dac1)
![Version](https://img.shields.io/badge/VERSION-0.4.0-blue?style=for-the-badge&labelColor=3b82f6&color=1e40af)
![License](https://img.shields.io/badge/LICENSE-Apache_2.0-green?style=for-the-badge&labelColor=10b981&color=047857)
![Python](https://img.shields.io/badge/PYTHON-3.11+-green?style=for-the-badge&labelColor=10b981&color=047857)
![MCP](https://img.shields.io/badge/MCP-stdio-purple?style=for-the-badge&labelColor=7c3aed&color=5b21b6)

Ambient relevance for multi-project ecosystems. Prior work from across your portfolio surfaces in every AI conversation — automatically, weighted by recency, and shaped by what you're doing.

## The problem

You don't work on one project. You work across an ecosystem — research feeding publications, code shaped by decisions made three repos away, a consulting thread that traces back to notes you wrote six months ago. The relevant piece is rarely in the project you're currently looking at.

AI coding agents are capable but contextually blind to this. Memory and RAG tools solve single-project recall. Relevance across a portfolio of projects is a different problem. Dumping context into a prompt without discrimination buries the useful cross-project connections under volume. And search requires you to know what you're looking for — which means it can't surface connections you didn't know existed.

The pain is specific: you're building in one repo and the decision that matters was documented in a session trace from a different project last week. You're writing and the concept you need was explored in research notes you haven't opened in a month. The context is there. It's just never where you are.

## What Sense does

Sense indexes your project portfolio and injects relevant context into every conversation automatically. You don't search. You don't paste. Prior work surfaces as you work.

It runs as an [MCP](https://modelcontextprotocol.io/) server for [Claude Code](https://docs.anthropic.com/en/docs/claude-code), with a companion hook that fires on every prompt. The result: your AI partner has peripheral awareness of the entire portfolio it's working across, not just the repo it's sitting in.

### How it's different

**Ambient, not invoked.** The auto-query hook fires on every prompt. Context arrives without being asked for. This is the primary interaction pattern — not a search box you type into, but a layer that's always running in the background, shaping what's visible.

**Trajectory-aware.** Sense doesn't treat each prompt in isolation. It tracks the semantic trajectory of your conversation — accumulating query embeddings, computing local curvature (delta-kappa) via Frenet-Serret geometry — and blends recent context into each search. When a conversation is converging on a thread, search narrows. When it's diverging, the aperture widens. The query you send is shaped by the queries that came before it.

**Knowledge metabolises.** A session trace from yesterday and a reference document from last year are not equally alive. Sense weights them differently — recent work surfaces more readily, old documentation fades, foundational reference stays evergreen. Different types of content have different half-lives because they are different kinds of knowledge.

When paired with [Vibe Harness](https://github.com/m3data/vibe-harness-mcp), what surfaces also changes based on what you're doing. Exploring widens the aperture — cross-project connections, unexpected adjacencies. Building narrows it to code and documentation in the current project. The same corpus looks different depending on your working mode.

**Source-classified, diversity-structured.** Files are classified into types (traces, code, research, documentation, reference, etc), each with distinct decay rates and mode dispositions. Foundational documents can be given epistemic weight — a prior on content importance that lifts research frames and primary publications above high-volume ephemeral content regardless of query or mode. Results are structured into confirmation slots (highest relevance), divergence slots (what challenges the current frame), and serendipity slots (from projects you weren't looking at). Slot allocation is adaptive: when top results cluster tightly in relevance, confirmation expands and diversity shrinks; when results are spread, diversity earns its full allocation. The goal is productive connections, not just the nearest match.

**Learns from feedback.** Sense auto-labels every result it surfaces (useful or noise) and tracks human corrections. Labels feed back into retrieval weights through a Bayesian prior — files that consistently prove useful surface more readily, persistent noise gets suppressed. The feedback store is append-only with latest-wins semantics for weight calculation, so the full correction history is available as training data.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

### Install and register

**Option A — `uvx` (zero install, runs directly):**

```bash
claude mcp add sense \
  -e OPENAI_API_KEY=sk-... \
  -e SENSE_ROOT=/path/to/your/project \
  -- uvx --from git+https://github.com/m3data/sense-mcp sense-mcp
```

**Option B — `uv tool install` (recommended for hook support):**

```bash
uv tool install sense-mcp --from git+https://github.com/m3data/sense-mcp
claude mcp add sense \
  -e OPENAI_API_KEY=sk-... \
  -e SENSE_ROOT=/path/to/your/project \
  -- sense-mcp
```

Both options register Sense as an MCP server with Claude Code. Option B also installs the `sense-mcp-hook` command for ambient context (below).

### Configure (optional)

For default settings, `OPENAI_API_KEY` and `SENSE_ROOT` env vars are enough. For deeper customisation, create a `sense.toml` in your project root or point to one with `SENSE_CONFIG`:

```bash
cp sense.example.toml sense.toml
```

See `sense.example.toml` for all options: corpus paths, excluded directories, classification rules, decay half-lives, mode profiles.

### Enable ambient context (recommended)

The companion hook fires on every user prompt and injects the top 3 relevant results as `<sense-context>` tags. This is what makes Sense ambient rather than on-demand.

Add to `.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "type": "command",
        "command": "sense-mcp-hook"
      }
    ]
  }
}
```

Requires `uv tool install` (Option B) — the hook runs on every prompt and needs sub-second startup, so `uvx` cold-start is too slow.

The hook gates on prompt length, cooldown, and continuation signals. It opens the SQLite DB in read-only mode and coexists safely with the running MCP server. Session state (resurfacing penalties, query history, cooldown) is shared between the hook and MCP server via a file-locked JSON store, so both callers contribute to and benefit from the same session context.

### Development install

For contributors or local hacking:

```bash
cd sense-mcp
uv venv && source .venv/bin/activate
uv pip install -e .
```

## Companion dashboard

A local web app for observing and correcting Sense's relevance judgments in real time.

```bash
python sense-mcp/dashboard/serve.py
# Open http://localhost:8111
```

The dashboard reads existing data stores (sense.db, session state, trajectory history) and renders:

- **Hit rate** — session-scoped relevance success as a hero metric, colour-coded by health
- **Query timeline** — every hook-fired query with expandable results showing file path, section, similarity score, source type, and label
- **Trajectory signal** — current semantic trajectory (converging/diverging/stable) with delta-kappa values
- **Feedback stream** — chronological log of auto-labels and human corrections

Click any result label to toggle it between useful and noise. Corrections write back to the feedback table as `corrected:<user>` entries and shift retrieval weights from the next query.

If using the Claude Code session hooks, the dashboard auto-starts when a session opens and stops when it closes.

## Tools

### `sense_search`

Search by natural language. Returns ranked results with similarity scores, temporal decay, and content previews. Supports optional filters: `project`, `source_type`, `limit`, `mode`.

### `sense_sync`

Build or update the index. Uses SHA-256 file hashing for change detection — unchanged files are skipped. Safe to run repeatedly.

### `sense_status`

Index statistics: chunk counts by project and source type, total tokens, last sync time.

### `sense_feedback`

Submit relevance feedback for a search result. Accepts `query_text`, `file_path`, `label` (useful/noise), and optional `note`. The dashboard calls this for human corrections; you can also call it directly.

### `sense_feedback_stats`

Summary statistics on collected feedback: total labels, breakdown by source (auto:hook, manual, corrected), correction rate, and per-file weight previews.

## Slash commands

If using Claude Code skills, copy `skills/sense/` and `skills/sense-sync/` into your `.claude/skills/` directory:

- `/sense <query>` — search with optional flags (`--project`, `--type`, `--limit`, `--mode`)
- `/sense` (no args) — auto-synthesizes a query from conversation context
- `/sense-sync` — rebuild the index
- `/sense-sync status` — show index stats

## Configuration reference

### Temporal decay

Content ages out based on source type. Configure half-lives in days:

```toml
[decay]
floor = 0.1  # Old content never fully vanishes

[decay.half_lives]
trace = 30          # Session traces
market-research = 60       # Market research
documentation = 90  # General docs
code = 90           # Source code
# Types not listed are evergreen (no decay)
```

### Classification rules

Rules are evaluated in order. First match wins. Each rule maps files to a source type used for decay and mode scoring.

| Matcher | Description |
|---------|-------------|
| `filename` | Regex against the filename |
| `path_contains` | Substring match in relative path |
| `path_segment` | Directory name(s) as path segments |
| `extension` | File extension(s) |

### Contextual query (trajectory blending)

The hook blends recent conversation context into each search query. Configure in `sense.toml`:

```toml
[hook]
context_window = 5              # Recent queries to blend
context_decay = 0.5             # Exponential decay (older = less weight)
max_context_weight = 0.4        # Cap on context contribution (current message >= 60%)
context_session_timeout = 7200  # Reset after 2 hours of inactivity
```

The trajectory signal adjusts the cap automatically: converging conversations shrink it to 0.2 (tighter focus), diverging conversations expand it to 0.6 (wider context).

### Relevance feedback

Controls how auto-labels and human corrections influence retrieval weights:

```toml
[feedback]
boost_factor = 0.3       # Weight multiplier range
prior = 2.0              # Bayesian prior — higher = more labels needed to shift weights
weight_cache_ttl = 60    # Seconds before recalculating from feedback table
```

### Mode-aware retrieval

When paired with [Vibe Harness](https://github.com/m3data/vibe-harness-mcp), search results are shaped by the current working mode:

| Mode | Behaviour |
|------|-----------|
| **explore** | Cross-project boost, research-heavy, wide diversity slots |
| **build** | Code-focused, same-project, narrow results |
| **think-with** | Research + reference, wide diversity, unexpected adjacencies |
| **ship** | Code + docs, narrow, high-confidence results |
| **cool-off** | Suppressed surfacing, minimal interruption |

Mode signals use an **additive bias model**: contextual preferences (source type, cross-project, resurfacing history, topic recurrence) are summed and applied as a bounded adjustment to the base relevance score. This ensures mode can reorder near-ties but never bury a highly relevant result behind an irrelevant one — the failure mode that multiplicative approaches produce over long sessions.

The resurfacing penalty is **trajectory-aware**: when the conversation is diverging, recurring content is treated as coherent anchoring (penalty halved); when converging, it's treated as circling (penalty applied normally). Anti-entrainment widens diversity for narrow modes during convergence, and shifts already-wide modes toward more serendipity.

Mode profiles are fully configurable in `sense.toml` under `[mode.profiles.*]`.

### Environment variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Required. Embedding API key. |
| `SENSE_CONFIG` | Optional. Absolute path to config file. |
| `SENSE_ROOT` | Optional. Corpus root (overrides config). |

## Design direction

Sense is also a research artifact. It investigates whether relevance realisation — the pre-reflective process by which organisms determine what matters — can be partially externalised into infrastructure.

The current implementation composes five signals: semantic similarity, temporal decay, mode awareness, conversation trajectory, and relevance feedback. The architecture has extension points for additional signals as they become available: decision anchoring (epistemic posture), graph adjacency (structural connections via [zetl](https://codeberg.org/anuna/zetl)), and biosignal responsiveness (physiological state, via [vibe-harness](https://github.com/m3data/vibe-harness-mcp), influencing what surfaces).

The feedback loop closes the cybernetic circuit: Sense observes → auto-labels → the dashboard renders the observation → the human corrects → weights shift → Sense changes what it surfaces. The observation infrastructure is itself observable — a Baradian cut made visible.

The system scaffolds the human's relevance realisation — it does not replace it. But through its responsiveness to working context, it participates in the coupling dynamic that produces relevance.

See `DESIGN_DIRECTION_relevance-realisation.md` and `ARCHITECTURE-DECISIONS.md` for the full design rationale.

## License

Apache 2.0
