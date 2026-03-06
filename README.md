# Sense

![Repo Status](https://img.shields.io/badge/REPO_STATUS-Active_Research-blue?style=for-the-badge&labelColor=8b5e3c&color=e5dac1)
![Version](https://img.shields.io/badge/VERSION-0.2.0-blue?style=for-the-badge&labelColor=3b82f6&color=1e40af)
![License](https://img.shields.io/badge/LICENSE-Apache_2.0-green?style=for-the-badge&labelColor=10b981&color=047857)
![Python](https://img.shields.io/badge/PYTHON-3.11+-green?style=for-the-badge&labelColor=10b981&color=047857)
![MCP](https://img.shields.io/badge/MCP-stdio-purple?style=for-the-badge&labelColor=7c3aed&color=5b21b6)

Relevant context from your project ecosystem, injected into every AI conversation — automatically, weighted by recency, and shaped by what you're doing.

## The problem

You work across a project ecosystem. There's documentation, code, research, decisions, regulatory files, customer feedback, session traces — it's all there. But when you're in a conversation with your AI agent, none of it shows up unless you go find it and paste it in. The AI agent is capable but contextually blind to your ecosystem. Particularly if you are running multiple projects, the relevant context is a moving target. You can't predict what you'll need in advance, and you usually won't know what you needed until after the fact.

Memory and RAG tools try to solve this by storing and retrieving everything. But recall is not relevance. Dumping context into a prompt without discrimination buries the useful connections under volume. And search requires you to know what you're looking for — which means it can't surface the connections you didn't know to make.

## What Sense does

Sense indexes your project ecosystem and injects relevant context into every conversation automatically. You don't search. You don't paste. Relevant prior work surfaces based on what you're talking about right now.

It runs as an [MCP](https://modelcontextprotocol.io/) server for [Claude Code](https://docs.anthropic.com/en/docs/claude-code), with a companion hook that fires on every prompt. The result: your AI partner always has peripheral awareness of the ecosystem it's working in.

### How it's different

**Ambient, not invoked.** The auto-query hook fires on every prompt. Context arrives without being asked for. This is the primary interaction pattern — not a search box you type into, but a layer that's always running in the background, shaping what's visible.

**Knowledge metabolises.** A session trace from yesterday and a reference document from last year are not equally alive. Sense weights them differently — recent work surfaces more readily, old documentation fades, foundational reference stays evergreen. Different types of content have different half-lives because they are different kinds of knowledge.

When paired with [Vibe Harness](https://github.com/m3data/vibe-harness-mcp), what surfaces also changes based on what you're doing. Exploring widens the aperture — cross-project connections, unexpected adjacencies. Building narrows it to code and documentation in the current project. The same corpus looks different depending on your working mode.

**Source-classified, diversity-structured.** Files are classified into types (traces, code, research, documentation, reference, research etc), each with distinct decay rates that can be set and mode weightings. Results are structured into confirmation slots (highest relevance), divergence slots (what challenges the current frame), and serendipity slots (from projects you weren't looking at). The candidate pool uses stratified sampling to guarantee representation across source types, so minority types can surface when mode multipliers promote them. The goal is productive connections, not just the nearest match.

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

## Tools

### `sense_search`

Search by natural language. Returns ranked results with similarity scores, temporal decay, and content previews. Supports optional filters: `project`, `source_type`, `limit`, `mode`.

### `sense_sync`

Build or update the index. Uses SHA-256 file hashing for change detection — unchanged files are skipped. Safe to run repeatedly.

### `sense_status`

Index statistics: chunk counts by project and source type, total tokens, last sync time.

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

### Mode-aware retrieval

When paired with [Vibe Harness](https://github.com/m3data/vibe-harness-mcp), search results are shaped by the current working mode:

| Mode | Behaviour |
|------|-----------|
| **explore** | Cross-project boost, research-heavy, wide diversity slots |
| **build** | Code-focused, same-project, narrow results |
| **think-with** | Research + reference, wide diversity, unexpected adjacencies |
| **ship** | Code + docs, narrow, high-confidence results |
| **cool-off** | Suppressed surfacing, minimal interruption |

Mode profiles are fully configurable in `sense.toml` under `[mode.profiles.*]`.

### Environment variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Required. Embedding API key. |
| `SENSE_CONFIG` | Optional. Absolute path to config file. |
| `SENSE_ROOT` | Optional. Corpus root (overrides config). |

## Design direction

Sense is also a research artifact. It investigates whether relevance realisation — the pre-reflective process by which organisms determine what matters — can be partially externalised into infrastructure.

The current implementation composes three signals: semantic similarity, temporal decay, and mode awareness. The architecture is designed to accommodate additional signals as they become available: decision anchoring (epistemic posture), graph adjacency (structural connections via [zetl](https://codeberg.org/anuna/zetl)), and biosignal responsiveness (physiological state influencing what surfaces) through [vibe-harness](https://github.com/m3data/vibe-harness-mcp).

The system scaffolds the human's relevance realisation — it does not replace it. But through its responsiveness to working context, it participates in the coupling dynamic that produces relevance.

See `DESIGN_DIRECTION_relevance-realisation.md` and `ARCHITECTURE-DECISIONS.md` for the full design rationale.

## License

Apache 2.0
