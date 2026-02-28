# Sense

Semantic retrieval MCP server for markdown-heavy project ecosystems. Indexes files, chunks them, embeds with OpenAI, and serves search over MCP (Model Context Protocol).

## What it does

- Walks a project root, discovers markdown/text/code files
- Chunks by `##` sections (markdown) or paragraphs (fallback)
- Embeds via OpenAI `text-embedding-3-small`
- Stores in SQLite with file-hash change detection
- Serves three MCP tools: `sense_search`, `sense_sync`, `sense_status`
- Temporal decay: recent content scores higher (configurable half-lives per source type)
- Mode-aware retrieval: integrates with [Vibe Harness](https://github.com/anuna-research/vibe-harness) for context-sensitive search (optional)

## Install

```bash
cd sense-mcp
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Requires Python 3.11+ (uses `tomllib` from stdlib).

## Configure

Copy the example config and adjust:

```bash
cp sense.example.toml sense.toml
```

At minimum, set:

- `[corpus] root` — directory to index (default: parent of sense-mcp/)
- `[corpus] env_file` — path to `.env` with your `OPENAI_API_KEY`
- `[corpus] excluded_dirs` — directories to skip
- `[[classification.rules]]` — rules mapping paths to source types

See `sense.example.toml` for all options with documentation.

### Environment variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Required. Embedding API key. |
| `SENSE_CONFIG` | Optional. Absolute path to config file (overrides default location). |
| `SENSE_ROOT` | Optional. Corpus root (overrides config `[corpus] root`). |

## Wire up to Claude Code

Add to your Claude Code MCP settings (`.claude.json` or via `claude mcp add`):

```json
{
  "mcpServers": {
    "sense": {
      "command": "/path/to/sense-mcp/.venv/bin/python",
      "args": ["/path/to/sense-mcp/server.py"],
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

## Auto-query hook (optional)

The companion hook at `.claude/hooks/sense-auto-query.py` fires on every user prompt and injects relevant context automatically. It:

- Gates on prompt length, cooldown, and continuation signals
- Opens the SQLite DB in read-only mode (coexists with running MCP server)
- Injects top-3 results as `<sense-context>` tags

To wire it up, add to `.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "type": "command",
        "command": "/path/to/sense-mcp/.venv/bin/python /path/to/.claude/hooks/sense-auto-query.py"
      }
    ]
  }
}
```

## Tools

### `sense_search`

Search by natural language. Returns ranked results with similarity scores, temporal decay, and content previews.

### `sense_sync`

Build or update the index. Skips unchanged files (SHA-256 hash). Safe to run repeatedly.

### `sense_status`

Show index statistics: chunk counts by project and source type, total tokens, last sync time.

## Classification

Files are classified into source types using ordered rules in `sense.toml`. Each rule has a type and a matcher:

| Matcher | Description |
|---------|-------------|
| `filename` | Regex matched against the filename |
| `path_contains` | Substring match in the relative path |
| `path_segment` | Directory name(s) matched as path segments |
| `extension` | File extension(s) |

First matching rule wins. Unmatched files get the `[classification] default` type (default: `documentation`).

Source types control temporal decay (via `[decay.half_lives]`) and mode-aware scoring (via `[mode.profiles]`).

## Mode-aware retrieval

When paired with Vibe Harness, search results are shaped by the current working mode:

| Mode | Bias |
|------|------|
| explore | Cross-project, research-heavy, wide diversity |
| build | Code-focused, narrow, same-project |
| think-with | Research + reference, wide diversity |
| ship | Code + docs, narrow, same-project |
| cool-off | Suppressed results, minimal surfacing |

Mode profiles are fully configurable in `sense.toml`.

## License

Apache 2.0 — Anuna Research Cooperative
