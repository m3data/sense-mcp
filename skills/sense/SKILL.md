---
name: sense
description: Search the indexed ecosystem by semantic similarity. Finds relevant content across traces, docs, and research.
---

# Sense Search

Search the indexed ecosystem using the `sense_search` MCP tool.

## With arguments

Parse `$ARGUMENTS` as follows:
- The main text is the search query
- `--project <name>` filters to a specific project
- `--type <source_type>` filters by type (trace, documentation, project_claude, reference, research, teaching, code)
- `--limit <n>` sets max results (default 10)
- `--mode <mode>` overrides Vibe Harness mode for search shaping (explore, build, think-with, ship, cool-off, none). Omit to auto-detect. Use `none` to force flat cosine search regardless of Vibe Harness state.

Examples:
- `/sense how does authentication work`
- `/sense API error handling --project backend`
- `/sense deployment configuration --type documentation --limit 5`
- `/sense architecture decisions --mode explore`

## Without arguments

When `$ARGUMENTS` is empty, synthesize a search query from the conversation context:

1. Review the last 3-5 exchanges in the current conversation
2. Identify the core topic, concepts, or question being discussed
3. Compose a 3-8 word query that captures the semantic thread
4. Prefix results with "Searching for: [synthesized query]" so the user sees what was searched
5. Call `sense_search` with the synthesized query

This enables mid-conversation discovery — just type `/sense` to surface related prior work without crafting a query.

## General

Call `sense_search` with the parsed arguments and display the results. If the index is empty, suggest running `/sense-sync` first.
