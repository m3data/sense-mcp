"""Tests for chunk_file() and _split_paragraphs() — REQ-002.

Eight scenarios:
1. Small files return single chunk
2. Markdown splits on ## headers
3. Section headers preserved in content
4. Large sections sub-split by paragraph
5. Paragraph merging respects max_tokens
6. Empty paragraphs excluded
7. Non-markdown falls through to paragraph splitting
8. Section metadata propagates through sub-splits
"""

from pathlib import Path

from sense_mcp.server import _split_paragraphs, chunk_file, count_tokens

# Verified with cl100k_base:
#   "Section body content with enough words to be meaningful. " * 20 → ~201 tokens
#   "This paragraph contains enough words to contribute meaningfully to the token count. " * 10 → ~141 tokens
#   "Short paragraph here." → 4 tokens


# ── Scenario 1 ───────────────────────────────────────────────────────────────

def test_small_file_returns_single_chunk():
    """Files ≤512 tokens are returned as one unsplit chunk with section=None."""
    content = "This is a short document.\n\nIt has two paragraphs."
    assert count_tokens(content) <= 512

    chunks = chunk_file(Path("notes.md"), content)

    assert len(chunks) == 1
    assert chunks[0]["section"] is None
    assert chunks[0]["content"] == content
    assert chunks[0]["token_count"] == count_tokens(content)


# ── Scenario 2 ───────────────────────────────────────────────────────────────

def test_markdown_splits_on_h2_headers():
    """Markdown files with ## sections produce one chunk per section."""
    # 30 reps ≈ 300 tokens per section body; two sections ≈ 620 tokens > 512
    sec_body = "Section body content with enough words to be meaningful. " * 30
    content = (
        "# Project Notes\n\nIntroductory paragraph.\n\n"
        f"## Alpha\n\n{sec_body}\n\n"
        f"## Beta\n\n{sec_body}"
    )
    assert count_tokens(content) > 512

    chunks = chunk_file(Path("doc.md"), content)
    sections = {c["section"] for c in chunks}

    assert "Alpha" in sections
    assert "Beta" in sections


# ── Scenario 3 ───────────────────────────────────────────────────────────────

def test_section_header_preserved_in_content():
    """Each section chunk's content begins with its ## header line."""
    # 70 reps ≈ 700 tokens for section; total > 512 but section < 1024 (no sub-split)
    sec_body = "Content for this section with several sentences. " * 70
    content = (
        "# Preamble\n\nSome introduction text here.\n\n"
        f"## My Section\n\n{sec_body}"
    )
    assert count_tokens(content) > 512
    section_full = f"## My Section\n\n{sec_body}"
    assert count_tokens(section_full) < 1024, "Section must be under 1024 to avoid sub-splitting"

    chunks = chunk_file(Path("doc.md"), content)
    section_chunks = [c for c in chunks if c["section"] == "My Section"]

    assert section_chunks, "Expected at least one chunk for 'My Section'"
    assert section_chunks[0]["content"].startswith("## My Section")


# ── Scenario 4 ───────────────────────────────────────────────────────────────

def test_large_section_sub_split_by_paragraph():
    """Sections exceeding 1024 tokens produce multiple sub-chunks."""
    # 10 reps ≈ 141 tokens per paragraph; 8 paragraphs ≈ 1133 tokens > 1024
    para = "This paragraph contains enough words to contribute meaningfully to the token count. " * 10
    big_body = "\n\n".join([para] * 8)
    content = f"# Doc\n\nShort intro.\n\n## Big Section\n\n{big_body}"

    section_tokens = count_tokens(f"## Big Section\n\n{big_body}")
    assert section_tokens > 1024, f"Section must exceed 1024 tokens, got {section_tokens}"

    chunks = chunk_file(Path("doc.md"), content)
    big_chunks = [c for c in chunks if c["section"] == "Big Section"]

    assert len(big_chunks) > 1, "Large section should produce multiple sub-chunks"


# ── Scenario 5 ───────────────────────────────────────────────────────────────

def test_paragraph_merging_respects_max_tokens():
    """Paragraphs are merged up to max_tokens; overflow starts a new chunk.

    With max_tokens=15 and 4-token paragraphs: 3 paragraphs merge to 12 tokens,
    then the 4th (12 + 4 = 16 > 15) triggers a new chunk.
    """
    small_para = "Short paragraph here."  # 4 tokens
    assert count_tokens(small_para) == 4

    # 3 paragraphs merge fine (12 tokens); 4th would exceed max_tokens=15
    paras = [small_para] * 4
    text = "\n\n".join(paras)

    chunks = _split_paragraphs(text, max_tokens=15)

    assert len(chunks) >= 2, "Overflow paragraph should start a new chunk"
    for c in chunks:
        assert c["token_count"] <= 15, f"Chunk exceeds max_tokens: {c['token_count']}"


# ── Scenario 6 ───────────────────────────────────────────────────────────────

def test_empty_paragraphs_excluded():
    """Multiple blank lines do not produce empty chunks."""
    content = "First paragraph.\n\n\n\n\n\nSecond paragraph.\n\n\n\nThird paragraph."

    chunks = _split_paragraphs(content, max_tokens=512)

    assert all(c["content"].strip() for c in chunks), "No chunk should have empty content"
    combined = " ".join(c["content"] for c in chunks)
    assert "First paragraph" in combined
    assert "Second paragraph" in combined
    assert "Third paragraph" in combined


# ── Scenario 7 ───────────────────────────────────────────────────────────────

def test_non_markdown_falls_through_to_paragraph_splitting():
    """Non-.md files use paragraph splitting; all chunks have section=None."""
    line = "result = compute_value(x, y)  # evaluate expression\n"
    code_content = line * 60
    assert count_tokens(code_content) > 512

    chunks = chunk_file(Path("module.py"), code_content)

    assert len(chunks) >= 1
    assert all(c["section"] is None for c in chunks)


# ── Scenario 8 ───────────────────────────────────────────────────────────────

def test_section_metadata_propagates_through_sub_splits():
    """All sub-chunks from a large section carry the original section name."""
    # 10 reps ≈ 141 tokens per paragraph; 8 paragraphs ≈ 1133 tokens > 1024
    para = "This paragraph contains enough words to contribute meaningfully to the token count. " * 10
    big_body = "\n\n".join([para] * 8)
    content = f"# Doc\n\nIntro.\n\n## Analysis\n\n{big_body}"

    section_tokens = count_tokens(f"## Analysis\n\n{big_body}")
    assert section_tokens > 1024, f"Need >1024 tokens, got {section_tokens}"

    chunks = chunk_file(Path("doc.md"), content)
    sub_chunks = [c for c in chunks if c["section"] == "Analysis"]

    assert len(sub_chunks) > 1, "Large section should produce multiple sub-chunks"
    for c in sub_chunks:
        assert c["section"] == "Analysis"
