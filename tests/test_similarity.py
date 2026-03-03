"""Tests for similarity utilities — REQ-003.

Covers:
    - cosine_similarity: identical, orthogonal, and zero-vector cases
    - count_tokens: known-string token counts using cl100k_base
    - _truncate_for_embedding: beyond-limit text shortened, within-limit text passes through
"""

import numpy as np
import pytest

from sense_mcp.server import (
    _truncate_for_embedding,
    cosine_similarity,
    count_tokens,
)


class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector_returns_zero_without_error(self):
        zero = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        other = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(zero, other) == 0.0


class TestCountTokens:
    def test_single_word_token_count(self):
        # "hello" is a single token in cl100k_base
        assert count_tokens("hello") == 1

    def test_two_word_token_count(self):
        # "hello world" encodes as ["hello", " world"] = 2 tokens in cl100k_base
        assert count_tokens("hello world") == 2


class TestTruncateForEmbedding:
    def test_text_beyond_limit_is_shortened(self, test_env):
        # "hello " * 10000 ≈ 10 001 tokens — well over the 8 000 token limit
        long_text = "hello " * 10000
        result = _truncate_for_embedding(long_text)
        assert len(result) < len(long_text)
        assert count_tokens(result) <= test_env.max_input_tokens

    def test_text_within_limit_passes_through(self):
        short_text = "hello world"
        assert _truncate_for_embedding(short_text) == short_text
