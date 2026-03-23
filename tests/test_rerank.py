"""Tests for cross-encoder reranking."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.rerank import rerank, rerank_batch


@pytest.fixture
def mock_reranker():
    """Mock CrossEncoder that returns deterministic logits."""
    reranker = MagicMock()
    return reranker


class TestRerank:
    def test_sigmoid_applied_and_sorted(self, mock_reranker):
        """rerank() should apply sigmoid to logits and sort descending."""
        # Logits: -2.0, 0.0, 3.0 -> sigmoid: ~0.119, 0.5, ~0.953
        mock_reranker.predict.return_value = np.array([-2.0, 0.0, 3.0])

        candidates = [
            {"target_id": "a", "score": 0.1, "text": "alpha"},
            {"target_id": "b", "score": 0.2, "text": "beta"},
            {"target_id": "c", "score": 0.3, "text": "gamma"},
        ]

        result = rerank(mock_reranker, "query", candidates, top_n=3)

        assert len(result) == 3
        assert result[0]["target_id"] == "c"
        assert result[1]["target_id"] == "b"
        assert result[2]["target_id"] == "a"

        # Verify sigmoid values
        assert abs(result[0]["score"] - 1 / (1 + np.exp(-3.0))) < 1e-6
        assert abs(result[1]["score"] - 0.5) < 1e-6
        assert abs(result[2]["score"] - 1 / (1 + np.exp(2.0))) < 1e-6

        # Verify all scores in [0, 1]
        for r in result:
            assert 0.0 <= r["score"] <= 1.0

    def test_top_n_truncates(self, mock_reranker):
        """rerank() should return only top_n results."""
        mock_reranker.predict.return_value = np.array([1.0, 2.0, 3.0, 4.0])

        candidates = [
            {"target_id": f"t{i}", "score": 0.1, "text": f"text-{i}"}
            for i in range(4)
        ]

        result = rerank(mock_reranker, "query", candidates, top_n=2)

        assert len(result) == 2
        assert result[0]["target_id"] == "t3"
        assert result[1]["target_id"] == "t2"

    def test_empty_candidates_raises(self, mock_reranker):
        """rerank() should raise ValueError for empty candidates."""
        with pytest.raises(ValueError, match="Empty candidates list"):
            rerank(mock_reranker, "query", [], top_n=10)

    def test_pairs_formed_correctly(self, mock_reranker):
        """rerank() should pass (query, text) pairs to predict()."""
        mock_reranker.predict.return_value = np.array([0.0, 0.0])

        candidates = [
            {"target_id": "a", "score": 0.1, "text": "first text"},
            {"target_id": "b", "score": 0.2, "text": "second text"},
        ]

        rerank(mock_reranker, "my query", candidates, top_n=2)

        pairs = mock_reranker.predict.call_args[0][0]
        assert pairs == [("my query", "first text"), ("my query", "second text")]

    def test_result_has_no_text_field(self, mock_reranker):
        """rerank() output should contain only target_id and score."""
        mock_reranker.predict.return_value = np.array([1.0])

        candidates = [{"target_id": "a", "score": 0.5, "text": "some text"}]
        result = rerank(mock_reranker, "query", candidates, top_n=1)

        assert set(result[0].keys()) == {"target_id", "score"}


class TestRerankBatch:
    def test_multiple_queries(self, mock_reranker):
        """rerank_batch() should process each query independently."""
        mock_reranker.predict.side_effect = [
            np.array([1.0, 2.0]),
            np.array([3.0, 0.0]),
        ]

        queries = ["q1", "q2"]
        candidates_per_query = [
            [
                {"target_id": "a", "score": 0.1, "text": "a"},
                {"target_id": "b", "score": 0.2, "text": "b"},
            ],
            [
                {"target_id": "c", "score": 0.3, "text": "c"},
                {"target_id": "d", "score": 0.4, "text": "d"},
            ],
        ]

        results = rerank_batch(
            mock_reranker, queries, candidates_per_query, top_n=2,
        )

        assert len(results) == 2
        # First query: logits [1.0, 2.0] -> b first
        assert results[0][0]["target_id"] == "b"
        # Second query: logits [3.0, 0.0] -> c first
        assert results[1][0]["target_id"] == "c"

    def test_empty_candidates_for_one_query(self, mock_reranker):
        """rerank_batch() should return empty list for empty candidates."""
        mock_reranker.predict.return_value = np.array([1.0])

        queries = ["q1", "q2"]
        candidates_per_query = [
            [],
            [{"target_id": "a", "score": 0.1, "text": "text"}],
        ]

        results = rerank_batch(
            mock_reranker, queries, candidates_per_query, top_n=10,
        )

        assert len(results) == 2
        assert results[0] == []
        assert len(results[1]) == 1
