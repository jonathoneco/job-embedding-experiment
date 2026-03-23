"""Tests for BGE-M3 multi-modal embedding module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from src.bgem3 import (
    encode_bgem3,
    rank_bgem3_colbert,
    rank_bgem3_dense,
    rank_bgem3_sparse,
    run_bgem3,
)


class MockBGEM3:
    """Deterministic mock for BGEM3FlagModel."""

    def encode(
        self,
        texts,
        batch_size=64,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    ):
        n = len(texts)
        dim = 1024
        rng = np.random.RandomState(42)
        dense = rng.randn(n, dim).astype(np.float32)
        sparse = [{str(j): rng.random() for j in range(5)} for _ in range(n)]
        colbert = [rng.randn(3, dim).astype(np.float32) for _ in range(n)]
        return {
            "dense": dense,
            "lexical_weights": sparse,
            "colbert_vecs": colbert,
        }


@pytest.fixture
def mock_bgem3():
    return MockBGEM3()


def _make_targets(n=5, granularity="role"):
    return [
        {"id": f"{granularity}-{i}", "text": f"target text {i}", "granularity": granularity}
        for i in range(n)
    ]


class TestEncodeBgem3:
    def test_returns_dict_with_three_keys(self, mock_bgem3):
        result = encode_bgem3(mock_bgem3, ["hello", "world"])
        assert set(result.keys()) == {"dense", "sparse", "colbert"}

    def test_dense_shape(self, mock_bgem3):
        result = encode_bgem3(mock_bgem3, ["a", "b", "c"])
        assert result["dense"].shape == (3, 1024)

    def test_dense_is_l2_normalized(self, mock_bgem3):
        result = encode_bgem3(mock_bgem3, ["a", "b", "c"])
        norms = np.linalg.norm(result["dense"], axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_sparse_is_list_of_dicts(self, mock_bgem3):
        result = encode_bgem3(mock_bgem3, ["a", "b"])
        assert isinstance(result["sparse"], list)
        assert len(result["sparse"]) == 2
        assert isinstance(result["sparse"][0], dict)

    def test_colbert_is_list_of_arrays(self, mock_bgem3):
        result = encode_bgem3(mock_bgem3, ["a", "b"])
        assert isinstance(result["colbert"], list)
        assert len(result["colbert"]) == 2
        assert isinstance(result["colbert"][0], np.ndarray)

    def test_zero_norm_raises(self):
        """Zero-norm dense embedding should raise ValueError."""
        model = MagicMock()
        model.encode.return_value = {
            "dense": np.zeros((2, 1024)),
            "lexical_weights": [{}, {}],
            "colbert_vecs": [np.zeros((3, 1024)), np.zeros((3, 1024))],
        }
        with pytest.raises(ValueError, match="Zero-norm"):
            encode_bgem3(model, ["a", "b"])

    def test_batch_size_forwarded(self):
        model = MagicMock()
        rng = np.random.RandomState(0)
        model.encode.return_value = {
            "dense": rng.randn(1, 1024).astype(np.float32),
            "lexical_weights": [{"0": 0.5}],
            "colbert_vecs": [rng.randn(3, 1024).astype(np.float32)],
        }
        encode_bgem3(model, ["text"], batch_size=32)
        _, kwargs = model.encode.call_args
        assert kwargs["batch_size"] == 32


class TestRankBgem3Dense:
    def test_returns_correct_format(self, mock_bgem3):
        targets = _make_targets(5)
        query_repr = encode_bgem3(mock_bgem3, ["query 1"])
        target_repr = encode_bgem3(mock_bgem3, [t["text"] for t in targets])
        results = rank_bgem3_dense(query_repr, target_repr, targets, top_k=3)

        assert len(results) == 1
        assert len(results[0]) == 3
        assert "target_id" in results[0][0]
        assert "score" in results[0][0]

    def test_sorted_descending(self, mock_bgem3):
        targets = _make_targets(5)
        query_repr = encode_bgem3(mock_bgem3, ["query 1"])
        target_repr = encode_bgem3(mock_bgem3, [t["text"] for t in targets])
        results = rank_bgem3_dense(query_repr, target_repr, targets, top_k=5)

        scores = [r["score"] for r in results[0]]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self, mock_bgem3):
        targets = _make_targets(10)
        query_repr = encode_bgem3(mock_bgem3, ["query"])
        target_repr = encode_bgem3(mock_bgem3, [t["text"] for t in targets])
        results = rank_bgem3_dense(query_repr, target_repr, targets, top_k=3)

        assert len(results[0]) == 3

    def test_multiple_queries(self, mock_bgem3):
        targets = _make_targets(5)
        query_repr = encode_bgem3(mock_bgem3, ["query 1", "query 2"])
        target_repr = encode_bgem3(mock_bgem3, [t["text"] for t in targets])
        results = rank_bgem3_dense(query_repr, target_repr, targets, top_k=3)

        assert len(results) == 2


class TestRankBgem3Sparse:
    def test_returns_correct_format(self, mock_bgem3):
        targets = _make_targets(5)
        query_repr = encode_bgem3(mock_bgem3, ["query 1"])
        target_repr = encode_bgem3(mock_bgem3, [t["text"] for t in targets])
        results = rank_bgem3_sparse(query_repr, target_repr, targets, top_k=3)

        assert len(results) == 1
        assert len(results[0]) == 3
        assert "target_id" in results[0][0]
        assert "score" in results[0][0]

    def test_sorted_descending(self, mock_bgem3):
        targets = _make_targets(5)
        query_repr = encode_bgem3(mock_bgem3, ["query 1"])
        target_repr = encode_bgem3(mock_bgem3, [t["text"] for t in targets])
        results = rank_bgem3_sparse(query_repr, target_repr, targets, top_k=5)

        scores = [r["score"] for r in results[0]]
        assert scores == sorted(scores, reverse=True)

    def test_no_overlap_gives_zero_score(self):
        """Queries and targets with disjoint tokens should have zero scores."""
        query_repr = {"sparse": [{"a": 1.0, "b": 2.0}]}
        target_repr = {"sparse": [{"c": 1.0, "d": 2.0}]}
        targets = [{"id": "t-0"}]
        results = rank_bgem3_sparse(query_repr, target_repr, targets, top_k=1)

        assert results[0][0]["score"] == 0.0

    def test_overlap_produces_positive_score(self):
        """Overlapping tokens with positive weights should produce positive scores."""
        query_repr = {"sparse": [{"tok1": 2.0, "tok2": 3.0}]}
        target_repr = {"sparse": [{"tok1": 1.5, "tok3": 1.0}]}
        targets = [{"id": "t-0"}]
        results = rank_bgem3_sparse(query_repr, target_repr, targets, top_k=1)

        # 2.0 * 1.5 = 3.0 (only tok1 overlaps)
        assert abs(results[0][0]["score"] - 3.0) < 1e-6


class TestRankBgem3Colbert:
    def test_returns_correct_format(self, mock_bgem3):
        targets = _make_targets(5)
        query_repr = encode_bgem3(mock_bgem3, ["query 1"])
        target_repr = encode_bgem3(mock_bgem3, [t["text"] for t in targets])
        results = rank_bgem3_colbert(query_repr, target_repr, targets, top_k=3)

        assert len(results) == 1
        assert len(results[0]) == 3
        assert "target_id" in results[0][0]
        assert "score" in results[0][0]

    def test_sorted_descending(self, mock_bgem3):
        targets = _make_targets(5)
        query_repr = encode_bgem3(mock_bgem3, ["query 1"])
        target_repr = encode_bgem3(mock_bgem3, [t["text"] for t in targets])
        results = rank_bgem3_colbert(query_repr, target_repr, targets, top_k=5)

        scores = [r["score"] for r in results[0]]
        assert scores == sorted(scores, reverse=True)

    def test_maxsim_computation(self):
        """Verify MaxSim: max per query token, then average."""
        # 2 query tokens, 2 target tokens, dim=3
        q_vecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        t_vecs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        query_repr = {"colbert": [q_vecs]}
        target_repr = {"colbert": [t_vecs]}
        targets = [{"id": "t-0"}]

        results = rank_bgem3_colbert(query_repr, target_repr, targets, top_k=1)

        # q_token_0 vs t: max(1.0, 0.0) = 1.0
        # q_token_1 vs t: max(0.0, 0.0) = 0.0
        # average = 0.5
        assert abs(results[0][0]["score"] - 0.5) < 1e-6


class TestRunBgem3:
    @patch("src.bgem3.load_bgem3")
    def test_returns_s01_format(self, mock_load):
        mock_load.return_value = MockBGEM3()
        config = {
            "bgem3": {
                "enabled": True,
                "model": "BAAI/bge-m3",
                "granularities": ["role"],
            },
        }
        target_sets = {"role": _make_targets(5, "role")}
        test_cases = [{"id": "tc-1", "input_title": "Engineer"}]

        results = run_bgem3(config, target_sets, test_cases)

        assert len(results) > 0
        for r in results:
            assert "test_case_id" in r
            assert "method" in r
            assert "granularity" in r
            assert "ranked_results" in r

    @patch("src.bgem3.load_bgem3")
    def test_produces_all_three_methods(self, mock_load):
        mock_load.return_value = MockBGEM3()
        config = {
            "bgem3": {
                "enabled": True,
                "model": "BAAI/bge-m3",
                "granularities": ["role"],
            },
        }
        target_sets = {"role": _make_targets(5, "role")}
        test_cases = [{"id": "tc-1", "input_title": "Engineer"}]

        results = run_bgem3(config, target_sets, test_cases)

        methods = {r["method"] for r in results}
        assert methods == {"bgem3-dense", "bgem3-sparse", "bgem3-colbert"}

    @patch("src.bgem3.load_bgem3")
    def test_multiple_granularities(self, mock_load):
        mock_load.return_value = MockBGEM3()
        config = {
            "bgem3": {
                "enabled": True,
                "model": "BAAI/bge-m3",
                "granularities": ["role", "category"],
            },
        }
        target_sets = {
            "role": _make_targets(5, "role"),
            "category": _make_targets(3, "category"),
        }
        test_cases = [{"id": "tc-1", "input_title": "Engineer"}]

        results = run_bgem3(config, target_sets, test_cases)

        granularities = {r["granularity"] for r in results}
        assert granularities == {"role", "category"}
        # 3 methods * 2 granularities * 1 test case = 6 results
        assert len(results) == 6

    @patch("src.bgem3.load_bgem3")
    def test_skips_missing_granularity(self, mock_load):
        mock_load.return_value = MockBGEM3()
        config = {
            "bgem3": {
                "enabled": True,
                "model": "BAAI/bge-m3",
                "granularities": ["role", "category"],
            },
        }
        # Only provide role targets
        target_sets = {"role": _make_targets(5, "role")}
        test_cases = [{"id": "tc-1", "input_title": "Engineer"}]

        results = run_bgem3(config, target_sets, test_cases)

        granularities = {r["granularity"] for r in results}
        assert granularities == {"role"}

    @patch("src.bgem3.load_bgem3")
    def test_multiple_test_cases(self, mock_load):
        mock_load.return_value = MockBGEM3()
        config = {
            "bgem3": {
                "enabled": True,
                "model": "BAAI/bge-m3",
                "granularities": ["role"],
            },
        }
        target_sets = {"role": _make_targets(5, "role")}
        test_cases = [
            {"id": "tc-1", "input_title": "Engineer"},
            {"id": "tc-2", "input_title": "Manager"},
        ]

        results = run_bgem3(config, target_sets, test_cases)

        # 3 methods * 1 granularity * 2 test cases = 6 results
        assert len(results) == 6
        tc_ids = {r["test_case_id"] for r in results}
        assert tc_ids == {"tc-1", "tc-2"}


class TestConfigBgem3Section:
    def test_config_has_bgem3_section(self):
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        assert "bgem3" in config

    def test_bgem3_disabled_by_default(self):
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        assert config["bgem3"]["enabled"] is False

    def test_bgem3_model_is_bge_m3(self):
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        assert config["bgem3"]["model"] == "BAAI/bge-m3"

    def test_bgem3_granularities(self):
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        assert config["bgem3"]["granularities"] == ["role", "category"]

    def test_fusion_has_bgem3_config(self):
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        fusion_names = [c["name"] for c in config["fusion"]["configs"]]
        assert "fusion-bgem3-all" in fusion_names

    def test_fusion_bgem3_methods(self):
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        bgem3_fusion = [
            c for c in config["fusion"]["configs"] if c["name"] == "fusion-bgem3-all"
        ][0]
        assert bgem3_fusion["methods"] == ["bgem3-dense", "bgem3-sparse", "bgem3-colbert"]
        assert bgem3_fusion["granularity"] == "role"
