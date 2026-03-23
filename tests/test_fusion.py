"""Tests for RRF score fusion."""

import pytest

from src.fusion import fuse_all, fuse_rankings


class TestFuseRankings:
    def test_overlapping_candidates(self):
        """RRF should accumulate scores for candidates appearing in both methods."""
        rankings_by_method = {
            "method_a": [[
                {"target_id": "x", "score": 0.9},
                {"target_id": "y", "score": 0.8},
            ]],
            "method_b": [[
                {"target_id": "y", "score": 0.9},
                {"target_id": "x", "score": 0.8},
            ]],
        }
        k = 60

        result = fuse_rankings(rankings_by_method, k=k, top_n=10)

        assert len(result) == 1
        query_result = result[0]
        assert len(query_result) == 2

        # Both x and y appear at rank 1 in one list and rank 2 in other
        # So they should have equal RRF scores: 1/(60+1) + 1/(60+2)
        expected_score = 1.0 / (k + 1) + 1.0 / (k + 2)
        assert abs(query_result[0]["score"] - expected_score) < 1e-10
        assert abs(query_result[1]["score"] - expected_score) < 1e-10

    def test_disjoint_candidates(self):
        """Disjoint candidates should each get RRF score from their method only."""
        rankings_by_method = {
            "method_a": [[
                {"target_id": "a1", "score": 0.9},
                {"target_id": "a2", "score": 0.8},
            ]],
            "method_b": [[
                {"target_id": "b1", "score": 0.9},
                {"target_id": "b2", "score": 0.8},
            ]],
        }
        k = 60

        result = fuse_rankings(rankings_by_method, k=k, top_n=10)

        query_result = result[0]
        assert len(query_result) == 4

        # All rank-1 items should tie, all rank-2 items should tie
        scores = {r["target_id"]: r["score"] for r in query_result}
        assert abs(scores["a1"] - scores["b1"]) < 1e-10
        assert abs(scores["a2"] - scores["b2"]) < 1e-10
        # Rank 1 score > rank 2 score
        assert scores["a1"] > scores["a2"]

    def test_top_n_truncates(self):
        """fuse_rankings() should return at most top_n results."""
        rankings_by_method = {
            "m": [[
                {"target_id": f"t{i}", "score": 0.1}
                for i in range(20)
            ]],
        }

        result = fuse_rankings(rankings_by_method, k=60, top_n=5)

        assert len(result[0]) == 5

    def test_mismatched_query_counts_raises(self):
        """fuse_rankings() should raise ValueError for mismatched query counts."""
        rankings_by_method = {
            "method_a": [[{"target_id": "x", "score": 0.9}]],
            "method_b": [
                [{"target_id": "y", "score": 0.9}],
                [{"target_id": "z", "score": 0.8}],
            ],
        }

        with pytest.raises(ValueError, match="Mismatched query counts"):
            fuse_rankings(rankings_by_method, k=60)

    def test_rrf_score_computation(self):
        """Verify exact RRF score formula: sum of 1/(k + rank)."""
        k = 60
        rankings_by_method = {
            "a": [[
                {"target_id": "t1", "score": 0.9},
                {"target_id": "t2", "score": 0.8},
            ]],
            "b": [[
                {"target_id": "t1", "score": 0.7},
            ]],
        }

        result = fuse_rankings(rankings_by_method, k=k, top_n=10)

        scores = {r["target_id"]: r["score"] for r in result[0]}
        # t1: rank 1 in both -> 1/(60+1) + 1/(60+1) = 2/(61)
        expected_t1 = 1.0 / (k + 1) + 1.0 / (k + 1)
        # t2: rank 2 in 'a' only -> 1/(60+2) = 1/62
        expected_t2 = 1.0 / (k + 2)

        assert abs(scores["t1"] - expected_t1) < 1e-10
        assert abs(scores["t2"] - expected_t2) < 1e-10


class TestFuseAll:
    @pytest.fixture
    def test_cases(self):
        return [{"id": "tc-1"}, {"id": "tc-2"}]

    def test_grouping_and_output_format(self, test_cases):
        """fuse_all() should group by (method, granularity) and produce S0.1 dicts."""
        all_rankings = [
            {
                "test_case_id": "tc-1", "method": "m1", "granularity": "role",
                "ranked_results": [{"target_id": "a", "score": 0.9}],
            },
            {
                "test_case_id": "tc-2", "method": "m1", "granularity": "role",
                "ranked_results": [{"target_id": "b", "score": 0.8}],
            },
            {
                "test_case_id": "tc-1", "method": "m2", "granularity": "role",
                "ranked_results": [{"target_id": "c", "score": 0.7}],
            },
            {
                "test_case_id": "tc-2", "method": "m2", "granularity": "role",
                "ranked_results": [{"target_id": "d", "score": 0.6}],
            },
        ]

        fusion_configs = [{
            "name": "fused-m1-m2",
            "methods": ["m1", "m2"],
            "granularity": "role",
        }]

        result = fuse_all(all_rankings, fusion_configs, test_cases, k=60)

        assert len(result) == 2
        assert result[0]["test_case_id"] == "tc-1"
        assert result[0]["method"] == "fused-m1-m2"
        assert result[0]["granularity"] == "role"
        assert isinstance(result[0]["ranked_results"], list)

        assert result[1]["test_case_id"] == "tc-2"

    def test_missing_method_skipped(self, test_cases):
        """fuse_all() should skip fusion config when a method is missing."""
        all_rankings = [
            {
                "test_case_id": "tc-1", "method": "m1", "granularity": "role",
                "ranked_results": [{"target_id": "a", "score": 0.9}],
            },
            {
                "test_case_id": "tc-2", "method": "m1", "granularity": "role",
                "ranked_results": [{"target_id": "b", "score": 0.8}],
            },
        ]

        fusion_configs = [{
            "name": "fused-missing",
            "methods": ["m1", "m_nonexistent"],
            "granularity": "role",
        }]

        result = fuse_all(all_rankings, fusion_configs, test_cases, k=60)

        assert result == []

    def test_multiple_fusion_configs(self, test_cases):
        """fuse_all() should process multiple fusion configs."""
        all_rankings = [
            {
                "test_case_id": "tc-1", "method": "m1", "granularity": "role",
                "ranked_results": [{"target_id": "a", "score": 0.9}],
            },
            {
                "test_case_id": "tc-2", "method": "m1", "granularity": "role",
                "ranked_results": [{"target_id": "b", "score": 0.8}],
            },
            {
                "test_case_id": "tc-1", "method": "m2", "granularity": "role",
                "ranked_results": [{"target_id": "c", "score": 0.7}],
            },
            {
                "test_case_id": "tc-2", "method": "m2", "granularity": "role",
                "ranked_results": [{"target_id": "d", "score": 0.6}],
            },
            {
                "test_case_id": "tc-1", "method": "m3", "granularity": "role",
                "ranked_results": [{"target_id": "e", "score": 0.5}],
            },
            {
                "test_case_id": "tc-2", "method": "m3", "granularity": "role",
                "ranked_results": [{"target_id": "f", "score": 0.4}],
            },
        ]

        fusion_configs = [
            {"name": "fused-12", "methods": ["m1", "m2"], "granularity": "role"},
            {"name": "fused-13", "methods": ["m1", "m3"], "granularity": "role"},
        ]

        result = fuse_all(all_rankings, fusion_configs, test_cases, k=60)

        # 2 configs * 2 test cases = 4 results
        assert len(result) == 4
        methods = [r["method"] for r in result]
        assert methods.count("fused-12") == 2
        assert methods.count("fused-13") == 2
