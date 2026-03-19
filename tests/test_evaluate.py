"""Tests for metrics engine and statistical analysis."""

import pytest

from src.evaluate import compute_metrics, _is_correct
from src.statistics import bootstrap_ci


# ── Fixtures ─────────────────────────────────────────────────────────

def _make_target_sets():
    """Build minimal target sets for testing."""
    return {
        "role": [
            {"id": "T-role-0001", "text": "Role A", "role": "Role A",
             "category": "Cat1", "granularity": "role"},
            {"id": "T-role-0002", "text": "Role B", "role": "Role B",
             "category": "Cat1", "granularity": "role"},
            {"id": "T-role-0003", "text": "Role C", "role": "Role C",
             "category": "Cat2", "granularity": "role"},
            {"id": "T-role-0004", "text": "Role D", "role": "Role D",
             "category": "Cat2", "granularity": "role"},
        ],
        "category": [
            {"id": "T-cat-0001", "text": "Cat1", "roles": ["Role A", "Role B"],
             "category": "Cat1", "granularity": "category"},
            {"id": "T-cat-0002", "text": "Cat2", "roles": ["Role C", "Role D"],
             "category": "Cat2", "granularity": "category"},
        ],
    }


def _make_test_cases(n=3, difficulty="easy"):
    """Build test cases where correct roles are Role A, B, C respectively."""
    roles = ["Role A", "Role B", "Role C"]
    cats = ["Cat1", "Cat1", "Cat2"]
    cases = []
    for i in range(n):
        cases.append({
            "id": f"TC-{i+1:04d}",
            "input_title": f"Title {i+1}",
            "correct_roles": [{"role": roles[i % len(roles)],
                               "category": cats[i % len(cats)]}],
            "difficulty": difficulty,
            "variation_type": "synonym",
            "source": "rule-based",
        })
    return cases


# ── Tests ────────────────────────────────────────────────────────────


class TestMrrComputation:
    """test_mrr_computation: Verify MRR with correct targets at known ranks."""

    def test_mrr_known_ranks(self):
        """3 queries with correct target at rank 1, 2, 3. MRR ~ 0.611."""
        target_sets = _make_target_sets()
        test_cases = [
            {"id": "TC-0001", "input_title": "a",
             "correct_roles": [{"role": "Role A", "category": "Cat1"}],
             "difficulty": "easy", "variation_type": "synonym", "source": "rule-based"},
            {"id": "TC-0002", "input_title": "b",
             "correct_roles": [{"role": "Role B", "category": "Cat1"}],
             "difficulty": "easy", "variation_type": "synonym", "source": "rule-based"},
            {"id": "TC-0003", "input_title": "c",
             "correct_roles": [{"role": "Role C", "category": "Cat2"}],
             "difficulty": "easy", "variation_type": "synonym", "source": "rule-based"},
        ]

        # TC-0001: correct (Role A) at rank 1
        # TC-0002: correct (Role B) at rank 2
        # TC-0003: correct (Role C) at rank 3
        rankings = [
            {"test_case_id": "TC-0001", "method": "test", "granularity": "role",
             "ranked_results": [
                 {"target_id": "T-role-0001", "score": 0.9},
                 {"target_id": "T-role-0002", "score": 0.8},
                 {"target_id": "T-role-0003", "score": 0.7},
             ]},
            {"test_case_id": "TC-0002", "method": "test", "granularity": "role",
             "ranked_results": [
                 {"target_id": "T-role-0001", "score": 0.9},
                 {"target_id": "T-role-0002", "score": 0.8},
                 {"target_id": "T-role-0003", "score": 0.7},
             ]},
            {"test_case_id": "TC-0003", "method": "test", "granularity": "role",
             "ranked_results": [
                 {"target_id": "T-role-0001", "score": 0.9},
                 {"target_id": "T-role-0002", "score": 0.8},
                 {"target_id": "T-role-0003", "score": 0.7},
             ]},
        ]

        metrics = compute_metrics(rankings, test_cases, target_sets, [1, 3, 5])
        # MRR = (1/1 + 1/2 + 1/3) / 3 = (1 + 0.5 + 0.333) / 3 ≈ 0.611
        assert abs(metrics["metrics"]["mrr"] - 0.611) < 0.01


class TestTopkAccuracy:
    """test_topk_accuracy: Verify top-K at different K values."""

    def test_topk_at_various_ranks(self):
        """4 queries, correct at ranks 1, 3, 6, 1. top1=0.5, top3=0.75, top5=0.75."""
        target_sets = _make_target_sets()
        # Add more targets so we can have rank 6
        for i in range(5, 11):
            target_sets["role"].append({
                "id": f"T-role-{i:04d}", "text": f"Role X{i}",
                "role": f"Role X{i}", "category": "Cat1", "granularity": "role",
            })

        test_cases = [
            {"id": "TC-0001", "input_title": "a",
             "correct_roles": [{"role": "Role A", "category": "Cat1"}],
             "difficulty": "easy", "variation_type": "synonym", "source": "rule-based"},
            {"id": "TC-0002", "input_title": "b",
             "correct_roles": [{"role": "Role B", "category": "Cat1"}],
             "difficulty": "easy", "variation_type": "synonym", "source": "rule-based"},
            {"id": "TC-0003", "input_title": "c",
             "correct_roles": [{"role": "Role C", "category": "Cat2"}],
             "difficulty": "medium", "variation_type": "synonym", "source": "rule-based"},
            {"id": "TC-0004", "input_title": "d",
             "correct_roles": [{"role": "Role D", "category": "Cat2"}],
             "difficulty": "medium", "variation_type": "synonym", "source": "rule-based"},
        ]

        def _make_ranking(case_id, correct_rank, correct_target_id):
            """Build a ranking where the correct target is at a specific rank."""
            filler_ids = [f"T-role-{i:04d}" for i in range(5, 11)]
            results = []
            score = 0.95
            for rank in range(1, 11):
                if rank == correct_rank:
                    results.append({"target_id": correct_target_id, "score": score})
                else:
                    if filler_ids:
                        results.append({"target_id": filler_ids.pop(0), "score": score})
                score -= 0.05
            return {
                "test_case_id": case_id, "method": "test", "granularity": "role",
                "ranked_results": results,
            }

        rankings = [
            _make_ranking("TC-0001", 1, "T-role-0001"),  # Rank 1
            _make_ranking("TC-0002", 3, "T-role-0002"),  # Rank 3
            _make_ranking("TC-0003", 6, "T-role-0003"),  # Rank 6
            _make_ranking("TC-0004", 1, "T-role-0004"),  # Rank 1
        ]

        metrics = compute_metrics(rankings, test_cases, target_sets, [1, 3, 5])
        assert metrics["metrics"]["top1"] == 0.5    # 2/4
        assert metrics["metrics"]["top3"] == 0.75   # 3/4
        assert metrics["metrics"]["top5"] == 0.75   # 3/4 (rank 6 is beyond top 5)


class TestCategoryAccuracy:
    """test_category_accuracy: Verify category matching at rank 1."""

    def test_category_match(self):
        """Rank-1 target category matches expected for 2 of 3 cases."""
        target_sets = _make_target_sets()
        test_cases = [
            {"id": "TC-0001", "input_title": "a",
             "correct_roles": [{"role": "Role A", "category": "Cat1"}],
             "difficulty": "easy", "variation_type": "synonym", "source": "rule-based"},
            {"id": "TC-0002", "input_title": "b",
             "correct_roles": [{"role": "Role B", "category": "Cat1"}],
             "difficulty": "easy", "variation_type": "synonym", "source": "rule-based"},
            {"id": "TC-0003", "input_title": "c",
             "correct_roles": [{"role": "Role C", "category": "Cat2"}],
             "difficulty": "easy", "variation_type": "synonym", "source": "rule-based"},
        ]

        # TC-0001: rank-1 is T-role-0001 (Cat1) — matches Cat1 ✓
        # TC-0002: rank-1 is T-role-0003 (Cat2) — expected Cat1 ✗
        # TC-0003: rank-1 is T-role-0003 (Cat2) — matches Cat2 ✓
        rankings = [
            {"test_case_id": "TC-0001", "method": "test", "granularity": "role",
             "ranked_results": [
                 {"target_id": "T-role-0001", "score": 0.9},
                 {"target_id": "T-role-0002", "score": 0.8},
             ]},
            {"test_case_id": "TC-0002", "method": "test", "granularity": "role",
             "ranked_results": [
                 {"target_id": "T-role-0003", "score": 0.9},
                 {"target_id": "T-role-0002", "score": 0.8},
             ]},
            {"test_case_id": "TC-0003", "method": "test", "granularity": "role",
             "ranked_results": [
                 {"target_id": "T-role-0003", "score": 0.9},
                 {"target_id": "T-role-0001", "score": 0.8},
             ]},
        ]

        metrics = compute_metrics(rankings, test_cases, target_sets, [1, 3, 5])
        # 2 of 3 have correct category at rank 1
        assert abs(metrics["metrics"]["category_accuracy"] - 2.0 / 3.0) < 0.01


class TestIsCorrectWithAcceptSets:
    """test_is_correct_with_accept_sets: Multiple correct roles and granularity handling."""

    def test_role_granularity_match(self):
        """Role granularity: direct role name match."""
        target = {"id": "T-role-0001", "role": "Role A", "category": "Cat1"}
        assert _is_correct(target, {"Role A", "Role B"}, "role") is True
        assert _is_correct(target, {"Role C"}, "role") is False

    def test_role_desc_granularity_match(self):
        """role_desc granularity: same as role, uses 'role' field."""
        target = {"id": "T-rdesc-0001", "role": "Role A", "category": "Cat1"}
        assert _is_correct(target, {"Role A"}, "role_desc") is True
        assert _is_correct(target, {"Role C"}, "role_desc") is False

    def test_category_granularity_match(self):
        """Category granularity: any role in 'roles' array overlaps."""
        target = {"id": "T-cat-0001", "roles": ["Role A", "Role B"], "category": "Cat1"}
        assert _is_correct(target, {"Role A"}, "category") is True
        assert _is_correct(target, {"Role B"}, "category") is True
        assert _is_correct(target, {"Role C"}, "category") is False

    def test_cluster_granularity_match(self):
        """Cluster granularity uses 'roles' array like category."""
        target = {"id": "T-clust-0001", "roles": ["Role A", "Role B"], "category": "Cat1"}
        assert _is_correct(target, {"Role B"}, "cluster") is True
        assert _is_correct(target, {"Role X"}, "cluster") is False

    def test_category_desc_granularity_match(self):
        """category_desc granularity uses 'roles' array."""
        target = {"id": "T-cdesc-0001", "roles": ["Role A"], "category": "Cat1"}
        assert _is_correct(target, {"Role A", "Role B"}, "category_desc") is True
        assert _is_correct(target, {"Role C"}, "category_desc") is False


class TestSimilarityGap:
    """test_similarity_gap: Known score gaps produce expected mean."""

    def test_known_gaps(self):
        """Scores [0.9, 0.7], [0.8, 0.6], [0.5, 0.4] → gaps 0.2, 0.2, 0.1 → mean 0.167."""
        target_sets = _make_target_sets()
        test_cases = _make_test_cases(3)

        rankings = [
            {"test_case_id": "TC-0001", "method": "test", "granularity": "role",
             "ranked_results": [
                 {"target_id": "T-role-0001", "score": 0.9},
                 {"target_id": "T-role-0002", "score": 0.7},
             ]},
            {"test_case_id": "TC-0002", "method": "test", "granularity": "role",
             "ranked_results": [
                 {"target_id": "T-role-0001", "score": 0.8},
                 {"target_id": "T-role-0002", "score": 0.6},
             ]},
            {"test_case_id": "TC-0003", "method": "test", "granularity": "role",
             "ranked_results": [
                 {"target_id": "T-role-0001", "score": 0.5},
                 {"target_id": "T-role-0002", "score": 0.4},
             ]},
        ]

        metrics = compute_metrics(rankings, test_cases, target_sets, [1, 3, 5])
        # Gaps: 0.2, 0.2, 0.1 → mean = 0.5 / 3 ≈ 0.1667
        assert abs(metrics["metrics"]["mean_similarity_gap"] - 0.1667) < 0.01


class TestBootstrapCi:
    """test_bootstrap_ci: Deterministic data and seed reproducibility."""

    def test_all_correct_narrow_ci(self):
        """When all queries are correct at rank 1, CI should be tight around 1.0."""
        target_sets = _make_target_sets()
        test_cases = [
            {"id": "TC-0001", "input_title": "a",
             "correct_roles": [{"role": "Role A", "category": "Cat1"}],
             "difficulty": "easy", "variation_type": "synonym", "source": "rule-based"},
            {"id": "TC-0002", "input_title": "b",
             "correct_roles": [{"role": "Role B", "category": "Cat1"}],
             "difficulty": "easy", "variation_type": "synonym", "source": "rule-based"},
            {"id": "TC-0003", "input_title": "c",
             "correct_roles": [{"role": "Role C", "category": "Cat2"}],
             "difficulty": "easy", "variation_type": "synonym", "source": "rule-based"},
        ]

        # All correct at rank 1
        rankings = [
            {"test_case_id": "TC-0001", "method": "test", "granularity": "role",
             "ranked_results": [
                 {"target_id": "T-role-0001", "score": 0.9},
                 {"target_id": "T-role-0002", "score": 0.8},
             ]},
            {"test_case_id": "TC-0002", "method": "test", "granularity": "role",
             "ranked_results": [
                 {"target_id": "T-role-0002", "score": 0.9},
                 {"target_id": "T-role-0001", "score": 0.8},
             ]},
            {"test_case_id": "TC-0003", "method": "test", "granularity": "role",
             "ranked_results": [
                 {"target_id": "T-role-0003", "score": 0.9},
                 {"target_id": "T-role-0001", "score": 0.8},
             ]},
        ]

        def top1_fn(rankings, cases, ts):
            tl = {}
            for _g, tgts in ts.items():
                for t in tgts:
                    tl[t["id"]] = t
            hits = 0
            for r in rankings:
                cl = {c["id"]: c for c in cases}
                case = cl[r["test_case_id"]]
                correct = {cr["role"] for cr in case["correct_roles"]}
                rank1 = tl[r["ranked_results"][0]["target_id"]]
                if _is_correct(rank1, correct, "role"):
                    hits += 1
            return hits / len(rankings)

        lower, upper = bootstrap_ci(
            rankings, test_cases, target_sets, top1_fn,
            n_resamples=500, seed=42,
        )
        # All correct → CI should be [1.0, 1.0]
        assert lower == 1.0
        assert upper == 1.0

    def test_seed_reproducibility(self):
        """Same seed produces identical CI bounds."""
        target_sets = _make_target_sets()
        test_cases = _make_test_cases(3)

        rankings = [
            {"test_case_id": f"TC-{i+1:04d}", "method": "test", "granularity": "role",
             "ranked_results": [
                 {"target_id": "T-role-0001", "score": 0.9},
                 {"target_id": "T-role-0002", "score": 0.8},
             ]}
            for i in range(3)
        ]

        def mrr_fn(rankings, cases, ts):
            return 0.5  # Constant for reproducibility test

        ci_a = bootstrap_ci(rankings, test_cases, target_sets, mrr_fn,
                            n_resamples=100, seed=42)
        ci_b = bootstrap_ci(rankings, test_cases, target_sets, mrr_fn,
                            n_resamples=100, seed=42)
        assert ci_a == ci_b

        # Different seed should also give same result for constant fn
        ci_c = bootstrap_ci(rankings, test_cases, target_sets, mrr_fn,
                            n_resamples=100, seed=99)
        assert ci_c == ci_a  # Constant function always returns 0.5
