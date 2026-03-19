"""Tests for taxonomy parsing, category grouping, cluster building, and targets."""

import os

import pytest

from src.taxonomy import parse_taxonomy, get_categories
from src.clusters import build_clusters
from src.targets import build_target_sets


# ── Synthetic markdown for unit tests ─────────────────────────────────

SYNTHETIC_MD = """\
# Job Roles

## Software & Engineering
- Frontend Developer
- Backend Developer
- DevOps Engineer (Cloud)
- QA Analyst
- Systems Architect

## Finance & Accounting
- Financial Analyst
- Tax Specialist
- Accounts Payable Coordinator
- Revenue Accountant
- Budget Planner

## Human Resources & People
- HR Business Partner
- Recruiter
- Compensation & Benefits Analyst
- DEI Program Manager
- Learning & Development Specialist
"""


class TestParseTaxonomy:
    """Tests for parse_taxonomy function."""

    def test_parse_taxonomy(self, tmp_path):
        """Parse a synthetic markdown with 3 categories and 5 roles each."""
        md_file = tmp_path / "test-roles.md"
        md_file.write_text(SYNTHETIC_MD)

        roles = parse_taxonomy(str(md_file))

        # Correct total count
        assert len(roles) == 15

        # Correct structure
        for entry in roles:
            assert "role" in entry
            assert "category" in entry
            assert isinstance(entry["role"], str)
            assert isinstance(entry["category"], str)

        # Categories present
        categories = {r["category"] for r in roles}
        assert categories == {
            "Software & Engineering",
            "Finance & Accounting",
            "Human Resources & People",
        }

        # Roles with parenthetical qualifiers preserved
        role_names = [r["role"] for r in roles]
        assert "DevOps Engineer (Cloud)" in role_names

        # Categories with & preserved
        assert any(
            r["category"] == "Finance & Accounting" for r in roles
        )

        # 5 roles per category
        for cat in categories:
            cat_roles = [r for r in roles if r["category"] == cat]
            assert len(cat_roles) == 5

    def test_get_categories(self, tmp_path):
        """Verify grouping from parsed roles."""
        md_file = tmp_path / "test-roles.md"
        md_file.write_text(SYNTHETIC_MD)

        roles = parse_taxonomy(str(md_file))
        cats = get_categories(roles)

        # 3 categories
        assert len(cats) == 3

        # Each category has 5 roles
        for cat_name, role_list in cats.items():
            assert len(role_list) == 5

        # Spot-check specific groupings
        assert "Recruiter" in cats["Human Resources & People"]
        assert "Tax Specialist" in cats["Finance & Accounting"]
        assert "Frontend Developer" in cats["Software & Engineering"]

    @pytest.mark.skipif(not os.path.exists("job-roles.md"), reason="requires taxonomy file")
    def test_build_clusters(self):
        """Build clusters from the REAL taxonomy and validate constraints."""
        roles = parse_taxonomy("job-roles.md")
        categories = get_categories(roles)
        clusters = build_clusters(roles)

        # Cluster count in expected range
        assert 80 <= len(clusters) <= 120, (
            f"Expected 80-120 clusters, got {len(clusters)}"
        )

        # Every role appears in exactly one cluster
        all_taxonomy_roles = {r["role"] for r in roles}
        role_to_cluster: dict[str, str] = {}
        for cluster in clusters:
            for role in cluster["roles"]:
                assert role not in role_to_cluster, (
                    f"Role '{role}' appears in clusters "
                    f"'{role_to_cluster[role]}' and '{cluster['cluster_label']}'"
                )
                role_to_cluster[role] = cluster["cluster_label"]

        clustered_roles = set(role_to_cluster.keys())
        assert clustered_roles == all_taxonomy_roles, (
            f"Missing: {all_taxonomy_roles - clustered_roles}, "
            f"Extra: {clustered_roles - all_taxonomy_roles}"
        )

        # No empty clusters
        for cluster in clusters:
            assert len(cluster["roles"]) > 0, (
                f"Empty cluster: {cluster['cluster_label']}"
            )

        # Small categories (<10 roles) have exactly 1 cluster
        for cat_name, cat_roles in categories.items():
            if len(cat_roles) < 10:
                cat_clusters = [
                    c for c in clusters if c["category"] == cat_name
                ]
                assert len(cat_clusters) == 1, (
                    f"Category '{cat_name}' has {len(cat_roles)} roles "
                    f"but {len(cat_clusters)} clusters (expected 1)"
                )

        # Verify cluster schema
        for cluster in clusters:
            assert "cluster_label" in cluster
            assert "category" in cluster
            assert "roles" in cluster
            assert isinstance(cluster["cluster_label"], str)
            assert isinstance(cluster["category"], str)
            assert isinstance(cluster["roles"], list)

    def test_build_target_sets(self):
        """Build target sets from mock data and validate structure."""
        # Mock roles: 3 roles across 2 categories
        mock_roles = [
            {"role": "Alpha Engineer", "category": "Engineering"},
            {"role": "Beta Analyst", "category": "Engineering"},
            {"role": "Gamma Manager", "category": "Operations"},
        ]

        # Mock clusters: 2 clusters
        mock_clusters = [
            {
                "cluster_label": "Engineering Core",
                "category": "Engineering",
                "roles": ["Alpha Engineer", "Beta Analyst"],
            },
            {
                "cluster_label": "Operations Management",
                "category": "Operations",
                "roles": ["Gamma Manager"],
            },
        ]

        # Mock descriptions
        mock_descriptions = {
            "roles": {
                "Alpha Engineer": "Designs and builds core platform systems",
                "Beta Analyst": "Analyzes data to drive engineering decisions",
                "Gamma Manager": "Oversees daily operational workflows and teams",
            },
            "categories": {
                "Engineering": "engineering, design, systems, platform, development",
                "Operations": "operations, management, workflow, coordination",
            },
        }

        target_sets = build_target_sets(
            mock_roles, mock_clusters, mock_descriptions
        )

        # Correct granularity keys
        assert set(target_sets.keys()) == {
            "role", "role_desc", "cluster", "category_desc", "category",
        }

        # Correct target counts
        assert len(target_sets["role"]) == 3
        assert len(target_sets["role_desc"]) == 3
        assert len(target_sets["cluster"]) == 2
        assert len(target_sets["category_desc"]) == 2
        assert len(target_sets["category"]) == 2

        # ID format validation
        assert target_sets["role"][0]["id"] == "T-role-0001"
        assert target_sets["role_desc"][0]["id"] == "T-rdesc-0001"
        assert target_sets["cluster"][0]["id"] == "T-clust-0001"
        assert target_sets["category_desc"][0]["id"] == "T-cdesc-0001"
        assert target_sets["category"][0]["id"] == "T-cat-0001"

        # Text fields non-empty
        for granularity, targets in target_sets.items():
            for target in targets:
                assert target["text"], (
                    f"Empty text in {granularity} target {target['id']}"
                )

        # role and role_desc targets have "role" field (string), not "roles"
        for target in target_sets["role"]:
            assert "role" in target
            assert isinstance(target["role"], str)
            assert "roles" not in target

        for target in target_sets["role_desc"]:
            assert "role" in target
            assert isinstance(target["role"], str)
            assert "roles" not in target
            # role_desc text includes description
            assert ":" in target["text"]

        # cluster, category_desc, and category targets have "roles" (list)
        for granularity in ["cluster", "category_desc", "category"]:
            for target in target_sets[granularity]:
                assert "roles" in target
                assert isinstance(target["roles"], list)
                assert len(target["roles"]) > 0
                assert "role" not in target  # no singular "role" field

        # Every role appears in at least one target per granularity
        all_role_names = {r["role"] for r in mock_roles}
        for granularity in ["role", "role_desc"]:
            target_roles = {t["role"] for t in target_sets[granularity]}
            assert target_roles == all_role_names, (
                f"Missing roles in {granularity}: "
                f"{all_role_names - target_roles}"
            )

        for granularity in ["cluster", "category_desc", "category"]:
            target_roles: set[str] = set()
            for t in target_sets[granularity]:
                target_roles.update(t["roles"])
            assert target_roles == all_role_names, (
                f"Missing roles in {granularity}: "
                f"{all_role_names - target_roles}"
            )
