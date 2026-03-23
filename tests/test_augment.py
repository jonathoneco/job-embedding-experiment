"""Tests for target augmentation module."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.augment import (
    _strip_code_fences,
    generate_augmented_targets,
    load_augmented_targets,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def roles():
    """Minimal role list for testing."""
    return [
        {"role": "Software Engineer", "category": "Engineering"},
        {"role": "Data Scientist", "category": "Engineering"},
        {"role": "Product Manager", "category": "Product"},
    ]


@pytest.fixture
def config():
    """Minimal config for augmentation."""
    return {
        "generation": {
            "api_model": "claude-sonnet-4-20250514",
            "max_tokens": 16384,
        },
    }


def _make_api_response(aliases_by_role: dict[str, list[str]], stop_reason: str = "end_turn"):
    """Build a mock Anthropic API response."""
    payload = [
        {"role": role, "aliases": aliases}
        for role, aliases in aliases_by_role.items()
    ]
    response = MagicMock()
    response.stop_reason = stop_reason
    response.content = [MagicMock(text=json.dumps(payload))]
    return response


# ── Tests: _strip_code_fences ────────────────────────────────────────


class TestStripCodeFences:
    def test_plain_json(self):
        text = '[{"role": "A", "aliases": ["a1"]}]'
        assert _strip_code_fences(text) == text

    def test_json_code_fence(self):
        text = '```json\n[{"role": "A", "aliases": ["a1"]}]\n```'
        assert _strip_code_fences(text) == '[{"role": "A", "aliases": ["a1"]}]'

    def test_plain_code_fence(self):
        text = '```\n[{"role": "A"}]\n```'
        assert _strip_code_fences(text) == '[{"role": "A"}]'

    def test_whitespace_around(self):
        text = '  ```json\n{"x": 1}\n```  '
        assert _strip_code_fences(text) == '{"x": 1}'


# ── Tests: generate_augmented_targets ────────────────────────────────


class TestGenerateAugmentedTargets:
    """Test LLM-based augmented target generation with mocked API."""

    def test_output_schema(self, roles, config, tmp_path):
        """Generated targets have all required fields."""
        output_path = str(tmp_path / "augmented.json")

        # Engineering category: 2 roles
        eng_response = _make_api_response({
            "Software Engineer": ["SWE", "Dev", "Software Dev"],
            "Data Scientist": ["DS", "Data Analyst", "ML Scientist"],
        })
        # Product category: 1 role
        prod_response = _make_api_response({
            "Product Manager": ["PM", "PdM", "Product Lead"],
        })

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [eng_response, prod_response]

        with patch("src.augment.anthropic.Anthropic", return_value=mock_client):
            targets = generate_augmented_targets(roles, config, output_path)

        assert len(targets) == 9  # 3 + 3 + 3
        for t in targets:
            assert "id" in t
            assert t["id"].startswith("T-raug-")
            assert "text" in t
            assert "role" in t
            assert "category" in t
            assert t["granularity"] == "role_augmented"
            assert "source_role_id" in t
            assert t["source_role_id"].startswith("T-role-")

    def test_deduplication(self, roles, config, tmp_path):
        """Duplicate aliases (case-insensitive) are removed."""
        output_path = str(tmp_path / "augmented.json")

        eng_response = _make_api_response({
            "Software Engineer": ["SWE", "swe", "Swe", "Dev"],
            "Data Scientist": ["DS"],
        })
        prod_response = _make_api_response({
            "Product Manager": ["PM"],
        })

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [eng_response, prod_response]

        with patch("src.augment.anthropic.Anthropic", return_value=mock_client):
            targets = generate_augmented_targets(roles, config, output_path)

        # SWE appears 3x but should be deduped to 1
        swe_targets = [t for t in targets if t["role"] == "Software Engineer"]
        swe_texts = [t["text"] for t in swe_targets]
        assert len(swe_texts) == 2  # "SWE" (first seen) + "Dev"
        assert "SWE" in swe_texts
        assert "Dev" in swe_texts

    def test_excludes_original_role_name(self, roles, config, tmp_path):
        """Original role name must not appear as an alias."""
        output_path = str(tmp_path / "augmented.json")

        eng_response = _make_api_response({
            "Software Engineer": ["Software Engineer", "SWE", "software engineer"],
            "Data Scientist": ["DS"],
        })
        prod_response = _make_api_response({
            "Product Manager": ["PM", "Product Manager"],
        })

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [eng_response, prod_response]

        with patch("src.augment.anthropic.Anthropic", return_value=mock_client):
            targets = generate_augmented_targets(roles, config, output_path)

        texts = [t["text"] for t in targets]
        assert "Software Engineer" not in texts
        assert "software engineer" not in texts
        assert "Product Manager" not in texts

    def test_role_field_is_canonical(self, roles, config, tmp_path):
        """The 'role' field should be the parent role's canonical name."""
        output_path = str(tmp_path / "augmented.json")

        eng_response = _make_api_response({
            "Software Engineer": ["SWE"],
            "Data Scientist": ["DS"],
        })
        prod_response = _make_api_response({
            "Product Manager": ["PM"],
        })

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [eng_response, prod_response]

        with patch("src.augment.anthropic.Anthropic", return_value=mock_client):
            targets = generate_augmented_targets(roles, config, output_path)

        role_names = {t["role"] for t in targets}
        assert role_names == {"Software Engineer", "Data Scientist", "Product Manager"}

    def test_idempotent_loads_from_cache(self, roles, config, tmp_path):
        """If output file exists, load from cache without calling API."""
        output_path = str(tmp_path / "augmented.json")

        cached_data = [
            {
                "id": "T-raug-00001",
                "text": "SWE",
                "role": "Software Engineer",
                "category": "Engineering",
                "granularity": "role_augmented",
                "source_role_id": "T-role-0001",
            },
        ]
        with open(output_path, "w") as f:
            json.dump(cached_data, f)

        mock_client = MagicMock()
        with patch("src.augment.anthropic.Anthropic", return_value=mock_client):
            targets = generate_augmented_targets(roles, config, output_path)

        # Should not have called the API
        mock_client.messages.create.assert_not_called()
        assert targets == cached_data

    def test_one_call_per_category(self, roles, config, tmp_path):
        """Should make exactly one API call per category."""
        output_path = str(tmp_path / "augmented.json")

        eng_response = _make_api_response({
            "Software Engineer": ["SWE"],
            "Data Scientist": ["DS"],
        })
        prod_response = _make_api_response({
            "Product Manager": ["PM"],
        })

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [eng_response, prod_response]

        with patch("src.augment.anthropic.Anthropic", return_value=mock_client):
            generate_augmented_targets(roles, config, output_path)

        assert mock_client.messages.create.call_count == 2  # 2 categories

    def test_truncation_raises(self, roles, config, tmp_path):
        """Truncated response (max_tokens) should raise ValueError."""
        output_path = str(tmp_path / "augmented.json")

        truncated_response = _make_api_response(
            {"Software Engineer": ["SWE"]},
            stop_reason="max_tokens",
        )

        mock_client = MagicMock()
        mock_client.messages.create.return_value = truncated_response

        with patch("src.augment.anthropic.Anthropic", return_value=mock_client):
            with pytest.raises(ValueError, match="truncated"):
                generate_augmented_targets(roles, config, output_path)

    def test_sequential_ids(self, roles, config, tmp_path):
        """IDs should be sequential across all targets."""
        output_path = str(tmp_path / "augmented.json")

        eng_response = _make_api_response({
            "Software Engineer": ["SWE", "Dev"],
            "Data Scientist": ["DS"],
        })
        prod_response = _make_api_response({
            "Product Manager": ["PM", "PdM"],
        })

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [eng_response, prod_response]

        with patch("src.augment.anthropic.Anthropic", return_value=mock_client):
            targets = generate_augmented_targets(roles, config, output_path)

        ids = [t["id"] for t in targets]
        assert ids == [
            "T-raug-00001", "T-raug-00002", "T-raug-00003",
            "T-raug-00004", "T-raug-00005",
        ]


# ── Tests: load_augmented_targets ────────────────────────────────────


class TestLoadAugmentedTargets:
    def test_load_from_file(self, tmp_path):
        """load_augmented_targets reads and returns the JSON array."""
        data = [
            {
                "id": "T-raug-00001",
                "text": "SWE",
                "role": "Software Engineer",
                "category": "Engineering",
                "granularity": "role_augmented",
                "source_role_id": "T-role-0001",
            },
            {
                "id": "T-raug-00002",
                "text": "Dev",
                "role": "Software Engineer",
                "category": "Engineering",
                "granularity": "role_augmented",
                "source_role_id": "T-role-0001",
            },
        ]
        path = str(tmp_path / "targets.json")
        with open(path, "w") as f:
            json.dump(data, f)

        result = load_augmented_targets(path)
        assert result == data
        assert len(result) == 2


# ── Tests: W-04 integration ─────────────────────────────────────────


class TestBuildTargetSetsAugmented:
    """Test that build_target_sets() handles role_augmented targets."""

    @pytest.fixture
    def base_args(self):
        """Minimal arguments for build_target_sets()."""
        roles = [
            {"role": "Role A", "category": "Cat1"},
            {"role": "Role B", "category": "Cat1"},
        ]
        clusters = [
            {"cluster_label": "Cluster 1", "category": "Cat1",
             "roles": ["Role A", "Role B"]},
        ]
        descriptions = {
            "roles": {"Role A": "desc a", "Role B": "desc b"},
            "categories": {"Cat1": "keywords"},
        }
        return roles, clusters, descriptions

    def test_includes_augmented_when_file_exists(self, base_args, tmp_path, monkeypatch):
        """build_target_sets() returns role_augmented when file exists."""
        from src.targets import build_target_sets

        augmented_data = [
            {
                "id": "T-raug-00001",
                "text": "Alias A",
                "role": "Role A",
                "category": "Cat1",
                "granularity": "role_augmented",
                "source_role_id": "T-role-0001",
            },
        ]
        aug_path = tmp_path / "data" / "taxonomy" / "augmented_targets.json"
        aug_path.parent.mkdir(parents=True)
        with open(aug_path, "w") as f:
            json.dump(augmented_data, f)

        # Patch the augmented path to use our tmp_path
        monkeypatch.setattr(
            "src.targets.AUGMENTED_PATH",
            str(aug_path),
        )

        roles, clusters, descriptions = base_args
        result = build_target_sets(roles, clusters, descriptions)

        assert "role_augmented" in result
        assert result["role_augmented"] == augmented_data

    def test_omits_augmented_when_file_missing(self, base_args, monkeypatch):
        """build_target_sets() does NOT have role_augmented when file absent."""
        from src.targets import build_target_sets

        monkeypatch.setattr(
            "src.targets.AUGMENTED_PATH",
            "/nonexistent/path/augmented_targets.json",
        )

        roles, clusters, descriptions = base_args
        result = build_target_sets(roles, clusters, descriptions)

        assert "role_augmented" not in result
        # Original 5 granularities still present
        assert set(result.keys()) == {
            "role", "role_desc", "cluster", "category_desc", "category",
        }

    def test_existing_granularities_unchanged(self, base_args, tmp_path, monkeypatch):
        """Adding augmented targets does not affect existing 5 granularities."""
        from src.targets import build_target_sets

        augmented_data = [
            {
                "id": "T-raug-00001",
                "text": "Alias A",
                "role": "Role A",
                "category": "Cat1",
                "granularity": "role_augmented",
                "source_role_id": "T-role-0001",
            },
        ]
        aug_path = tmp_path / "data" / "taxonomy" / "augmented_targets.json"
        aug_path.parent.mkdir(parents=True)
        with open(aug_path, "w") as f:
            json.dump(augmented_data, f)

        monkeypatch.setattr("src.targets.AUGMENTED_PATH", str(aug_path))

        roles, clusters, descriptions = base_args
        result = build_target_sets(roles, clusters, descriptions)

        assert "role" in result
        assert "role_desc" in result
        assert "cluster" in result
        assert "category_desc" in result
        assert "category" in result
        assert len(result) == 6


class TestIsCorrectRoleAugmented:
    """Test _is_correct() for role_augmented granularity."""

    def test_role_augmented_matches_on_role_field(self):
        from src.evaluate import _is_correct

        target = {
            "id": "T-raug-00001",
            "text": "SWE",
            "role": "Software Engineer",
            "category": "Engineering",
            "granularity": "role_augmented",
            "source_role_id": "T-role-0001",
        }
        assert _is_correct(target, {"Software Engineer"}, "role_augmented") is True
        assert _is_correct(target, {"Data Scientist"}, "role_augmented") is False

    def test_role_augmented_with_multiple_correct_roles(self):
        from src.evaluate import _is_correct

        target = {
            "id": "T-raug-00001",
            "text": "Dev",
            "role": "Software Engineer",
            "category": "Engineering",
            "granularity": "role_augmented",
            "source_role_id": "T-role-0001",
        }
        assert _is_correct(
            target, {"Software Engineer", "Data Scientist"}, "role_augmented"
        ) is True
        assert _is_correct(
            target, {"Product Manager"}, "role_augmented"
        ) is False
