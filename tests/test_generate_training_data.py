"""Tests for training data generator."""

import json
import os

import pytest

from src.generate_training_data import (
    generate_contrastive_pairs,
    generate_tsdae_corpus,
    generate_variants,
    mine_hard_negatives,
)

# Small role sample spanning 3 categories for fast tests
_SAMPLE_ROLES = [
    {"role": "Software Engineer", "category": "Engineering"},
    {"role": "Data Analyst", "category": "Engineering"},
    {"role": "Systems Administrator", "category": "Engineering"},
    {"role": "Project Manager", "category": "Management"},
    {"role": "Product Manager", "category": "Management"},
    {"role": "Operations Director", "category": "Management"},
    {"role": "Sales Representative", "category": "Sales"},
    {"role": "Account Executive", "category": "Sales"},
    {"role": "Business Development Manager", "category": "Sales"},
    {"role": "Marketing Coordinator", "category": "Marketing"},
]


class TestGenerateVariants:
    def test_produces_variants_for_each_role(self):
        variants = generate_variants(_SAMPLE_ROLES, seed=42)
        assert set(variants.keys()) == {r["role"] for r in _SAMPLE_ROLES}

    def test_variant_count_per_role(self):
        """Each role should have 5-8 variants (at least the 5 base transforms)."""
        variants = generate_variants(_SAMPLE_ROLES, seed=42)
        for name, var_list in variants.items():
            assert len(var_list) >= 5, f"{name} has only {len(var_list)} variants"
            assert len(var_list) <= 8, f"{name} has {len(var_list)} variants"

    def test_no_exact_duplicates(self):
        variants = generate_variants(_SAMPLE_ROLES, seed=42)
        for name, var_list in variants.items():
            assert len(var_list) == len(set(var_list)), (
                f"Duplicate variants for {name}"
            )

    def test_variants_differ_from_canonical(self):
        variants = generate_variants(_SAMPLE_ROLES, seed=42)
        for name, var_list in variants.items():
            assert name not in var_list, f"Canonical '{name}' found in its variants"

    def test_seed_determinism(self):
        v1 = generate_variants(_SAMPLE_ROLES, seed=99)
        v2 = generate_variants(_SAMPLE_ROLES, seed=99)
        assert v1 == v2

    def test_seed_offset(self):
        """Different base seeds produce different variants."""
        v1 = generate_variants(_SAMPLE_ROLES, seed=1)
        v2 = generate_variants(_SAMPLE_ROLES, seed=2)
        assert v1 != v2


class TestMineHardNegatives:
    def test_returns_negatives_for_each_role(self):
        negs = mine_hard_negatives(_SAMPLE_ROLES)
        assert set(negs.keys()) == {r["role"] for r in _SAMPLE_ROLES}

    def test_negatives_from_different_category(self):
        cat_lookup = {r["role"]: r["category"] for r in _SAMPLE_ROLES}
        negs = mine_hard_negatives(_SAMPLE_ROLES)
        for role_name, neg_list in negs.items():
            role_cat = cat_lookup[role_name]
            for neg in neg_list:
                assert cat_lookup[neg] != role_cat, (
                    f"Negative '{neg}' has same category as '{role_name}'"
                )

    def test_returns_up_to_5_negatives(self):
        negs = mine_hard_negatives(_SAMPLE_ROLES)
        for neg_list in negs.values():
            assert 1 <= len(neg_list) <= 5


class TestGenerateContrastivePairs:
    def test_jsonl_format(self, tmp_path):
        out = tmp_path / "pairs.jsonl"
        count = generate_contrastive_pairs(_SAMPLE_ROLES, seed=42, output_path=str(out))
        assert count > 0
        assert out.exists()

        with open(out) as f:
            for line in f:
                obj = json.loads(line)
                assert "anchor" in obj
                assert "positive" in obj
                assert "negative" in obj

    def test_pair_count_matches_return(self, tmp_path):
        out = tmp_path / "pairs.jsonl"
        count = generate_contrastive_pairs(_SAMPLE_ROLES, seed=42, output_path=str(out))
        with open(out) as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) == count

    def test_deterministic_output(self, tmp_path):
        out1 = tmp_path / "p1.jsonl"
        out2 = tmp_path / "p2.jsonl"
        generate_contrastive_pairs(_SAMPLE_ROLES, seed=42, output_path=str(out1))
        generate_contrastive_pairs(_SAMPLE_ROLES, seed=42, output_path=str(out2))
        assert out1.read_text() == out2.read_text()

    def test_negatives_from_different_category(self, tmp_path):
        """Negative in each triplet should be from a different category than anchor."""
        cat_lookup = {r["role"]: r["category"] for r in _SAMPLE_ROLES}
        out = tmp_path / "pairs.jsonl"
        generate_contrastive_pairs(_SAMPLE_ROLES, seed=42, output_path=str(out))

        with open(out) as f:
            for line in f:
                obj = json.loads(line)
                anchor_cat = cat_lookup[obj["anchor"]]
                neg_cat = cat_lookup[obj["negative"]]
                assert anchor_cat != neg_cat, (
                    f"Anchor '{obj['anchor']}' and negative '{obj['negative']}' "
                    f"share category '{anchor_cat}'"
                )


class TestGenerateTsdaeCorpus:
    def test_contains_canonical_roles(self, tmp_path):
        variants = generate_variants(_SAMPLE_ROLES, seed=42)
        out = tmp_path / "corpus.txt"
        generate_tsdae_corpus(_SAMPLE_ROLES, variants, str(out))

        lines = out.read_text().splitlines()
        for role in _SAMPLE_ROLES:
            assert role["role"] in lines

    def test_contains_variants(self, tmp_path):
        variants = generate_variants(_SAMPLE_ROLES, seed=42)
        out = tmp_path / "corpus.txt"
        generate_tsdae_corpus(_SAMPLE_ROLES, variants, str(out))

        lines = set(out.read_text().splitlines())
        for var_list in variants.values():
            for v in var_list:
                assert v in lines

    def test_deduplicated(self, tmp_path):
        variants = generate_variants(_SAMPLE_ROLES, seed=42)
        out = tmp_path / "corpus.txt"
        count = generate_tsdae_corpus(_SAMPLE_ROLES, variants, str(out))

        lines = out.read_text().splitlines()
        assert len(lines) == len(set(lines)), "Corpus contains duplicates"
        assert count == len(lines)

    def test_count_returned(self, tmp_path):
        variants = generate_variants(_SAMPLE_ROLES, seed=42)
        out = tmp_path / "corpus.txt"
        count = generate_tsdae_corpus(_SAMPLE_ROLES, variants, str(out))
        assert count > len(_SAMPLE_ROLES)  # Should have roles + variants
