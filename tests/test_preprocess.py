"""Tests for query preprocessing (abbreviation expansion)."""

import pytest

from src.preprocess import ABBREVIATION_MAP, expand_abbreviations


class TestAbbreviationMap:
    def test_has_14_entries(self):
        assert len(ABBREVIATION_MAP) == 14

    def test_keys_are_lowercase(self):
        for key in ABBREVIATION_MAP:
            assert key == key.lower()


class TestExpandAbbreviations:
    def test_single_abbreviation(self):
        assert expand_abbreviations("Sr Mgr") == "Sr Manager"

    def test_multiple_abbreviations(self):
        assert expand_abbreviations("Sr Dev Mgr") == "Sr Developer Manager"

    def test_case_insensitive_uppercase(self):
        assert expand_abbreviations("MGR") == "Manager"

    def test_case_insensitive_mixed(self):
        assert expand_abbreviations("mgr") == "Manager"

    def test_word_boundary_devops(self):
        """'Dev' inside 'DevOps' should NOT be expanded."""
        assert expand_abbreviations("DevOps Engineer") == "DevOps Engineer"

    def test_no_match_passthrough(self):
        assert expand_abbreviations("Software Architect") == "Software Architect"

    def test_empty_string(self):
        assert expand_abbreviations("") == ""

    def test_all_abbreviations_expand(self):
        """Spot-check a few more abbreviations."""
        assert expand_abbreviations("Eng") == "Engineer"
        assert expand_abbreviations("Admin") == "Administrator"
        assert expand_abbreviations("Acct") == "Accountant"

    def test_abbreviation_in_sentence(self):
        assert expand_abbreviations("The Dir of Sales") == "The Director of Sales"
