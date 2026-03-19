"""Tests for validate, deduplicate, and split functions."""

import pytest

from src.taxonomy import parse_taxonomy
from src.validate import validate_cases, deduplicate_cases, split_dev_test


# ── Fixtures ─────────────────────────────────────────────────────────

MOCK_ROLES = [
    {"role": "Software Engineer", "category": "Software Engineering"},
    {"role": "Data Scientist", "category": "Data & Analytics"},
    {"role": "HR Manager", "category": "Human Resources"},
    {"role": "Financial Analyst", "category": "Finance"},
    {"role": "Product Manager", "category": "Project Management"},
]


def _make_case(
    case_id="TC-0001",
    title="Sr. Software Engineer",
    role="Software Engineer",
    category="Software Engineering",
    difficulty="easy",
    variation_type="level-prefix",
    source="rule-based",
    notes="test case",
):
    return {
        "id": case_id,
        "input_title": title,
        "correct_roles": [{"role": role, "category": category}],
        "difficulty": difficulty,
        "variation_type": variation_type,
        "source": source,
        "notes": notes,
    }


# ── Tests ────────────────────────────────────────────────────────────


class TestValidateMissingFields:
    """test_validate_missing_fields: Cases missing required fields raise ValueError."""

    def test_missing_id(self):
        case = _make_case()
        del case["id"]
        with pytest.raises(ValueError, match="missing required fields"):
            validate_cases([case], MOCK_ROLES)

    def test_missing_input_title(self):
        case = _make_case()
        del case["input_title"]
        with pytest.raises(ValueError, match="missing required fields"):
            validate_cases([case], MOCK_ROLES)

    def test_missing_correct_roles(self):
        case = _make_case()
        del case["correct_roles"]
        with pytest.raises(ValueError, match="missing required fields"):
            validate_cases([case], MOCK_ROLES)

    def test_missing_difficulty(self):
        case = _make_case()
        del case["difficulty"]
        with pytest.raises(ValueError, match="missing required fields"):
            validate_cases([case], MOCK_ROLES)

    def test_missing_source(self):
        case = _make_case()
        del case["source"]
        with pytest.raises(ValueError, match="missing required fields"):
            validate_cases([case], MOCK_ROLES)

    def test_missing_variation_type(self):
        case = _make_case()
        del case["variation_type"]
        with pytest.raises(ValueError, match="missing required fields"):
            validate_cases([case], MOCK_ROLES)


class TestValidateTaxonomyMembership:
    """test_validate_taxonomy_membership: Non-existent roles raise ValueError."""

    def test_nonexistent_role(self):
        case = _make_case(role="Nonexistent Role", category="Fake Category")
        with pytest.raises(ValueError, match="not found in taxonomy"):
            validate_cases([case], MOCK_ROLES)

    def test_empty_correct_roles(self):
        case = _make_case()
        case["correct_roles"] = []
        with pytest.raises(ValueError, match="non-empty list"):
            validate_cases([case], MOCK_ROLES)

    def test_valid_case_passes(self):
        case = _make_case()
        result = validate_cases([case], MOCK_ROLES)
        assert result == [case]


class TestDeduplicate:
    """test_deduplicate: Near-duplicates removed, higher difficulty kept."""

    def test_high_overlap_removes_lower_difficulty(self):
        cases = [
            _make_case(
                case_id="TC-0001",
                title="Senior Software Engineer",
                difficulty="easy",
            ),
            _make_case(
                case_id="TC-0002",
                title="Software Engineer Senior",  # Same words, different order
                difficulty="hard",
            ),
            _make_case(
                case_id="TC-0003",
                title="Data Scientist Lead",
                role="Data Scientist",
                category="Data & Analytics",
                difficulty="medium",
            ),
        ]
        result = deduplicate_cases(cases, threshold=0.75)
        # The two "Senior Software Engineer" have word Jaccard 1.0, hard kept
        assert len(result) == 2
        ids = {c["id"] for c in result}
        assert "TC-0002" in ids  # hard difficulty kept
        assert "TC-0003" in ids  # unrelated, kept
        assert "TC-0001" not in ids  # easy, removed

    def test_below_threshold_kept(self):
        cases = [
            _make_case(case_id="TC-0001", title="Software Engineer"),
            _make_case(
                case_id="TC-0002",
                title="HR Manager",
                role="HR Manager",
                category="Human Resources",
            ),
        ]
        result = deduplicate_cases(cases, threshold=0.75)
        assert len(result) == 2


class TestSplitStratification:
    """test_split_stratification: Proportions preserved in split."""

    def test_proportions_preserved(self):
        # Create 20 cases with known difficulty/category split
        cases = []
        for i in range(10):
            cases.append(
                _make_case(
                    case_id=f"TC-{i+1:04d}",
                    title=f"Software Engineer Variant {i}",
                    difficulty="easy",
                )
            )
        for i in range(10, 20):
            cases.append(
                _make_case(
                    case_id=f"TC-{i+1:04d}",
                    title=f"Data Scientist Variant {i}",
                    role="Data Scientist",
                    category="Data & Analytics",
                    difficulty="medium",
                )
            )

        dev, test = split_dev_test(cases, dev_size=6, seed=42)

        assert len(dev) == 6
        assert len(test) == 14

        # Check stratification: dev should have ~3 easy, ~3 medium
        dev_easy = sum(1 for c in dev if c["difficulty"] == "easy")
        dev_medium = sum(1 for c in dev if c["difficulty"] == "medium")
        assert 2 <= dev_easy <= 4, f"Expected 2-4 easy in dev, got {dev_easy}"
        assert 2 <= dev_medium <= 4, f"Expected 2-4 medium in dev, got {dev_medium}"


class TestIdFormat:
    """test_id_format: Verify TC-NNNN format with zero-padding."""

    def test_valid_id_format(self):
        import re

        pattern = re.compile(r"^TC-\d{4}$")

        cases = [
            _make_case(case_id=f"TC-{i+1:04d}", title=f"Unique Title {i}")
            for i in range(5)
        ]

        for case in cases:
            assert pattern.match(case["id"]), (
                f"ID '{case['id']}' does not match TC-NNNN format"
            )

    def test_zero_padded(self):
        assert _make_case(case_id="TC-0001")["id"] == "TC-0001"
        assert _make_case(case_id="TC-0099")["id"] == "TC-0099"

    def test_id_uniqueness_enforced(self):
        cases = [
            _make_case(case_id="TC-0001", title="Title A"),
            _make_case(case_id="TC-0001", title="Title B"),
        ]
        with pytest.raises(ValueError, match="Duplicate case id"):
            validate_cases(cases, MOCK_ROLES)
