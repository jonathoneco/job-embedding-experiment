# Spec 02 — Taxonomy & Targets (C1)

**Dependencies**: C0 (project setup)
**Refs**: Spec 00 (role schema, target schema, config, file paths)

---

## Overview

Parse the taxonomy source file, build subclusters, generate role/category descriptions via Claude API, and construct the 5 granularity-level target sets. All outputs are committed JSON files under `data/taxonomy/`.

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/taxonomy.py` | Parse `job-roles.md` → structured role list |
| `src/clusters.py` | Build subclusters from categories |
| `src/descriptions.py` | Generate descriptions via Claude API |
| `src/targets.py` | Construct 5-level target sets |
| `scripts/prep_taxonomy.py` | Orchestrator: run taxonomy → clusters → descriptions → targets |
| `tests/test_taxonomy.py` | Unit tests for taxonomy parsing |

---

## Subcomponent: Taxonomy Parser (`src/taxonomy.py`)

### Function: `parse_taxonomy(source_path: str) -> list[dict]`

Parse `job-roles.md` markdown into a list of role dicts.

**Parsing rules**:
- Lines matching `## <Category Name>` start a new category
- Lines matching `- <Role Name>` within a category define roles
- Ignore blank lines and the `# Job Roles` header
- Strip whitespace from role names and category names

**Output**: List of `{"role": str, "category": str}` dicts (spec 00 Role schema). Written to `data/taxonomy/roles.json`.

**Acceptance criteria**:
- Produces exactly 692 roles across 42 categories
- Category names match `## ` headers exactly (including parenthetical suffixes like `(AEC)`)
- Role names match `- ` entries exactly (including parenthetical qualifiers like `(Office)`)
- Output written as pretty-printed JSON with 2-space indent

### Function: `get_categories(roles: list[dict]) -> dict[str, list[str]]`

Return `{category_name: [role_name, ...]}` mapping. Used by other modules.

---

## Subcomponent: Cluster Builder (`src/clusters.py`)

### Function: `build_clusters(roles: list[dict]) -> list[dict]`

Build subclusters using the category-scaffold approach.

**Splitting rules**:
- Categories with **fewer than 10 roles**: Keep intact as a single cluster. Label = category name.
- Categories with **10-15 roles**: Split into **2** subclusters.
- Categories with **16-25 roles**: Split into **2-3** subclusters.
- Categories with **26+ roles**: Split into **3-4** subclusters.

**Subcluster labels**: Descriptive functional groupings (not numbered). Example: "HR Strategy & Business Partnership" rather than "Human Resources - Cluster 1".

**Subcluster definitions are hardcoded** in `clusters.py` as a Python dict mapping `category -> list of {"label": str, "roles": list[str]}`. This is manual curation, not algorithmic clustering.

### Example Subclusters (5 largest categories)

**Information Technology (39 roles → 4 subclusters)**:
1. "IT Operations & Support" — IT Administrator, IT Support Specialist, IT Service Desk Manager, Desktop Engineer, IT Professional, IT Operations Analyst
2. "IT Infrastructure & Engineering" — Cloud Administrator, Network Administrator, Network Engineer, DevOps Engineer, Site Reliability Engineer (SRE), Systems Administrator, Configuration Manager
3. "IT Security & Governance" — GRC Analyst, Identity & Access Management Analyst, Incident Response Coordinator, IT Compliance Analyst, IT Governance Analyst, SOC Manager, IT Asset Manager, IT Financial Management (FinOps) Analyst, IT Vendor Manager, IT Procurement Analyst
4. "IT Product & Platform" — IT Product Manager, Platform Product Manager, Solutions Architect, Business Systems Analyst, QA Analyst (Software), Release Train Engineer, Scrum Program Lead, Service Delivery Manager, Technical Program Manager, SharePoint Administrator, M365 Administrator, Power Platform Admin, CMDB Administrator, Launch Infrastructure Manager, Observability/Monitoring Analyst, Field IT Manager

**Finance (34 roles → 3 subclusters)**:
1. "Financial Accounting & Reporting" — Accountant, Senior Accountant, Revenue Accountant, Cost Accountant, Fixed Assets Accountant, Financial Accounting, Financial Reporting Analyst, Controller, Bookkeeper, Hedge Accounting Specialist
2. "Financial Planning & Analysis" — Financial Analyst, FP&A Manager, Financial Planning and Analysis, Financial Operations, Capital Planning Analyst, Billing Analyst, M&A Analyst, Valuation Analyst, Finance Business Partner, Finance Manager, ALM Analyst, Vendor Engagement Manager
3. "Tax, Compliance & Treasury" — Tax Manager, Tax Specialist, Income Tax Compliance Manager, SOX Compliance Analyst, Internal Auditor, Audit, Risk and Compliance (ARC) Data Solution Manager, Treasury Analyst, Treasury Operations Analyst, AP Specialist, AR Specialist, Payroll Specialist, Credit Analyst

**Design (33 roles → 3 subclusters)**:
1. "UX Design & Research" — User Experience Designer, User Experience Researcher, UX Researcher, User Experience Writer, UX Writer, Interaction Designer, Service Designer, Experience Designer, Design Researcher, Information Architect
2. "Visual & Product Design" — Graphic Designer, Brand Designer, Visual Designer, Motion Designer, Packaging Designer, Industrial Designer, Material Designer, Environmental Designer, Product Designer, Content Designer, Information Designer, Design Engineer
3. "Design Leadership & Operations" — Creative Director, Design Director, Experience Director, Design Operations Manager, Design Program Manager, Design Strategist, Design Systems Manager, Brand Strategist, Research Operations Manager, Video Producer (Admin)

**Software Engineering (31 roles → 3 subclusters)**:
1. "Application Development" — Frontend Software Engineer, Backend Software Engineer, Full-Stack Software Engineer, Mobile Software Engineer (Android), Mobile Software Engineer (iOS), Web Developer, Game Developer, Software Engineer, AR/VR Software Engineer
2. "Platform & Infrastructure" — Cloud Engineer, Platform Engineer, Systems Software Engineer, DevSecOps Engineer, Build/Release Engineer, Embedded Software Engineer, Firmware Engineer, Data Platform Engineer, Graphics/Rendering Engineer, Robotics Software Engineer
3. "Engineering Management & Quality" — Engineering Manager, Software Development Manager, Principal Software Engineer, Staff Software Engineer, Quality Engineer (Software), SDET (Software Development Engineer in Test), Test Automation Engineer, Tools Engineer, MLOps Engineer, Solutions Engineer (Pre-Sales), Application Security Engineer

**Education (29 roles → 3 subclusters)**:
1. "Teaching & Academic Programs" — Professor, Teacher, Academic Advisor, Curriculum Developer, Student Success Advisor, Career Services Coordinator, Corporate Trainer
2. "Student Services & Enrollment" — Admissions Counselor, Financial Aid Officer, Enrollment Manager, Enrollment Marketing Manager, International Student Services Coordinator, Registrar, Student Records Coordinator, Bursar, Bursar Operations Analyst, Scheduling Officer
3. "Educational Technology & Administration" — Instructional Designer, Instructional Technologist, eLearning Developer, Learning Experience Designer, Institutional Research Analyst, Registrar Systems Analyst, Assessment & Accreditation Coordinator, Educational Program Coordinator, Department Coordinator, Dean's Office Administrator, School Administrator, Alumni Relations Manager

**Estimated total**: ~96 subclusters (5 intact + ~91 split). Within the 80-120 range.

**Output schema** (`data/taxonomy/clusters.json`):

```json
[
  {
    "cluster_label": "HR Strategy & Business Partnership",
    "category": "Human Resources",
    "roles": ["HR Business Partner", "HR Consultant", "People Analytics Analyst", ...]
  }
]
```

**Acceptance criteria**:
- Every role in `roles.json` appears in exactly one cluster
- No cluster is empty
- Small categories (< 10 roles) have exactly 1 cluster with label matching the category name
- Total cluster count is between 80 and 120
- Cluster labels are descriptive (not numbered)

---

## Subcomponent: Description Generator (`src/descriptions.py`)

### Function: `generate_descriptions(roles: list[dict], config: dict) -> dict`

Call Claude API to generate one-line functional descriptions for all 692 roles and 42 categories.

**Role description prompt**:

```
System: Generate concise functional descriptions for job roles. Each description
should be 10-15 words capturing the core day-to-day responsibility. Be specific
and concrete — focus on what the person actually does, not aspirational language.

User: Generate a one-line functional description for each role in the
"{category}" category. Return valid JSON mapping role name to description.

Roles:
{newline-separated role names}

Example output:
{
  "HR Business Partner": "Strategic HR advisor who aligns people strategy with business objectives",
  "Recruiter": "Sources, screens, and hires candidates to fill open positions"
}
```

**Category description prompt**:

```
System: Generate a brief keyword summary for job categories. List 5-8 key
functional terms that distinguish this category from others. These terms should
help a matching system identify roles belonging to this category.

User: Generate a keyword summary for the "{category}" category, which contains
these roles: {comma-separated role names}

Return a single line of 5-8 comma-separated key terms.
Example: "recruitment, compensation, benefits, employee relations, talent management"
```

**API batching**: Process one category at a time (42 API calls for roles, 42 for categories = 84 total). Each call handles all roles in one category.

**Output schema** (`data/taxonomy/descriptions.json`):

```json
{
  "roles": {
    "HR Business Partner": "Strategic HR advisor who aligns people strategy with business objectives",
    ...
  },
  "categories": {
    "Human Resources": "recruitment, compensation, benefits, employee relations, talent management",
    ...
  }
}
```

**Acceptance criteria**:
- Every role in `roles.json` has a description
- Every category has a keyword summary
- Role descriptions are 10-15 words (tolerance: 8-20 words)
- Category descriptions are 5-8 comma-separated terms
- Output passes JSON schema validation
- Spot-check: 10 random descriptions are functionally accurate (manual verification during implementation)

---

## Subcomponent: Target Builder (`src/targets.py`)

### Function: `build_target_sets(roles: list[dict], clusters: list[dict], descriptions: dict) -> dict[str, list[dict]]`

Construct the 5 granularity-level target sets.

**Construction rules**:

| Granularity | Source | Text field | Count |
|-------------|--------|-----------|-------|
| `role` | `roles.json` | `f"{role}"` | 692 |
| `role_desc` | `roles.json` + `descriptions.json` | `f"{role}: {description}"` | 692 |
| `cluster` | `clusters.json` | `f"{cluster_label}"` | ~96 |
| `category_desc` | categories + `descriptions.json` | `f"{category}: {key_terms}"` | 42 |
| `category` | categories | `f"{category}"` | 42 |

For `role` and `role_desc` targets: each target has `role` and `category` fields.
For `cluster` targets: each target has `roles` (list), `category`, and `cluster_label` fields.
For `category` and `category_desc` targets: each target has `roles` (list) and `category` fields.

**Output**: `data/taxonomy/target_sets.json` — dict keyed by granularity label, each value is a list of target dicts (spec 00 Target schema).

**Acceptance criteria**:
- `role` and `role_desc` sets each contain exactly 692 targets
- `category` and `category_desc` sets each contain exactly 42 targets
- `cluster` set contains between 80 and 120 targets
- Every target has a unique `id` following the `T-<prefix>-NNNN` convention
- Every role appears in at least one target in each granularity level
- `text` field is non-empty for all targets

---

## Orchestrator: `scripts/prep_taxonomy.py`

Sequential pipeline:

```python
roles = parse_taxonomy(config["taxonomy"]["source"])
save_json(roles, "data/taxonomy/roles.json")

clusters = build_clusters(roles)
save_json(clusters, "data/taxonomy/clusters.json")

descriptions = generate_descriptions(roles, config)
save_json(descriptions, "data/taxonomy/descriptions.json")

target_sets = build_target_sets(roles, clusters, descriptions)
save_json(target_sets, "data/taxonomy/target_sets.json")
```

Fail-closed: if any step raises, the script halts.

---

## Testing Strategy (`tests/test_taxonomy.py`)

1. **Test parse_taxonomy**: Feed a small synthetic markdown string (3 categories, 5 roles each). Verify correct count, structure, and edge cases (roles with parenthetical qualifiers, categories with `&`).
2. **Test get_categories**: Verify grouping from parsed roles.
3. **Test build_clusters**: Use the real taxonomy. Verify every role is in exactly one cluster, cluster count is 80-120, small categories are intact.
4. **Test build_target_sets**: Feed mock roles, clusters, descriptions. Verify target counts, ID formats, text fields, and that every role is reachable in every granularity.
5. **No tests for descriptions.py**: API calls are not unit-testable. Validated by spot-check during implementation.
