#!/usr/bin/env python3
"""Test data preparation pipeline: generate -> validate -> split."""

import os
import sys

import yaml

from src.generate_rules import generate_rule_cases
from src.generate_llm import generate_llm_cases
from src.validate import validate_cases, deduplicate_cases, split_dev_test
from src.utils import load_json, save_json


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    roles = load_json("data/taxonomy/roles.json")

    # Generate
    print("Generating rule-based cases...")
    rule_cases = generate_rule_cases(roles, config["experiment"]["seed"])
    print(f"  Rule-based: {len(rule_cases)} cases")

    print("Generating LLM cases...")
    llm_cases = generate_llm_cases(roles, config)
    print(f"  LLM: {len(llm_cases)} cases")

    manual_path = os.path.join(config["test_data"]["output_dir"], "manual.json")
    manual_cases = load_json(manual_path)
    print(f"  Manual: {len(manual_cases)} cases")

    # Filter LLM cases with hallucinated roles and fix categories from taxonomy
    taxonomy_lookup = {r["role"]: r["category"] for r in roles}
    valid_llm = []
    dropped = 0
    cat_fixed = 0
    for case in llm_cases:
        bad_roles = [
            cr["role"] for cr in case["correct_roles"]
            if cr["role"] not in taxonomy_lookup
        ]
        if bad_roles:
            dropped += 1
            print(f"  Dropping LLM case '{case['input_title']}': "
                  f"invalid role(s) {bad_roles}", file=sys.stderr)
        else:
            # Fix categories to match taxonomy (LLM may assign wrong ones)
            for cr in case["correct_roles"]:
                correct_cat = taxonomy_lookup[cr["role"]]
                if cr["category"] != correct_cat:
                    cat_fixed += 1
                    cr["category"] = correct_cat
            valid_llm.append(case)
    if dropped:
        print(f"  Filtered {dropped} LLM cases with invalid roles")
    if cat_fixed:
        print(f"  Fixed {cat_fixed} category assignments from taxonomy")
    llm_cases = valid_llm

    # Combine and assign IDs
    all_cases = rule_cases + llm_cases + manual_cases
    for i, case in enumerate(all_cases):
        case["id"] = f"TC-{i+1:04d}"
    print(f"Total before dedup: {len(all_cases)}")

    # Deduplicate
    all_cases = deduplicate_cases(all_cases)
    print(f"Total after dedup: {len(all_cases)}")

    # Re-assign IDs after dedup
    for i, case in enumerate(all_cases):
        case["id"] = f"TC-{i+1:04d}"

    # Validate
    validate_cases(all_cases, roles)
    print("Validation passed")

    # Save raw
    save_json(all_cases, f"{config['test_data']['output_dir']}/raw_cases.json")

    # Split
    dev, test = split_dev_test(
        all_cases,
        config["test_data"]["dev_size"],
        config["experiment"]["seed"],
    )
    save_json(dev, f"{config['test_data']['output_dir']}/dev.json")
    save_json(test, f"{config['test_data']['output_dir']}/test.json")
    print(f"Split: {len(dev)} dev, {len(test)} test")


if __name__ == "__main__":
    main()
