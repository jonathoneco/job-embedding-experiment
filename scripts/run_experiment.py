#!/usr/bin/env python3
"""Main experiment orchestrator: models -> baselines -> evaluate -> report."""

import os
import yaml

from src.embed import run_embedding_model
from src.baselines import run_all_baselines
from src.evaluate import evaluate_all
from src.statistics import run_statistical_tests
from src.report import generate_report, write_report
from src.utils import load_json, save_json


def validate_target_sets(target_sets):
    """Validate target_sets schema after JSON load."""
    for gran, targets in target_sets.items():
        for i, t in enumerate(targets):
            for key in ("id", "text", "granularity"):
                if key not in t:
                    raise ValueError(f"target_sets['{gran}'][{i}] missing required key '{key}'")
            if gran in ("role", "role_desc", "role_augmented", "curated_role"):
                if "role" not in t:
                    raise ValueError(f"target_sets['{gran}'][{i}] missing 'role' key")
            else:
                if "roles" not in t:
                    raise ValueError(f"target_sets['{gran}'][{i}] missing 'roles' key")


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Ensure output directories exist
    os.makedirs(config["report"]["metrics_dir"], exist_ok=True)
    os.makedirs(config["report"]["figures_dir"], exist_ok=True)

    # Load data
    target_sets = load_json("data/taxonomy/target_sets.json")
    validate_target_sets(target_sets)
    test_cases = load_json("data/test-cases/test.json")
    dev_cases = load_json("data/test-cases/dev.json")

    # Run embedding models
    all_rankings = []
    for model_config in config["models"]:
        print(f"Running model: {model_config['label']}...")
        rankings = run_embedding_model(model_config, target_sets, test_cases, config)
        all_rankings.extend(rankings)
        print(f"  Produced {len(rankings)} rankings")

    # Run baselines
    print("Running baselines...")
    baseline_rankings = run_all_baselines(target_sets, test_cases)
    all_rankings.extend(baseline_rankings)
    print(f"  Produced {len(baseline_rankings)} rankings")

    # Save raw rankings
    save_json(all_rankings, f"{config['report']['metrics_dir']}/all_rankings.json")

    # Evaluate
    print("Computing metrics...")
    metrics = evaluate_all(all_rankings, test_cases, target_sets, config)
    save_json(metrics, f"{config['report']['metrics_dir']}/all_metrics.json")

    # Statistical tests
    print("Running statistical tests...")
    significance = run_statistical_tests(all_rankings, test_cases, target_sets, config)
    save_json(significance, f"{config['report']['metrics_dir']}/significance.json")

    # Curated target library evaluation
    curated_config = config.get("curated", {})
    if curated_config.get("enabled", False):
        print("\n--- Curated Target Library Evaluation ---")

        # Load taxonomy data needed for filtering
        roles = load_json("data/taxonomy/roles.json")
        clusters = load_json("data/taxonomy/clusters.json")

        # Build curated targets
        from src.curated_targets import load_curated_roles, build_curated_target_sets, filter_covered_test_cases
        curated_roles = load_curated_roles(curated_config["roles_file"], roles)
        curated_target_sets = build_curated_target_sets(curated_roles, roles, clusters)
        validate_target_sets(curated_target_sets)

        # Filter test cases for coverage
        curated_test_cases, coverage_report = filter_covered_test_cases(test_cases, curated_roles)
        print(f"  Coverage: {coverage_report['covered']}/{coverage_report['total']} test cases "
              f"({coverage_report['coverage_pct']}%)")
        save_json(coverage_report, f"{config['report']['metrics_dir']}/curated_coverage.json")

        # Print curated target set sizes
        for gran, targets in curated_target_sets.items():
            print(f"  {gran}: {len(targets)} targets")

        if not curated_test_cases:
            print("  WARNING: No test cases covered by curated set. Skipping curated evaluation.")
        else:
            # Run embedding models against curated targets
            curated_rankings = []
            for model_config_entry in config["models"]:
                print(f"  Running model: {model_config_entry['label']} (curated)...")
                rankings = run_embedding_model(model_config_entry, curated_target_sets, curated_test_cases, config)
                curated_rankings.extend(rankings)

            # Run baselines against curated targets
            # run_all_baselines hardcodes granularity names, so call individual functions
            print("  Running baselines (curated)...")
            from src.baselines import run_tfidf, run_fuzzy, run_bm25
            for gran in ["curated_role", "curated_cluster", "curated_category"]:
                targets = curated_target_sets[gran]
                curated_rankings.extend(run_tfidf(targets, curated_test_cases, gran))
                curated_rankings.extend(run_fuzzy(targets, curated_test_cases, gran))
                curated_rankings.extend(run_bm25(targets, curated_test_cases, gran))

            # Save curated rankings
            save_json(curated_rankings, f"{config['report']['metrics_dir']}/curated_rankings.json")

            # Evaluate
            print("  Computing curated metrics...")
            curated_metrics = evaluate_all(curated_rankings, curated_test_cases, curated_target_sets, config)
            save_json(curated_metrics, f"{config['report']['metrics_dir']}/curated_metrics.json")

            # Summary
            if curated_metrics:
                best_curated = max(curated_metrics, key=lambda m: m["metrics"]["mrr"])
                print(f"\n  Best curated config: {best_curated['method']} @ {best_curated['granularity']}")
                print(f"    MRR: {best_curated['metrics']['mrr']:.3f}")
                print(f"    Top-1: {best_curated['metrics']['top1']:.3f}")
                print(f"    Top-3: {best_curated['metrics']['top3']:.3f}")
                print(f"    Top-5: {best_curated['metrics']['top5']:.3f}")

    # Dev set sanity check (first model only — sanity check, not full evaluation)
    print("Running dev set sanity check...")
    dev_model = config["models"][0]
    dev_rankings = run_embedding_model(dev_model, target_sets, dev_cases, config)
    dev_metrics = evaluate_all(dev_rankings, dev_cases, target_sets, config)
    save_json(dev_metrics, f"{config['report']['metrics_dir']}/dev_metrics.json")

    # Generate report
    print("Generating report...")
    report = generate_report(metrics, significance, test_cases, all_rankings, target_sets, config)
    write_report(report, config["report"]["output"])
    print(f"Report written to {config['report']['output']}")

    # Summary
    best = max(metrics, key=lambda m: m["metrics"]["mrr"])
    print(f"\nBest config: {best['method']} @ {best['granularity']}")
    print(f"  MRR: {best['metrics']['mrr']:.3f}")
    print(f"  Top-1: {best['metrics']['top1']:.3f}")
    print(f"  Top-3: {best['metrics']['top3']:.3f}")
    print(f"  Top-5: {best['metrics']['top5']:.3f}")


if __name__ == "__main__":
    main()
