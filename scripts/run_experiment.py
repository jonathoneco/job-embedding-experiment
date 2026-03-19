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
            if gran in ("role", "role_desc"):
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
