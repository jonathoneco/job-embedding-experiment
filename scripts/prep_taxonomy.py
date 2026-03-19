#!/usr/bin/env python3
"""Taxonomy preparation pipeline: parse -> cluster -> describe -> build targets."""

import yaml

from src.taxonomy import parse_taxonomy, get_categories
from src.clusters import build_clusters
from src.descriptions import generate_descriptions
from src.targets import build_target_sets
from src.utils import save_json


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    roles = parse_taxonomy(config["taxonomy"]["source"])
    save_json(roles, f"{config['taxonomy']['output_dir']}/roles.json")
    print(f"Parsed {len(roles)} roles")

    clusters = build_clusters(roles)
    save_json(clusters, f"{config['taxonomy']['output_dir']}/clusters.json")
    print(f"Built {len(clusters)} clusters")

    descriptions = generate_descriptions(roles, config)
    save_json(descriptions, f"{config['taxonomy']['output_dir']}/descriptions.json")
    print(f"Generated descriptions for {len(descriptions['roles'])} roles")

    target_sets = build_target_sets(roles, clusters, descriptions)
    save_json(target_sets, f"{config['taxonomy']['output_dir']}/target_sets.json")
    for gran, targets in target_sets.items():
        print(f"  {gran}: {len(targets)} targets")


if __name__ == "__main__":
    main()
