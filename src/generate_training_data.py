"""Training data generator for fine-tuning embedding models.

Produces contrastive triplets (anchor/positive/negative) and a TSDAE
corpus from the job taxonomy, using rule-based variant generation and
TF-IDF hard negative mining.
"""

import json
import os
import random

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.generate_rules import (
    _apply_abbreviation,
    _apply_level_prefix,
    _apply_level_suffix,
    _apply_minor_rewording,
    _apply_word_reorder,
)
from src.taxonomy import parse_taxonomy

_TRANSFORMS = [
    _apply_level_prefix,
    _apply_level_suffix,
    _apply_word_reorder,
    _apply_abbreviation,
    _apply_minor_rewording,
]


def generate_variants(roles: list[dict], seed: int) -> dict[str, list[str]]:
    """Generate 7-8 title variants per role using rule-based transforms.

    Uses seed offset ``seed + 10000`` to avoid overlap with test data.

    Args:
        roles: List of {"role": str, "category": str} dicts.
        seed: Base random seed.

    Returns:
        Dict mapping role name to list of variant strings.
    """
    rng = random.Random(seed + 10000)
    variants: dict[str, list[str]] = {}

    for role in roles:
        name = role["role"]
        seen: set[str] = {name}
        role_variants: list[str] = []

        # Pass 1: apply all 5 transforms once
        for fn in _TRANSFORMS:
            v = fn(name, rng)
            if v not in seen:
                seen.add(v)
                role_variants.append(v)

        # Pass 2: reapply transforms with different random states to reach 7-8
        attempts = 0
        while len(role_variants) < 7 and attempts < 30:
            fn = rng.choice(_TRANSFORMS)
            v = fn(name, rng)
            if v not in seen:
                seen.add(v)
                role_variants.append(v)
            attempts += 1

        variants[name] = role_variants

    return variants


def mine_hard_negatives(roles: list[dict]) -> dict[str, list[str]]:
    """Find top-5 hard negatives per role using TF-IDF similarity.

    Hard negatives are the most similar roles from a *different* category,
    making them challenging negative examples for contrastive learning.

    Args:
        roles: List of {"role": str, "category": str} dicts.

    Returns:
        Dict mapping role name to list of hard negative role names.
    """
    role_names = [r["role"] for r in roles]
    categories = [r["category"] for r in roles]

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    tfidf = vectorizer.fit_transform(role_names)
    sim_matrix = cosine_similarity(tfidf)

    hard_negatives: dict[str, list[str]] = {}
    for i, role in enumerate(roles):
        scores = sim_matrix[i].copy()
        # Zero out same-category roles
        for j in range(len(roles)):
            if categories[j] == categories[i]:
                scores[j] = -1.0
        top_indices = scores.argsort()[::-1][:5]
        hard_negatives[role["role"]] = [role_names[j] for j in top_indices]

    return hard_negatives


def generate_contrastive_pairs(
    roles: list[dict], seed: int, output_path: str,
) -> int:
    """Generate contrastive triplets and write to JSONL.

    Each triplet contains an anchor (canonical role), a positive (variant),
    and a negative (hard negative from a different category).

    Args:
        roles: List of {"role": str, "category": str} dicts.
        seed: Random seed for deterministic generation.
        output_path: Path to write JSONL output.

    Returns:
        Number of triplets generated.
    """
    rng = random.Random(seed)
    variants = generate_variants(roles, seed)
    hard_negs = mine_hard_negatives(roles)

    triplets: list[dict] = []
    for role in roles:
        name = role["role"]
        role_variants = variants.get(name, [])
        negatives = hard_negs.get(name, [])
        if not negatives:
            continue
        for v in role_variants:
            neg = rng.choice(negatives)
            triplets.append({
                "anchor": name,
                "positive": v,
                "negative": neg,
            })

    rng.shuffle(triplets)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for t in triplets:
            f.write(json.dumps(t) + "\n")

    return len(triplets)


def generate_tsdae_corpus(
    roles: list[dict], variants: dict[str, list[str]], output_path: str,
) -> int:
    """Write a deduplicated corpus of canonical roles + variants.

    Args:
        roles: List of {"role": str, "category": str} dicts.
        variants: Dict mapping role name to list of variant strings.
        output_path: Path to write one-title-per-line output.

    Returns:
        Number of corpus entries written.
    """
    seen: set[str] = set()
    entries: list[str] = []

    for role in roles:
        name = role["role"]
        if name not in seen:
            seen.add(name)
            entries.append(name)

    for name, var_list in variants.items():
        for v in var_list:
            if v not in seen:
                seen.add(v)
                entries.append(v)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for entry in entries:
            f.write(entry + "\n")

    return len(entries)


def main() -> None:
    """CLI entry point: generate training data from taxonomy."""
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    seed = config["experiment"]["seed"]
    taxonomy_path = config["taxonomy"]["source"]

    roles = parse_taxonomy(taxonomy_path)
    os.makedirs("data/training", exist_ok=True)

    pairs_path = "data/training/pairs.jsonl"
    corpus_path = "data/training/corpus.txt"

    n_pairs = generate_contrastive_pairs(roles, seed, pairs_path)

    variants = generate_variants(roles, seed)
    n_corpus = generate_tsdae_corpus(roles, variants, corpus_path)

    total_variants = sum(len(v) for v in variants.values())
    print(f"Variants: {total_variants} across {len(variants)} roles")
    print(f"Contrastive pairs: {n_pairs} -> {pairs_path}")
    print(f"TSDAE corpus: {n_corpus} entries -> {corpus_path}")


if __name__ == "__main__":
    main()
