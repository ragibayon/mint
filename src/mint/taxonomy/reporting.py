from __future__ import annotations

import csv
import io
import statistics
from collections import Counter, defaultdict

from .codebook import Codebook
from .models import AdapterResult, Assignment
from .storage import atomic_text


def top_level_id(leaf_id: str, codebook: Codebook) -> str:
    node = codebook.nodes[leaf_id]
    while node.parent_id is not None:
        node = codebook.nodes[node.parent_id]
    return node.id


def build_summary(
    adapted: AdapterResult,
    seed_assignments: list[Assignment],
    final_assignments: list[Assignment],
    codebook: Codebook,
    *,
    rounds: int,
    proposal_counts: dict[str, int],
) -> dict[str, object]:
    seed = Counter(item.status for item in seed_assignments)
    final = Counter(item.status for item in final_assignments)
    denominator = len(final_assignments)
    seed_denominator = len(seed_assignments)
    seed_coverage = (
        seed.get("classified", 0) / seed_denominator if seed_denominator else None
    )
    extended_coverage = (
        final.get("classified", 0) / denominator if denominator else None
    )
    discussion_by_id = {
        item.contract_id: item.discussion_id for item in adapted.contracts
    }
    contracts_per_discussion = Counter(discussion_by_id.values())
    count_values = sorted(contracts_per_discussion.values())
    quartiles = (
        statistics.quantiles(count_values, n=4, method="inclusive")
        if len(count_values) >= 2
        else ([count_values[0]] * 3 if count_values else [0, 0, 0])
    )
    discussions_by_category: dict[str, set[str]] = defaultdict(set)
    for item in final_assignments:
        if item.primary_leaf_id:
            discussions_by_category[item.primary_leaf_id].add(
                discussion_by_id[item.contract_id]
            )
    return {
        "population_note": "LLM-mined normalized candidate contracts; not gold-standard validity",
        "denominators": {
            "eligible_normalized_contracts": denominator,
            "seed_classified_contracts": seed_denominator,
            "contract_bearing_discussions": adapted.contract_bearing_discussions,
        },
        "corpus_funnel": {
            "total_discussions": adapted.total_discussions,
            "normalizer_status_counts": adapted.discussion_status_counts,
            "contract_bearing_discussions": adapted.contract_bearing_discussions,
            "eligible_normalized_contracts": denominator,
            "quarantined_records": len(adapted.quarantine),
            "final_status_counts": dict(sorted(final.items())),
        },
        "transferability": {
            "seed_classified": seed.get("classified", 0),
            "seed_denominator": seed_denominator,
            "seed_coverage": seed_coverage,
            "extended_classified": final.get("classified", 0),
            "extended_denominator": denominator,
            "extended_coverage": extended_coverage,
            "improvement_percentage_points": (
                (extended_coverage - seed_coverage) * 100
                if seed_coverage is not None and extended_coverage is not None
                else None
            ),
        },
        "taxonomy": {
            "version": codebook.document.version,
            "sha256": codebook.hash,
            "frozen": codebook.document.frozen,
            "seed_nodes": sum(
                node.taxonomy_origin == "emse_2023" for node in codebook.document.nodes
            ),
            "extension_nodes": sum(
                node.taxonomy_origin == "mint_extension"
                for node in codebook.document.nodes
            ),
        },
        "refinement": {"rounds": rounds, "proposal_counts": proposal_counts},
        "contracts_per_discussion": {
            "mean": statistics.fmean(count_values) if count_values else None,
            "median": statistics.median(count_values) if count_values else None,
            "q1": quartiles[0] if count_values else None,
            "q3": quartiles[2] if count_values else None,
        },
        "discussion_level_prevalence": {
            category: {
                "discussions": len(discussions),
                "denominator": adapted.contract_bearing_discussions,
                "proportion": (
                    len(discussions) / adapted.contract_bearing_discussions
                    if adapted.contract_bearing_discussions
                    else None
                ),
            }
            for category, discussions in sorted(discussions_by_category.items())
        },
    }


def write_distribution(path, assignments: list[Assignment], codebook: Codebook) -> None:
    counts = Counter(
        item.primary_leaf_id for item in assignments if item.primary_leaf_id
    )
    total = len(assignments)
    nodes = codebook.nodes
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=[
            "level",
            "top_level",
            "leaf_id",
            "name",
            "origin",
            "n",
            "percent_of_eligible",
        ],
    )
    writer.writeheader()
    top_counts = Counter(
        top_level_id(item.primary_leaf_id, codebook)
        for item in assignments
        if item.primary_leaf_id
    )
    for top_level, count in sorted(top_counts.items()):
        writer.writerow(
            {
                "level": "top_level",
                "top_level": top_level,
                "leaf_id": "",
                "name": nodes[top_level].name,
                "origin": nodes[top_level].taxonomy_origin,
                "n": count,
                "percent_of_eligible": count / total * 100 if total else 0,
            }
        )
    for leaf_id, count in sorted(counts.items()):
        node = nodes[leaf_id]
        writer.writerow(
            {
                "level": "leaf",
                "top_level": top_level_id(leaf_id, codebook),
                "leaf_id": leaf_id,
                "name": node.name,
                "origin": node.taxonomy_origin,
                "n": count,
                "percent_of_eligible": count / total * 100 if total else 0,
            }
        )
    atomic_text(path, stream.getvalue())


def write_framework_crosstab(
    path, assignments: list[Assignment], framework_by_id: dict[str, str | None]
) -> None:
    counts: Counter[tuple[str, str]] = Counter()
    totals: Counter[str] = Counter()
    for item in assignments:
        framework = framework_by_id.get(item.contract_id) or "unknown"
        totals[framework] += 1
        counts[(framework, item.primary_leaf_id or item.status)] += 1
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=["framework", "category", "n", "percent_within_framework"]
    )
    writer.writeheader()
    for (framework, category), count in sorted(counts.items()):
        writer.writerow(
            {
                "framework": framework,
                "category": category,
                "n": count,
                "percent_within_framework": count / totals[framework] * 100,
            }
        )
    atomic_text(path, stream.getvalue())
