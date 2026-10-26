from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .adapter import adapt_normalized
from .agents import Classifier, Designer, TaxonomyPrompts
from .codebook import Codebook, load_seed_codebook
from .models import Assignment, PipelineConfig
from .reporting import build_summary, write_distribution, write_framework_crosstab
from .sampling import stratified_pilot
from .storage import atomic_json, atomic_jsonl


def _classify_checkpointed(
    classifier: Classifier,
    records,
    codebook: Codebook,
    checkpoint_path: Path,
    batch_size: int,
) -> list[Assignment]:
    by_id = {item.contract_id: item for item in records}
    completed: dict[str, Assignment] = {}
    if checkpoint_path.is_file():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = Assignment.model_validate_json(line)
            if item.taxonomy_version != codebook.document.version:
                continue
            if item.contract_id in completed:
                raise ValueError("checkpoint contains duplicate contract IDs")
            completed[item.contract_id] = item
        unknown = set(completed) - set(by_id)
        if unknown:
            raise ValueError(
                f"checkpoint contains unknown contract IDs: {sorted(unknown)}"
            )
        for item in completed.values():
            codebook.validate_assignments(
                [item], {item.contract_id: by_id[item.contract_id].contract}
            )
    remaining = [item for item in records if item.contract_id not in completed]
    for offset in range(0, len(remaining), batch_size):
        batch = remaining[offset : offset + batch_size]
        assignments = classifier.classify(batch, codebook.document)
        codebook.validate_assignments(
            assignments, {item.contract_id: item.contract for item in batch}
        )
        completed.update((item.contract_id, item) for item in assignments)
        atomic_jsonl(
            checkpoint_path,
            [
                completed[item.contract_id]
                for item in records
                if item.contract_id in completed
            ],
        )
    values = [completed[item.contract_id] for item in records]
    codebook.validate_assignments(
        values, {item.contract_id: item.contract for item in records}
    )
    return values


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    classifier: Classifier,
    designer: Designer,
    config: PipelineConfig,
    *,
    seed_only: bool = False,
) -> dict[str, object]:
    adapted = adapt_normalized(input_path)
    codebook = load_seed_codebook()
    seed_codebook = codebook
    pilot, sample_manifest = stratified_pilot(
        adapted.contracts,
        fraction=config.pilot_fraction,
        cap=config.designer_sample_cap,
        seed=config.rng_seed,
    )
    prompts = TaxonomyPrompts()
    manifest = {
        "input_path": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "seed_taxonomy_sha256": codebook.hash,
        "classifier_model": config.classifier_model,
        "designer_model": config.designer_model,
        "prompt_version": config.prompt_version,
        "prompt_hashes": prompts.hashes(),
        "parameters": config.model_dump(mode="json"),
        "mode": "seed_only" if seed_only else "pilot_refinement_and_final",
    }
    atomic_json(output_dir / "manifest.json", manifest)
    atomic_json(output_dir / "seed_taxonomy.json", codebook.document)
    atomic_json(output_dir / "pilot_contract_ids.json", sample_manifest)
    atomic_jsonl(output_dir / "quarantine.jsonl", adapted.quarantine)

    pilot_seed_assignments = _classify_checkpointed(
        classifier,
        pilot,
        codebook,
        output_dir / "pilot_seed_checkpoint.jsonl",
        config.classification_batch_size,
    )
    atomic_jsonl(output_dir / "pilot_seed_assignments.jsonl", pilot_seed_assignments)
    proposals = []
    history = []
    rounds = 0
    if not seed_only:
        current = pilot_seed_assignments
        for round_number in range(1, config.max_refinement_rounds + 1):
            failures = [
                item for item in current if item.status in {"unclassified", "ambiguous"}
            ]
            if not failures:
                break
            by_id = {item.contract_id: item for item in pilot}
            classified = [item for item in current if item.status == "classified"]
            classified_records = [by_id[item.contract_id] for item in classified]
            if config.designer_assigned_sample_cap:
                representative_records, _ = stratified_pilot(
                    classified_records,
                    fraction=1,
                    cap=config.designer_assigned_sample_cap,
                    seed=config.rng_seed + round_number,
                )
            else:
                representative_records = []
            representative_ids = {item.contract_id for item in representative_records}
            representative_assignments = [
                item for item in classified if item.contract_id in representative_ids
            ]
            round_records = [by_id[item.contract_id] for item in failures]
            round_records.extend(representative_records)
            proposed = designer.propose(
                failures,
                representative_assignments,
                round_records,
                codebook.document,
                round_number,
            )
            proposed = proposed[: config.max_auto_extensions_per_round]
            reviewed = [
                codebook.review_proposal(
                    item,
                    {
                        record.contract_id: record.discussion_id
                        for record in adapted.contracts
                    },
                    min_distinct_discussions=config.min_distinct_discussions_for_automatic_accept,
                )
                for item in proposed
            ]
            if not config.automatic_extension_acceptance:
                reviewed = [
                    (
                        item.model_copy(
                            update={
                                "status": "human_review",
                                "decision_reason": "Automatic acceptance disabled for this run",
                            }
                        )
                        if item.status == "accepted"
                        else item
                    )
                    for item in reviewed
                ]
            proposals.extend(reviewed)
            accepted = [item for item in reviewed if item.status == "accepted"]
            if not accepted:
                break
            before = codebook.hash
            codebook = codebook.with_extensions(
                accepted, f"mint_extended_v{round_number}"
            )
            rounds = round_number
            history.append(
                {
                    "round": round_number,
                    "before_sha256": before,
                    "after_sha256": codebook.hash,
                    "accepted_proposal_ids": [item.proposal_id for item in accepted],
                }
            )
            current = _classify_checkpointed(
                classifier,
                pilot,
                codebook,
                output_dir / f"refinement_round_{round_number}_checkpoint.jsonl",
                config.classification_batch_size,
            )
    atomic_jsonl(output_dir / "extension_proposals.jsonl", proposals)
    atomic_jsonl(output_dir / "taxonomy_history.jsonl", history)
    codebook = codebook.freeze()
    taxonomy_name = f"extended_taxonomy_v{max(1, rounds)}.json"
    atomic_json(output_dir / taxonomy_name, codebook.document)

    seed_assignments = _classify_checkpointed(
        classifier,
        adapted.contracts,
        seed_codebook,
        output_dir / "seed_checkpoint.jsonl",
        config.classification_batch_size,
    )
    atomic_jsonl(output_dir / "seed_assignments.jsonl", seed_assignments)
    if seed_only:
        final_assignments = seed_assignments
    else:
        final_assignments = _classify_checkpointed(
            classifier,
            adapted.contracts,
            codebook,
            output_dir / "final_checkpoint.jsonl",
            config.classification_batch_size,
        )
    atomic_jsonl(output_dir / "final_assignments.jsonl", final_assignments)
    unresolved = [item for item in final_assignments if item.status != "classified"]
    atomic_jsonl(output_dir / "unresolved_assignments.jsonl", unresolved)
    proposal_counts = dict(sorted(Counter(item.status for item in proposals).items()))
    summary = build_summary(
        adapted,
        seed_assignments,
        final_assignments,
        codebook,
        rounds=rounds,
        proposal_counts=proposal_counts,
    )
    atomic_json(output_dir / "summary.json", summary)
    write_distribution(
        output_dir / "contract_type_distribution.csv", final_assignments, codebook
    )
    write_framework_crosstab(
        output_dir / "framework_crosstab.csv",
        final_assignments,
        {item.contract_id: item.framework for item in adapted.contracts},
    )
    return summary


def regenerate_reports(input_path: Path, output_dir: Path) -> dict[str, object]:
    from .models import TaxonomyDocument

    adapted = adapt_normalized(input_path)
    taxonomy_paths = sorted(output_dir.glob("extended_taxonomy_v*.json"))
    if not taxonomy_paths:
        raise ValueError("no frozen extended_taxonomy_vN.json artifact found")
    codebook = Codebook(
        TaxonomyDocument.model_validate_json(
            taxonomy_paths[-1].read_text(encoding="utf-8")
        )
    )

    def read_assignments(name: str) -> list[Assignment]:
        return [
            Assignment.model_validate(json.loads(line))
            for line in (output_dir / name).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    seed_values = read_assignments("seed_assignments.jsonl")
    final_values = read_assignments("final_assignments.jsonl")
    codebook.validate_assignments(
        final_values, {item.contract_id: item.contract for item in adapted.contracts}
    )
    proposal_path = output_dir / "extension_proposals.jsonl"
    proposals = [
        json.loads(line)
        for line in proposal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    history_path = output_dir / "taxonomy_history.jsonl"
    rounds = sum(
        bool(line.strip())
        for line in history_path.read_text(encoding="utf-8").splitlines()
    )
    summary = build_summary(
        adapted,
        seed_values,
        final_values,
        codebook,
        rounds=rounds,
        proposal_counts=dict(
            sorted(Counter(item["status"] for item in proposals).items())
        ),
    )
    atomic_json(output_dir / "summary.json", summary)
    write_distribution(
        output_dir / "contract_type_distribution.csv", final_values, codebook
    )
    write_framework_crosstab(
        output_dir / "framework_crosstab.csv",
        final_values,
        {item.contract_id: item.framework for item in adapted.contracts},
    )
    return summary
