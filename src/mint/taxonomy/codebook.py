from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .models import Assignment, ExtensionProposal, TaxonomyDocument, TaxonomyNode

SEED_PATH = Path(__file__).with_name("seed_taxonomy.json")


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class Codebook:
    def __init__(self, document: TaxonomyDocument) -> None:
        self.document = document
        self._validate_tree()

    @property
    def nodes(self) -> dict[str, TaxonomyNode]:
        return {node.id: node for node in self.document.nodes}

    @property
    def leaf_ids(self) -> set[str]:
        parents = {node.parent_id for node in self.document.nodes if node.parent_id}
        return {node.id for node in self.document.nodes if node.id not in parents}

    @property
    def hash(self) -> str:
        return content_hash(self.document.model_dump(mode="json"))

    def _validate_tree(self) -> None:
        nodes = self.nodes
        if len(nodes) != len(self.document.nodes):
            raise ValueError("taxonomy contains duplicate node IDs")
        for node in self.document.nodes:
            if node.parent_id is not None and node.parent_id not in nodes:
                raise ValueError(f"unknown parent {node.parent_id!r} for {node.id!r}")
            if node.taxonomy_origin == "emse_2023" and not node.immutable:
                raise ValueError(f"seed node must be immutable: {node.id}")
        for node_id in nodes:
            seen: set[str] = set()
            current = nodes[node_id]
            while current.parent_id:
                if current.id in seen:
                    raise ValueError(f"taxonomy cycle involving {node_id}")
                seen.add(current.id)
                current = nodes[current.parent_id]

    def validate_assignments(
        self, assignments: Iterable[Assignment], records: dict[str, str]
    ) -> list[Assignment]:
        values = list(assignments)
        ids = [item.contract_id for item in values]
        if len(ids) != len(set(ids)):
            raise ValueError("classifier returned duplicate contract IDs")
        missing = set(records) - set(ids)
        unknown = set(ids) - set(records)
        if missing or unknown:
            raise ValueError(
                f"assignment coverage mismatch; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        for item in values:
            if item.taxonomy_version != self.document.version:
                raise ValueError(
                    f"assignment uses wrong taxonomy version: {item.contract_id}"
                )
            selected = (
                ([item.primary_leaf_id] if item.primary_leaf_id else [])
                + list(item.candidate_alternatives)
                + list(item.considered_alternatives)
            )
            invalid = set(selected) - self.leaf_ids
            if invalid:
                raise ValueError(
                    f"assignment references non-leaf IDs: {sorted(invalid)}"
                )
            if (
                item.decisive_span
                and item.decisive_span not in records[item.contract_id]
            ):
                raise ValueError(
                    f"decisive span is not verbatim for {item.contract_id}: "
                    f"{item.decisive_span!r}"
                )
        return values

    def review_proposal(
        self,
        proposal: ExtensionProposal,
        contract_discussions: dict[str, str],
        *,
        min_distinct_discussions: int = 2,
    ) -> ExtensionProposal:
        proposal = proposal.model_copy(deep=True)
        unknown = set(proposal.triggering_contract_ids) - set(contract_discussions)
        if unknown:
            proposal.status = "rejected"
            proposal.decision_reason = (
                f"Unknown triggering contract IDs: {sorted(unknown)}"
            )
            return proposal
        target = self.nodes.get(proposal.target_parent_id or "")
        if proposal.action != "add_category":
            touched = self.nodes.get(proposal.proposed_id or "")
            if touched and touched.immutable:
                proposal.status = "rejected"
                proposal.decision_reason = "EMSE seed nodes are immutable"
                return proposal
            proposal.status = "human_review"
            proposal.decision_reason = "Non-additive extension revisions require review"
            return proposal
        required = (
            proposal.proposed_id,
            proposal.proposed_name,
            proposal.definition,
            proposal.inclusion_criteria,
            proposal.exclusion_criteria,
            proposal.distinct_from_siblings,
        )
        if not all(required):
            proposal.status = "rejected"
            proposal.decision_reason = (
                "Add-category proposal is structurally incomplete"
            )
            return proposal
        if proposal.target_parent_id is not None and target is None:
            proposal.status = "rejected"
            proposal.decision_reason = "Target parent does not exist"
            return proposal
        if proposal.proposed_id in self.nodes or not proposal.proposed_id.startswith(
            "MINT."
        ):
            proposal.status = "rejected"
            proposal.decision_reason = "Extension ID must be new and start with MINT."
            return proposal
        normalized = " ".join((proposal.definition or "").lower().split())
        if any(
            normalized == " ".join(node.definition.lower().split())
            for node in self.document.nodes
        ):
            proposal.status = "rejected"
            proposal.decision_reason = "Proposal duplicates an existing definition"
            return proposal
        support = {
            contract_discussions[item] for item in proposal.triggering_contract_ids
        }
        if len(support) < min_distinct_discussions:
            proposal.status = "provisional"
            proposal.decision_reason = (
                "Meaningful but below automatic support threshold"
            )
            return proposal
        review_note = (
            "; designer recommended human review"
            if proposal.status == "human_review"
            else ""
        )
        proposal.status = "accepted"
        proposal.decision_reason = (
            "Schema-valid addition with sufficient discussion support" + review_note
        )
        return proposal

    def with_extensions(
        self, proposals: Iterable[ExtensionProposal], version: str
    ) -> "Codebook":
        if self.document.frozen:
            raise ValueError("frozen taxonomies cannot accept ad-hoc extensions")
        document = copy.deepcopy(self.document)
        document.parent_version = self.document.version
        document.version = version
        for proposal in proposals:
            if proposal.status != "accepted" or proposal.action != "add_category":
                continue
            document.nodes.append(
                TaxonomyNode(
                    id=proposal.proposed_id or "",
                    name=proposal.proposed_name or "",
                    source_label=proposal.proposed_name or "",
                    parent_id=proposal.target_parent_id,
                    definition=proposal.definition or "",
                    inclusion_criteria=proposal.inclusion_criteria,
                    exclusion_criteria=proposal.exclusion_criteria,
                    taxonomy_origin="mint_extension",
                    citations=[
                        "MINT extension proposal " + proposal.proposal_id,
                        *proposal.triggering_contract_ids,
                    ],
                    immutable=False,
                )
            )
        return Codebook(document)

    def freeze(self) -> "Codebook":
        document = copy.deepcopy(self.document)
        document.frozen = True
        return Codebook(document)


def load_seed_codebook(path: Path = SEED_PATH) -> Codebook:
    return Codebook(
        TaxonomyDocument.model_validate_json(path.read_text(encoding="utf-8"))
    )
