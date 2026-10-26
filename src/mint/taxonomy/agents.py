from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Protocol

import requests

from mint.llm.prompt_renderer import render_prompt

from .models import (
    Assignment,
    ClassificationBatch,
    ContractRecord,
    ExtensionProposal,
    ProposalBatch,
    TaxonomyDocument,
)

PROMPT_VERSION = "refinement"


class Classifier(Protocol):
    def classify(
        self, records: list[ContractRecord], taxonomy: TaxonomyDocument
    ) -> list[Assignment]: ...


class Designer(Protocol):
    def propose(
        self,
        failures: list[Assignment],
        representative_assignments: list[Assignment],
        records: list[ContractRecord],
        taxonomy: TaxonomyDocument,
        round_number: int,
    ) -> list[ExtensionProposal]: ...


class TaxonomyPrompts:
    def __init__(self, prompts_dir: Path | None = None) -> None:
        self.prompts_dir = (
            prompts_dir or Path(__file__).with_name("prompts") / PROMPT_VERSION
        )

    def hashes(self) -> dict[str, str]:
        result = {}
        for name in (
            "classifier_system.j2",
            "classifier_user.j2",
            "designer_system.j2",
            "designer_user.j2",
        ):
            result[name.replace(".j2", "_sha256")] = hashlib.sha256(
                (self.prompts_dir / name).read_bytes()
            ).hexdigest()
        return result

    def classifier(
        self, records: list[ContractRecord], taxonomy: TaxonomyDocument
    ) -> tuple[str, str]:
        return (
            render_prompt(self.prompts_dir, "classifier_system.j2"),
            render_prompt(
                self.prompts_dir,
                "classifier_user.j2",
                taxonomy_json=json.dumps(
                    taxonomy.model_dump(mode="json"), ensure_ascii=False, indent=2
                ),
                contracts_json=json.dumps(
                    [item.model_dump(mode="json") for item in records],
                    ensure_ascii=False,
                    indent=2,
                ),
            ),
        )

    def designer(
        self,
        failures: list[Assignment],
        representative_assignments: list[Assignment],
        records: list[ContractRecord],
        taxonomy: TaxonomyDocument,
        round_number: int,
    ) -> tuple[str, str]:
        return (
            render_prompt(self.prompts_dir, "designer_system.j2"),
            render_prompt(
                self.prompts_dir,
                "designer_user.j2",
                round_number=round_number,
                taxonomy_json=json.dumps(
                    taxonomy.model_dump(mode="json"), ensure_ascii=False, indent=2
                ),
                failures_json=json.dumps(
                    [item.model_dump(mode="json") for item in failures],
                    ensure_ascii=False,
                    indent=2,
                ),
                representative_assignments_json=json.dumps(
                    [
                        item.model_dump(mode="json")
                        for item in representative_assignments
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                contracts_json=json.dumps(
                    [item.model_dump(mode="json") for item in records],
                    ensure_ascii=False,
                    indent=2,
                ),
            ),
        )


class FixtureAgent:
    """Model-free deterministic backend for tests and dry-run smoke checks."""

    def classify(
        self, records: list[ContractRecord], taxonomy: TaxonomyDocument
    ) -> list[Assignment]:
        leaf_ids = {
            node.id
            for node in taxonomy.nodes
            if node.id not in {other.parent_id for other in taxonomy.nodes}
        }
        results = []
        for record in records:
            text = record.contract.lower()
            leaf: str | None = None
            status = "classified"
            alternatives: list[str] = []
            if "[insufficient]" in text:
                status = "insufficient_information"
            elif "[nonfit]" in text:
                status = "unclassified"
            elif "[ambiguous]" in text:
                status = "ambiguous"
                alternatives = ["SAM.DT.PT", "SAM.BET.IC1"]
            else:
                markers = {
                    "[pt]": "SAM.DT.PT",
                    "[bit]": "SAM.DT.BIT",
                    "[rt]": "SAM.DT.RT",
                    "[mt]": "SAM.DT.MT",
                    "[ic1]": "SAM.BET.IC1",
                    "[ic2]": "SAM.BET.IC2",
                    "[g]": "AMO.G",
                    "[f]": "AMO.F",
                    "[sai]": "H.SAI",
                    "[sl-sam]": "H.SL.SAM",
                    "[sl-amo]": "H.SL.AMO",
                    "[sl-mixed]": "H.SL.MIXED",
                }
                leaf = next(
                    (value for marker, value in markers.items() if marker in text), None
                )
                if leaf is None or leaf not in leaf_ids:
                    status = "unclassified"
            results.append(
                Assignment(
                    contract_id=record.contract_id,
                    taxonomy_version=taxonomy.version,
                    status=status,
                    primary_leaf_id=leaf if status == "classified" else None,
                    reason="Deterministic fixture classification.",
                    no_quote_reason="Fixture backend uses explicit synthetic markers.",
                    candidate_alternatives=alternatives,
                )
            )
        return results

    def propose(
        self,
        failures: list[Assignment],
        representative_assignments: list[Assignment],
        records: list[ContractRecord],
        taxonomy: TaxonomyDocument,
        round_number: int,
    ) -> list[ExtensionProposal]:
        del failures, representative_assignments, records, taxonomy, round_number
        return []


def parse_classification_batch(value: object) -> ClassificationBatch:
    return ClassificationBatch.model_validate(value)


def parse_proposal_batch(value: object) -> ProposalBatch:
    return ProposalBatch.model_validate(value)


class OpenRouterTaxonomyAgent:
    """Explicit opt-in live backend; construction itself performs no requests."""

    def __init__(
        self,
        *,
        classifier_model: str,
        designer_model: str,
        timeout_seconds: float = 300,
    ) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self.api_key = api_key
        self.classifier_model = classifier_model
        self.designer_model = designer_model
        self.timeout_seconds = timeout_seconds
        self.prompts = TaxonomyPrompts()

    def _invoke(self, model: str, system: str, user: str, schema) -> object:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "strict": True,
                        "schema": schema.model_json_schema(),
                    },
                },
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") for item in content if item.get("type") == "text"
            )
        return json.loads(content)

    def classify(
        self, records: list[ContractRecord], taxonomy: TaxonomyDocument
    ) -> list[Assignment]:
        if not records:
            return []
        system, user = self.prompts.classifier(records, taxonomy)
        return parse_classification_batch(
            self._invoke(self.classifier_model, system, user, ClassificationBatch)
        ).assignments

    def propose(
        self,
        failures: list[Assignment],
        representative_assignments: list[Assignment],
        records: list[ContractRecord],
        taxonomy: TaxonomyDocument,
        round_number: int,
    ) -> list[ExtensionProposal]:
        if not failures:
            return []
        system, user = self.prompts.designer(
            failures,
            representative_assignments,
            records,
            taxonomy,
            round_number,
        )
        return parse_proposal_batch(
            self._invoke(self.designer_model, system, user, ProposalBatch)
        ).proposals
