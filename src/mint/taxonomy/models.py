from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContractRecord(StrictModel):
    contract_id: str = Field(min_length=1)
    discussion_id: str = Field(min_length=1)
    normalized_index: int = Field(ge=0)
    contract: str = Field(min_length=1)
    apis: list[str]
    framework: str | None = None
    source_kind: str | None = None
    lineage: list[dict[str, Any]] = Field(default_factory=list)
    source_locator: str | None = None
    source_record_hash: str = Field(min_length=64, max_length=64)


class QuarantinedRecord(StrictModel):
    discussion_id: str | None = None
    normalized_index: int | None = None
    reason: str
    source_record_hash: str


class AdapterResult(StrictModel):
    contracts: list[ContractRecord]
    discussion_status_counts: dict[str, int]
    total_discussions: int
    contract_bearing_discussions: int
    quarantine: list[QuarantinedRecord]


class Assignment(StrictModel):
    contract_id: str = Field(min_length=1)
    taxonomy_version: str = Field(min_length=1)
    status: Literal[
        "classified", "unclassified", "ambiguous", "insufficient_information"
    ]
    primary_leaf_id: str | None = None
    reason: str = Field(min_length=1)
    decisive_span: str | None = None
    no_quote_reason: str | None = None
    candidate_alternatives: list[str] = Field(default_factory=list)
    considered_alternatives: list[str] = Field(default_factory=list)
    non_leaf_suggestions: list[str] = Field(default_factory=list)
    uncertainty_note: str | None = None
    orthogonal_facets: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_shape(self) -> "Assignment":
        if self.status == "classified":
            if not self.primary_leaf_id:
                raise ValueError("classified assignments require primary_leaf_id")
            if self.candidate_alternatives:
                raise ValueError("classified assignments cannot have alternatives")
        elif self.primary_leaf_id is not None:
            raise ValueError("only classified assignments may have primary_leaf_id")
        if self.status == "ambiguous" and len(set(self.candidate_alternatives)) < 2:
            raise ValueError("ambiguous assignments require at least two alternatives")
        if self.status != "ambiguous" and self.candidate_alternatives:
            raise ValueError("alternatives are allowed only for ambiguous assignments")
        if self.status == "ambiguous" and self.considered_alternatives:
            raise ValueError("ambiguous assignments use candidate_alternatives")
        if self.primary_leaf_id in self.considered_alternatives:
            raise ValueError("primary leaf cannot also be a considered alternative")
        if not self.decisive_span and not self.no_quote_reason:
            raise ValueError("provide decisive_span or no_quote_reason")
        return self


class ClassificationBatch(StrictModel):
    assignments: list[Assignment]


ProposalAction = Literal[
    "add_category",
    "clarify_extension_definition",
    "replace_extension_dimension",
    "merge_extension_categories",
    "split_extension_category",
    "rename_extension_category",
    "no_change",
    "needs_human_review",
]


class ExtensionProposal(StrictModel):
    proposal_id: str = Field(min_length=1)
    action: ProposalAction
    target_parent_id: str | None = None
    proposed_id: str | None = None
    proposed_name: str | None = None
    definition: str | None = None
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)
    distinct_from_siblings: list[str] = Field(default_factory=list)
    triggering_contract_ids: list[str] = Field(default_factory=list)
    why_seed_fails: str = Field(min_length=1)
    minimality_argument: str = Field(min_length=1)
    status: Literal[
        "proposed", "accepted", "rejected", "provisional", "human_review"
    ] = "proposed"
    decision_reason: str | None = None


class ProposalBatch(StrictModel):
    proposals: list[ExtensionProposal]


class TaxonomyNode(StrictModel):
    id: str
    name: str
    source_label: str
    parent_id: str | None
    definition: str
    inclusion_criteria: list[str]
    exclusion_criteria: list[str]
    taxonomy_origin: Literal["emse_2023", "mint_extension"]
    citations: list[str]
    immutable: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaxonomyDocument(StrictModel):
    taxonomy_id: str
    version: str
    title: str
    frozen: bool = False
    source: dict[str, str]
    nodes: list[TaxonomyNode]
    parent_version: str | None = None


class PipelineConfig(StrictModel):
    pilot_fraction: float = Field(default=0.10, gt=0, le=1)
    designer_sample_cap: int = Field(default=80, ge=1)
    designer_assigned_sample_cap: int = Field(default=20, ge=0)
    rng_seed: int = 2023
    stratification_fields: list[str] = Field(
        default_factory=lambda: ["framework", "discussion_id"]
    )
    max_refinement_rounds: int = Field(default=3, ge=0)
    max_auto_extensions_per_round: int = Field(default=5, ge=0)
    min_distinct_discussions_for_automatic_accept: int = Field(default=2, ge=1)
    classification_batch_size: int = Field(default=20, ge=1)
    automatic_extension_acceptance: bool = True
    classifier_model: str = "anthropic/claude-sonnet"
    designer_model: str = "anthropic/claude-opus"
    prompt_version: str = "refinement"
