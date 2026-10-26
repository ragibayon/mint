from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...Inferrer.schemas.output import Contract


class SourceCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    candidate_index: int = Field(ge=0)


GithubEvidenceSection = Literal[
    "issue_title",
    "issue_body",
    "issue_comments",
    "commit_message",
    "code_before",
    "code_after",
    "patch",
    "changed_files",
]


class ContractLineage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_index: int = Field(ge=0)
    source_candidates: list[SourceCandidate] = Field(min_length=1)
    evidence_sections: list[GithubEvidenceSection] = Field(min_length=1)


class CandidateDisposition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    candidate_index: int = Field(ge=0)
    disposition: Literal["excluded", "modified"]
    reason: str = Field(min_length=1)


class UnresolvedConflict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_candidates: list[SourceCandidate] = Field(min_length=1)
    issue: str = Field(min_length=1)


class NormalizationAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lineage: list[ContractLineage]
    excluded_or_modified: list[CandidateDisposition]
    unresolved: list[UnresolvedConflict]


class GithubNormalizationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["contracts_found", "no_contract", "abstain"]
    contracts: list[Contract]
    normalization_audit: NormalizationAudit
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_output(self) -> "GithubNormalizationOutput":
        if self.decision == "contracts_found" and not self.contracts:
            raise ValueError("contracts_found requires at least one contract")
        if self.decision != "contracts_found" and self.contracts:
            raise ValueError(f"{self.decision} requires an empty contracts array")
        normalized = sorted(
            item.normalized_index for item in self.normalization_audit.lineage
        )
        if normalized != list(range(len(self.contracts))):
            raise ValueError(
                "lineage must cover every normalized contract index exactly"
            )
        return self
