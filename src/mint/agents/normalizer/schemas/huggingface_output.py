from typing import Literal

from pydantic import Field

from .github_output import (
    ContractLineage,
    GithubNormalizationOutput,
    NormalizationAudit,
)


class HuggingFaceContractLineage(ContractLineage):
    evidence_sections: list[
        Literal["topic_title", "topic_metadata", "topic_posts", "solution_markers"]
    ] = Field(min_length=1)


class HuggingFaceNormalizationAudit(NormalizationAudit):
    lineage: list[HuggingFaceContractLineage]


class HuggingFaceNormalizationOutput(GithubNormalizationOutput):
    normalization_audit: HuggingFaceNormalizationAudit
