from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_index: int = Field(ge=0)
    validity: Literal["supported", "partial", "unsupported"]
    api_correctness: Literal["correct", "partial", "incorrect", "not_applicable"]
    rationale: str = Field(min_length=1)


class MissingRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    evidence: str = Field(min_length=1)


class CompletenessReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["complete", "incomplete", "uncertain"]
    rationale: str = Field(min_length=1)
    missing_requirements: list[MissingRequirement]

    @model_validator(mode="after")
    def validate_missing_requirements(self) -> "CompletenessReview":
        if self.decision == "incomplete" and not self.missing_requirements:
            raise ValueError("incomplete requires at least one missing requirement")
        if self.decision != "incomplete" and self.missing_requirements:
            raise ValueError(
                "missing requirements are allowed only for an incomplete decision"
            )
        return self


class ReviewOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_reviews: list[ContractReview]
    completeness: CompletenessReview
