from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    apis: list[str] = Field(
        description=(
            "All API methods or operations governed by this contract, listed in "
            "relationship order when order matters. "
            "Preserve the most specific callable names supported by the "
            "discussion. Do not invent namespaces. Return an empty list "
            "when no specific callable can be identified."
        )
    )

    contract: str = Field(
        description=(
            "A precise, self-contained, evidence-supported natural-language "
            "behavioral requirement, preserving its applicable conditions "
            "and relationships between operations. Version-sensitive signatures, "
            "argument names, return shapes, and migration behavior must be scoped "
            "with concrete, human-readable context such as the named client, "
            "framework, version, or observed compatibility behavior. Avoid generic "
            "boilerplate when the evidence supports a more explicit condition."
        ),
        examples=[
            (
                "When using the OpenAI Python client exhibiting the reported "
                "engine-parameter requirement, callers must specify the model "
                "identifier through the 'engine' argument rather than the 'model' "
                "argument."
            )
        ],
    )


class InferenceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["contracts_found", "no_contract", "abstain"] = Field(
        description=(
            "The post-level inference outcome. Use contracts_found when at least "
            "one defensible contract is returned, no_contract when the evidence "
            "is sufficient and establishes no actionable behavioral obligation, "
            "or abstain when missing, ambiguous, or conflicting evidence prevents "
            "a reliable decision."
        )
    )

    contracts: list[Contract] = Field(
        description=(
            "All distinct evidence-supported software contracts. "
            "Return an empty list when no contract can be inferred."
        )
    )

    rationale: str = Field(
        description=(
            "A concise explanation of how the discussion supports the "
            "inferred contracts, or why no contract can be inferred."
        )
    )

    @model_validator(mode="after")
    def validate_decision(self) -> "InferenceOutput":
        if self.decision == "contracts_found" and not self.contracts:
            raise ValueError("contracts_found requires at least one contract")
        if self.decision != "contracts_found" and self.contracts:
            raise ValueError(f"{self.decision} requires an empty contracts array")
        return self
