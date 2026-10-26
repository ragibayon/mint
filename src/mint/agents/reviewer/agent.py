from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from mint.llm.prompt_renderer import render_prompt

from .schemas.output import ReviewOutput

REVIEW_PROMPT_VERSION = "stackoverflow"


class ContractReviewer:
    def __init__(
        self,
        prompt_version: str = REVIEW_PROMPT_VERSION,
        prompts_dir: str | Path | None = None,
    ) -> None:
        self.prompt_version = prompt_version
        self.prompts_dir = (
            Path(prompts_dir)
            if prompts_dir is not None
            else Path(__file__).parent / "prompts" / prompt_version
        )
        missing = [
            name
            for name in ("system.j2", "user.j2")
            if not (self.prompts_dir / name).is_file()
        ]
        if missing:
            raise ValueError(
                f"Unknown or incomplete reviewer prompt version {prompt_version!r}: "
                f"missing {', '.join(missing)}"
            )

    def prompt_hashes(self) -> dict[str, str]:
        contents = {
            name: (self.prompts_dir / name).read_bytes()
            for name in ("system.j2", "user.j2")
        }
        bundle = b"".join(
            name.encode() + b"\0" + contents[name] + b"\0" for name in sorted(contents)
        )
        return {
            "system_prompt_sha256": hashlib.sha256(contents["system.j2"]).hexdigest(),
            "user_prompt_sha256": hashlib.sha256(contents["user.j2"]).hexdigest(),
            "prompt_bundle_sha256": hashlib.sha256(bundle).hexdigest(),
        }

    def render(self, record: Mapping[str, Any]) -> tuple[str, str]:
        discussion = record["discussion"]
        contracts = record["normalized_contracts"]
        indexed_contracts = [
            {
                "contract_index": index,
                "contract": contract["contract"],
                "apis": contract["apis"],
            }
            for index, contract in enumerate(contracts)
        ]
        if self.prompt_version in {
            "github",
            "huggingface",
        }:
            return (
                render_prompt(self.prompts_dir, "system.j2"),
                render_prompt(
                    self.prompts_dir,
                    "user.j2",
                    artifact=json.dumps(discussion, ensure_ascii=False, indent=2),
                    contracts=json.dumps(
                        indexed_contracts, ensure_ascii=False, indent=2
                    ),
                ),
            )
        return (
            render_prompt(self.prompts_dir, "system.j2"),
            render_prompt(
                self.prompts_dir,
                "user.j2",
                title=discussion["title"],
                body=discussion["body"],
                accepted_answer=discussion["accepted_answer"],
                contracts=json.dumps(indexed_contracts, ensure_ascii=False, indent=2),
            ),
        )

    @staticmethod
    def validate_coverage(output: ReviewOutput, contract_count: int) -> None:
        indices = [item.contract_index for item in output.contract_reviews]
        if len(indices) != len(set(indices)):
            raise ValueError("contract review indices must be unique")
        expected = list(range(contract_count))
        if sorted(indices) != expected:
            raise ValueError(
                f"contract reviews must cover indices {expected}; got {sorted(indices)}"
            )

    @staticmethod
    def normalized_contracts(record: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        contracts = record.get("normalized_contracts")
        if not isinstance(contracts, list):
            raise ValueError("normalized_contracts must be a list")
        return contracts
