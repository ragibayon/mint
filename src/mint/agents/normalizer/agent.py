from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from mint.configs import NORMALIZATION_PROMPT_VERSION

from .schemas.output import NormalizationOutput
from ...llm.prompt_renderer import render_prompt


class ContractNormalizer:
    def __init__(
        self,
        prompt_version: str = NORMALIZATION_PROMPT_VERSION,
        prompts_dir: str | Path | None = None,
    ) -> None:
        self.prompt_version = prompt_version
        if prompts_dir is not None:
            self.prompts_dir = Path(prompts_dir)
        else:
            root = Path(__file__).parent / "prompts"
            self.prompts_dir = root / prompt_version
        missing = [
            name
            for name in ("system.j2", "user.j2")
            if not (self.prompts_dir / name).is_file()
        ]
        if missing:
            raise ValueError(
                f"Unknown or incomplete normalizer prompt version {prompt_version!r}: "
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
        candidate_outputs = record["candidate_outputs"]
        return (
            render_prompt(self.prompts_dir, "system.j2"),
            render_prompt(
                self.prompts_dir,
                "user.j2",
                discussion_json=json.dumps(discussion, ensure_ascii=False, indent=2),
                candidate_outputs_json=json.dumps(
                    candidate_outputs, ensure_ascii=False, indent=2
                ),
            ),
        )

    @staticmethod
    def candidate_refs(
        candidate_outputs: list[Mapping[str, Any]],
    ) -> set[tuple[str, int]]:
        return {
            (str(output["source"]), index)
            for output in candidate_outputs
            for index, _ in enumerate(output.get("contracts", []))
        }

    @classmethod
    def validate_audit(
        cls,
        output: NormalizationOutput,
        expected_refs: set[tuple[str, int]],
    ) -> None:
        audit = output.normalization_audit
        observed = {
            (ref.source, ref.candidate_index)
            for item in audit.lineage
            for ref in item.source_candidates
        }
        observed.update(
            (item.source, item.candidate_index) for item in audit.excluded_or_modified
        )
        observed.update(
            (ref.source, ref.candidate_index)
            for item in audit.unresolved
            for ref in item.source_candidates
        )
        unknown = observed - expected_refs
        missing = expected_refs - observed
        if unknown:
            raise ValueError(f"audit references unknown candidates: {sorted(unknown)}")
        if missing:
            raise ValueError(f"audit omits candidates: {sorted(missing)}")
