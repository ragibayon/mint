"""Replay or rerun the complete model flow for Stack Overflow #76633836."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator
import requests

from mint.agents.Inferrer.schemas.output import InferenceOutput
from mint.agents.normalizer.agent import ContractNormalizer
from mint.agents.normalizer.schemas.output import NormalizationOutput
from mint.agents.reviewer.agent import ContractReviewer
from mint.agents.reviewer.schemas.output import ReviewOutput
from mint.configs import PROMPT_VERSION
from mint.llm.prompt_renderer import render_prompt

LOGGER = logging.getLogger("mint.example")
INSTANCE_ID = "76633836"
TAXONOMY_DIR = ROOT / "src/mint/taxonomy/prompts/classification"
MODEL_DEFAULTS = {
    "gemini": "google/gemini-3.8-flash",
    "sonnet": "anthropic/claude-sonnet-5.5",
    "sol": "openai/gpt-6.1-sol",
    "normalizer": "anthropic/claude-opus-5.5",
    "qwen": "qwen/qwen3.8-max-0902",
    "deepseek": "deepseek/deepseek-v4.1-flash",
}


class TaxonomyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contract_id: str = Field(min_length=1)
    taxonomy_version: str = Field(min_length=1)
    taxonomy_hash: str = Field(min_length=64, max_length=64)
    status: Literal["classified", "ambiguous", "unclassified"]
    leaf_id: str | None
    competing_leaf_ids: list[str]
    unclassified_reason: Literal["no-matching-leaf", "insufficient-evidence"] | None
    justification: str = Field(min_length=1)
    evidence_quote: str | None

    @model_validator(mode="after")
    def consistent_status(self):
        if len(self.competing_leaf_ids) != len(set(self.competing_leaf_ids)):
            raise ValueError("Duplicate competing leaves")
        if self.status == "classified" and (
            not self.leaf_id or self.competing_leaf_ids or self.unclassified_reason
        ):
            raise ValueError("Inconsistent classified decision")
        if self.status == "ambiguous" and (
            self.leaf_id or len(self.competing_leaf_ids) < 2 or self.unclassified_reason
        ):
            raise ValueError("Inconsistent ambiguous decision")
        if self.status == "unclassified" and (
            self.leaf_id or self.competing_leaf_ids or not self.unclassified_reason
        ):
            raise ValueError("Inconsistent unclassified decision")
        return self


class AssignmentReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["ACCEPT", "ALTERNATIVE"]
    suggested_leaf: str | None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_decision(self):
        if self.decision == "ACCEPT" and self.suggested_leaf is not None:
            raise ValueError("ACCEPT requires suggested_leaf=null")
        if self.decision == "ALTERNATIVE" and not self.suggested_leaf:
            raise ValueError("ALTERNATIVE requires a suggested leaf")
        return self


def read_record(path, key):
    csv.field_size_limit(50_000_000)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        matches = [r for r in csv.DictReader(handle) if r[key] == INSTANCE_ID]
    if len(matches) != 1:
        raise ValueError(f"Expected one motivating record in {path}")
    return matches[0]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def messages(system, user):
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class Runner:
    def __init__(self, live, output, models):
        self.live, self.output, self.models = live, output, models
        output.mkdir(parents=True, exist_ok=True)
        if not live and any(output.glob("*.request.json")):
            raise ValueError("Choose a separate directory for offline replay")
        self.calls = 0
        self.stages = []
        self.token_limit = int(os.getenv("MINT_MAX_OUTPUT_TOKENS", "8000"))
        self.timeout = float(os.getenv("MINT_REQUEST_TIMEOUT_SECONDS", "300"))
        if self.token_limit < 1 or self.timeout <= 0:
            raise ValueError("Token limit and timeout must be positive")

    def run(
        self, stage, role, prompt, schema, saved, effort="low", response_format=None
    ):
        path = self.output / f"{stage}.json"
        if not self.live:
            parsed = schema.model_validate(saved)
            write_json(
                path, {"mode": "replay", "output": parsed.model_dump(mode="json")}
            )
        else:
            body = {
                "model": self.models[role],
                "messages": prompt,
                "response_format": response_format
                or {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "strict": True,
                        "schema": schema.model_json_schema(),
                    },
                },
                "max_tokens": self.token_limit,
                "reasoning": {"effort": effort},
                "provider": {"require_parameters": True},
                "usage": {"include": True},
            }
            digest = hashlib.sha256(
                json.dumps(body, sort_keys=True).encode()
            ).hexdigest()
            request_path = self.output / f"{stage}.request.json"
            if request_path.exists():
                previous = json.loads(request_path.read_text())
                if previous != body:
                    raise ValueError(
                        f"Changed request for {stage}; choose a new output directory"
                    )
            write_json(request_path, body)
            raw_path = self.output / f"{stage}.response.json"
            if path.exists():
                cached = json.loads(path.read_text())
                if (
                    cached.get("mode") != "live"
                    or cached.get("request_sha256") != digest
                ):
                    raise ValueError(f"Output mode/request mismatch for {stage}")
                parsed = schema.model_validate(cached["output"])
                LOGGER.info("%s: reusing saved successful stage", stage)
                self.stages.append(stage)
                return parsed.model_dump(mode="json")
            if raw_path.exists():
                raw = json.loads(raw_path.read_text())
                elapsed = None
            else:
                started = time.monotonic()
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", ""),
                        "X-OpenRouter-Title": os.getenv("OPENROUTER_SITE_NAME", "MINT"),
                    },
                    json=body,
                    timeout=self.timeout,
                )
                self.calls += 1
                elapsed = time.monotonic() - started
                response.raise_for_status()
                raw = response.json()
                write_json(raw_path, raw)
            if raw.get("error"):
                raise ValueError(
                    f"{stage}: provider returned an error; inspect {raw_path}"
                )
            choice = raw["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError(f"{stage}: incomplete response; inspect {raw_path}")
            parsed = schema.model_validate_json(choice["message"]["content"])
            write_json(
                path,
                {
                    "mode": "live",
                    "request_sha256": digest,
                    "model": raw.get("model", self.models[role]),
                    "elapsed_seconds": elapsed,
                    "usage": raw.get("usage"),
                    "output": parsed.model_dump(mode="json"),
                },
            )
        self.stages.append(stage)
        LOGGER.info("%s: schema-valid output", stage)
        return parsed.model_dump(mode="json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Make paid OpenRouter calls instead of offline replay.",
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Use the same directory to reuse matching successful live stages.",
    )
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)
    models = {
        role: os.getenv(f"MINT_{role.upper()}_MODEL", value)
        for role, value in MODEL_DEFAULTS.items()
    }
    if args.live and not os.getenv("OPENROUTER_API_KEY", "").strip():
        parser.error("Set OPENROUTER_API_KEY in .env before using --live")
    if args.live and any(model.endswith(":batch") for model in models.values()):
        parser.error("Use synchronous model IDs without :batch for --live")
    mode = "live" if args.live else "replay"
    default_name = (
        f"live-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}" if args.live else "replay"
    )
    output = args.output_dir or ROOT / "outputs/motivating_example" / default_name
    if args.live:
        # Verify configured model IDs before any generation request.
        response = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
        response.raise_for_status()
        catalog = {r["id"]: r for r in response.json()["data"]}
        for model in set(models.values()):
            if model not in catalog:
                parser.error(f"Model unavailable: {model}; update .env before running")
            if "response_format" not in catalog[model].get("supported_parameters", []):
                parser.error(f"Model lacks structured-output support: {model}")
    runner = Runner(args.live, output, models)
    corpus = read_record(ROOT / "data/mint_corpus/stackoverflow_corpus.csv", "id")
    archive = (
        read_record(ROOT / "inferred_contracts/stackoverflow.csv", "instance_id")
        if not args.live
        else {}
    )
    discussion = {k: corpus[k] for k in ("title", "body", "accepted_answer")}
    write_json(output / "input.json", {"id": INSTANCE_ID, "discussion": discussion})
    prompt_dir = ROOT / f"src/mint/agents/Inferrer/prompts/{PROMPT_VERSION}"
    context = {**discussion, "bug_context": None}
    infer_prompt = messages(
        *(
            render_prompt(prompt_dir, name, **context)
            for name in ("system.j2", "user.j2")
        )
    )
    candidates = []
    for role in ("gemini", "sonnet", "sol"):
        saved = (
            None
            if args.live
            else {
                "decision": archive[f"infer_{role}_decision"],
                "contracts": json.loads(archive[f"infer_{role}_contracts"]),
                "rationale": archive[f"infer_{role}_rationale"],
            }
        )
        result = runner.run(
            f"01-infer-{role}", role, infer_prompt, InferenceOutput, saved
        )
        candidates.append(
            {
                "source": role,
                "model": models[role] if args.live else archive[f"infer_{role}_model"],
                **result,
            }
        )
    normalizer = ContractNormalizer()
    normal_prompt = messages(
        *normalizer.render({"discussion": discussion, "candidate_outputs": candidates})
    )
    saved = (
        None
        if args.live
        else {
            "decision": archive["normalization_decision"],
            "contracts": json.loads(archive["normalization_contracts"]),
            "normalization_audit": json.loads(archive["normalization_audit"]),
            "rationale": archive["normalization_rationale"],
        }
    )
    normal = runner.run(
        "02-normalize",
        "normalizer",
        normal_prompt,
        NormalizationOutput,
        saved,
        effort="high",
    )
    normalizer.validate_audit(
        NormalizationOutput.model_validate(normal),
        normalizer.candidate_refs(candidates),
    )
    reviewer = ContractReviewer()
    review_prompt = messages(
        *reviewer.render(
            {"discussion": discussion, "normalized_contracts": normal["contracts"]}
        )
    )
    quality = {}
    for role, prefix in [("qwen", "qwen"), ("deepseek", "deepseek_flash")]:
        saved = (
            None
            if args.live
            else {
                "contract_reviews": json.loads(
                    archive[f"judge_{prefix}_contract_reviews"]
                ),
                "completeness": json.loads(archive[f"judge_{prefix}_completeness"]),
            }
        )
        quality[role] = runner.run(
            f"03-quality-{role}", role, review_prompt, ReviewOutput, saved
        )
        reviewer.validate_coverage(
            ReviewOutput.model_validate(quality[role]), len(normal["contracts"])
        )
    tree_path = TAXONOMY_DIR / "taxonomy.json"
    tree = json.loads(tree_path.read_text())
    taxonomy_hash = hashlib.sha256(tree_path.read_bytes()).hexdigest()
    nodes = {node["id"]: node for node in tree["nodes"]}
    leaves = sorted(
        set(nodes)
        - {node["parent_id"] for node in tree["nodes"] if node.get("parent_id")}
    )
    excerpt_full = json.dumps(discussion, ensure_ascii=False)
    excerpt = excerpt_full[:4000]
    final = (
        {}
        if args.live
        else {a["contract_id"]: a for a in json.loads(archive["taxonomy_assignments"])}
    )
    saved_reviews = (
        {}
        if args.live
        else {
            role: {
                r["contract_id"]: r
                for r in json.loads(archive[f"taxonomy_{role}_reviews"])
            }
            for role in ("qwen", "deepseek")
        }
    )
    assignments, assignment_reviews = [], []
    for position, contract in enumerate(normal["contracts"]):
        cid = f"{INSTANCE_ID}_n{position}"
        user = {
            "taxonomy_version": tree["version"],
            "taxonomy_hash": taxonomy_hash,
            "taxonomy": tree,
            "terminal_leaf_ids": leaves,
            "contract": {
                "contract_id": cid,
                **contract,
                "source_context_excerpt": excerpt,
                "context_truncated": len(excerpt_full) > 4000,
            },
        }
        saved = (
            None
            if args.live
            else {key: final[cid][key] for key in TaxonomyDecision.model_fields}
        )
        response_format = json.loads(
            (TAXONOMY_DIR / "response-schema.json").read_text()
        )
        assignment = runner.run(
            f"04-taxonomy-{position}",
            "sonnet",
            messages(
                render_prompt(TAXONOMY_DIR, "system.j2"),
                render_prompt(
                    TAXONOMY_DIR,
                    "user.j2",
                    payload_json=json.dumps(user, ensure_ascii=False),
                ),
            ),
            TaxonomyDecision,
            saved,
            effort="medium",
            response_format=response_format,
        )
        if (
            assignment["contract_id"] != cid
            or assignment["taxonomy_version"] != tree["version"]
            or assignment["taxonomy_hash"] != taxonomy_hash
        ):
            raise ValueError("Taxonomy identity mismatch")
        if assignment["leaf_id"] and assignment["leaf_id"] not in leaves:
            raise ValueError("Invalid terminal leaf")
        if any(leaf not in leaves for leaf in assignment["competing_leaf_ids"]):
            raise ValueError("Invalid competing leaf")
        assignments.append(assignment)
        if assignment["status"] != "classified":
            continue
        node = nodes[assignment["leaf_id"]]
        review_user = {
            "contract": contract["contract"],
            "associated_apis": contract["apis"],
            "framework": corpus["framework"],
            "developer_source": "stackoverflow",
            "source_evidence_excerpt": excerpt,
            "evidence_truncated": len(excerpt_full) > 4000,
            "original_assignment": {
                k: node[k]
                for k in (
                    "id",
                    "name",
                    "definition",
                    "inclusion_criteria",
                    "exclusion_criteria",
                )
            },
            "frozen_taxonomy": tree,
            "permissible_leaf_ids": leaves,
        }
        review_dir = ROOT / "src/mint/taxonomy/prompts/assignment-review"
        prompt = messages(
            render_prompt(review_dir, "system.j2"),
            render_prompt(
                review_dir,
                "user.j2",
                payload_json=json.dumps(review_user, ensure_ascii=False),
            ),
        )
        for role in ("qwen", "deepseek"):
            saved = (
                None
                if args.live
                else {
                    key: saved_reviews[role][cid][key]
                    for key in AssignmentReview.model_fields
                }
            )
            review = runner.run(
                f"05-assignment-{position}-{role}",
                role,
                prompt,
                AssignmentReview,
                saved,
            )
            if review["decision"] == "ALTERNATIVE" and (
                review["suggested_leaf"] not in leaves
                or review["suggested_leaf"] == node["id"]
            ):
                raise ValueError("Alternative must be a different valid terminal leaf")
            assignment_reviews.append({"contract_id": cid, "reviewer": role, **review})
    summary = {
        "mode": mode,
        "instance_id": INSTANCE_ID,
        "source_url": f"https://stackoverflow.com/questions/{INSTANCE_ID}",
        "model_calls_this_run": runner.calls,
        "completed_stages": runner.stages,
        "proposers": candidates,
        "normalization": normal,
        "quality_reviews": quality,
        "taxonomy_assignments": assignments,
        "taxonomy_reviews": assignment_reviews,
        "human_review_included": False,
        "scope": "Fixed final taxonomy; no new taxonomy construction or runtime API-contract execution.",
    }
    write_json(output / "summary.json", summary)
    LOGGER.info(
        "%s complete: %d normalized contracts; %d model calls this run; output: %s",
        mode,
        len(normal["contracts"]),
        runner.calls,
        output,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        main()
    except (ValueError, KeyError, OSError, requests.RequestException) as error:
        LOGGER.error("%s", error)
        sys.exit(1)
