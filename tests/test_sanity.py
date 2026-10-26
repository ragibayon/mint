"""Offline artifact sanity tests; never submit model requests."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "stackoverflow": (576, 716, 576),
    "github": (209, 266, 171),
    "huggingface": (60, 61, 50),
}
csv.field_size_limit(50_000_000)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def check_corpus():
    for source, (count, _, _) in EXPECTED.items():
        path = (
            ROOT
            / "data/mint_corpus"
            / (
                "stackoverflow_corpus.csv"
                if source == "stackoverflow"
                else f"{source}_corpus.jsonl"
            )
        )
        rows = (
            read_csv(path)
            if path.suffix == ".csv"
            else [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            ]
        )
        require(
            len(rows) == count, f"{source}: expected {count} records, found {len(rows)}"
        )
        ids = [r["id"] for r in rows]
        require(len(ids) == len(set(ids)), f"{source}: duplicate corpus IDs")
        if source == "stackoverflow":
            matches = [r for r in rows if r["id"] == "76633836"]
            require(len(matches) == 1, "Motivating example is missing or duplicated")
            require(
                all(matches[0][k] for k in ("title", "body", "accepted_answer")),
                "Motivating example has empty evidence",
            )


def check_taxonomy():
    path = ROOT / "src/mint/taxonomy/prompts/classification/taxonomy.json"
    tree = json.loads(path.read_text())
    nodes = {node["id"]: node for node in tree["nodes"]}
    require(len(nodes) == len(tree["nodes"]), "Duplicate taxonomy node IDs")
    parents = {n["parent_id"] for n in nodes.values() if n.get("parent_id")}
    require(parents <= nodes.keys(), "Taxonomy has missing parents")
    leaves = set(nodes) - parents
    require(len(leaves) == 19, "Expected 19 terminal taxonomy categories")
    require(
        sum(leaf.startswith("MINT.") for leaf in leaves) == 7,
        "Expected seven introduced categories",
    )
    for node in nodes.values():
        seen, current = set(), node["id"]
        while current:
            require(current not in seen, "Cycle in taxonomy hierarchy")
            seen.add(current)
            current = nodes[current].get("parent_id")
    return tree, leaves, hashlib.sha256(path.read_bytes()).hexdigest()


def check_results():
    tree, leaves, digest = check_taxonomy()
    for source, (count, contracts_expected, classified_expected) in EXPECTED.items():
        rows = read_csv(ROOT / f"inferred_contracts/{source}.csv")
        require(len(rows) == count, f"{source}: result row count mismatch")
        require(
            len({r["instance_id"] for r in rows}) == count,
            f"{source}: duplicate result IDs",
        )
        corpus_path = (
            ROOT
            / "data/mint_corpus"
            / (
                "stackoverflow_corpus.csv"
                if source == "stackoverflow"
                else f"{source}_corpus.jsonl"
            )
        )
        corpus = (
            read_csv(corpus_path)
            if source == "stackoverflow"
            else [
                json.loads(line)
                for line in corpus_path.read_text().splitlines()
                if line.strip()
            ]
        )
        require(
            {r["id"] for r in corpus} == {r["instance_id"] for r in rows},
            f"{source}: results/corpus IDs differ",
        )
        total = classified = 0
        for row in rows:
            contracts = json.loads(row["normalization_contracts"])
            total += len(contracts)
            require(
                len(contracts) == int(row["normalization_contract_count"]),
                "Normalized count mismatch",
            )
            for model in ("gemini", "sonnet", "sol"):
                require(
                    row[f"infer_{model}_status"] == "ok",
                    f"{source}: unsuccessful saved inference",
                )
                require(
                    isinstance(json.loads(row[f"infer_{model}_contracts"]), list),
                    "Invalid inference contracts",
                )
            assignments = json.loads(row["taxonomy_assignments"])
            require(len(assignments) == len(contracts), "Taxonomy coverage mismatch")
            require(
                sorted(a["normalized_index"] for a in assignments)
                == list(range(len(contracts))),
                "Taxonomy indices are incomplete or duplicated",
            )
            eligible = set()
            for assignment in assignments:
                require(
                    assignment["taxonomy_version"] == tree["version"]
                    and assignment["taxonomy_hash"] == digest,
                    "Taxonomy identity/hash mismatch",
                )
                if assignment["status"] == "classified":
                    require(assignment["leaf_id"] in leaves, "Unknown assigned leaf")
                    eligible.add(assignment["contract_id"])
                    classified += 1
            for judge in ("qwen", "deepseek_flash"):
                reviews = json.loads(row[f"judge_{judge}_contract_reviews"])
                require(
                    sorted(r["contract_index"] for r in reviews)
                    == list(range(len(contracts))),
                    "Quality review coverage mismatch",
                )
                require(
                    json.loads(row[f"judge_{judge}_completeness"])["decision"]
                    in ("complete", "incomplete", "uncertain"),
                    "Invalid completeness decision",
                )
            for judge in ("qwen", "deepseek"):
                reviews = json.loads(row[f"taxonomy_{judge}_reviews"])
                require(
                    len(reviews) == len(eligible)
                    and {r["contract_id"] for r in reviews} == eligible,
                    "Assignment-review coverage mismatch",
                )
                require(
                    all(r["reviewed_leaf_id"] in leaves for r in reviews),
                    "Unknown reviewed leaf",
                )
        require(
            (total, classified) == (contracts_expected, classified_expected),
            f"{source}: contract/classification counts differ from the paper",
        )


def check_templates():
    from jinja2 import Environment

    sys.path.insert(0, str(ROOT / "src"))
    from mint.llm.prompt_renderer import render_prompt
    from mint.agents.normalizer.agent import ContractNormalizer
    from mint.agents.reviewer.agent import ContractReviewer

    for path in (ROOT / "src/mint").rglob("*.j2"):
        Environment().parse(path.read_text())
    directory = ROOT / "src/mint/agents/Inferrer/prompts/stackoverflow"
    context = {
        "title": "Example",
        "body": "Input must be a string.",
        "accepted_answer": "Pass a string.",
        "bug_context": None,
    }
    for name in ("system.j2", "user.j2"):
        require(
            bool(render_prompt(directory, name, **context)), "Empty inference prompt"
        )
    normalizer = ContractNormalizer()
    require(
        all(normalizer.render({"discussion": context, "candidate_outputs": []})),
        "Empty normalization prompt",
    )
    reviewer = ContractReviewer()
    require(
        all(reviewer.render({"discussion": context, "normalized_contracts": []})),
        "Empty quality-review prompt",
    )
    for folder in ("classification", "assignment-review"):
        directory = ROOT / "src/mint/taxonomy/prompts" / folder
        require(bool(render_prompt(directory, "system.j2")), "Empty taxonomy prompt")
        require(
            json.loads(
                render_prompt(directory, "user.j2", payload_json='{"check":true}')
            )
            == {"check": True},
            "Invalid taxonomy JSON template",
        )


def check_replay():
    with tempfile.TemporaryDirectory(prefix="mint-sanity-") as temporary:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/run_motivating_example.py"),
                "--output-dir",
                temporary,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=60,
        )
        require(
            result.returncode == 0,
            "Replay failed: " + (result.stderr or result.stdout)[-1500:],
        )
        summary = json.loads((Path(temporary) / "summary.json").read_text())
        require(
            summary["mode"] == "replay" and summary["model_calls_this_run"] == 0,
            "Expected offline replay with zero model calls",
        )
        require(
            len(summary["completed_stages"]) == 12,
            "Expected 12 completed replay stages",
        )
        require(
            len(summary["normalization"]["contracts"]) == 2,
            "Expected two saved normalized contracts",
        )


def check_live_settings():
    from dotenv import dotenv_values
    import os

    config = {**dotenv_values(ROOT / ".env"), **os.environ}
    key = (config.get("OPENROUTER_API_KEY") or "").strip()
    require(
        key
        and key not in ("your_actual_openrouter_api_key", "your_openrouter_api_key"),
        "Set your own OPENROUTER_API_KEY in .env",
    )
    require(not key.startswith("Bearer "), "Store only the API key, without Bearer")
    limit = int(config.get("MINT_MAX_OUTPUT_TOKENS", "8000"))
    timeout = float(config.get("MINT_REQUEST_TIMEOUT_SECONDS", "300"))
    require(limit > 0, "MINT_MAX_OUTPUT_TOKENS must be a positive integer")
    require(
        0 < timeout < float("inf"),
        "MINT_REQUEST_TIMEOUT_SECONDS must be positive and finite",
    )
    defaults = dotenv_values(ROOT / ".env.example")
    for role in ("GEMINI", "SONNET", "SOL", "NORMALIZER", "QWEN", "DEEPSEEK"):
        model = config.get(f"MINT_{role}_MODEL", defaults[f"MINT_{role}_MODEL"])
        require(
            model and "/" in model and not model.endswith(":batch"),
            f"MINT_{role}_MODEL must be a synchronous provider/model ID",
        )


def test_python_version():
    require(
        sys.version_info >= (3, 12),
        "Activate the Python 3.12 environment described in README.md",
    )


@pytest.mark.parametrize("module", ["jinja2", "pydantic", "dotenv", "requests"])
def test_example_dependencies(module):
    importlib.import_module(module)


def test_corpus_counts_and_motivating_evidence():
    check_corpus()


def test_taxonomy_hierarchy_and_19_leaves():
    check_taxonomy()


def test_result_alignment_and_paper_counts():
    check_results()


def test_prompt_templates_and_rendering():
    check_templates()


def test_complete_offline_replay():
    check_replay()


def test_live_run_rejects_missing_key_before_requests(tmp_path):
    import os

    environment = dict(os.environ)
    environment.pop("OPENROUTER_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_motivating_example.py"),
            "--live",
            "--env-file",
            str(tmp_path / "no-env-file"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=15,
    )
    require(
        result.returncode == 2 and "Set OPENROUTER_API_KEY" in result.stderr,
        "Live execution must reject a missing key before provider checks",
    )
    require(
        not (tmp_path / "output").exists(),
        "Missing-key run unexpectedly wrote stage artifacts",
    )


def test_local_live_configuration(request):
    if not request.config.getoption("--check-live"):
        pytest.skip(
            "Optional: fill .env and run pytest --check-live to check local live settings"
        )
    check_live_settings()
