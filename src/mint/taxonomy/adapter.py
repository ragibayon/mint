from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping

from pydantic import ValidationError

from .codebook import canonical_json
from .models import AdapterResult, ContractRecord, QuarantinedRecord


def _decode(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _records(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        csv.field_size_limit(50_000_000)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                yield {key: _decode(value) for key, value in row.items()}
        return
    parsed = json.loads(path.read_text(encoding="utf-8")) if suffix == ".json" else None
    if parsed is not None:
        values = parsed if isinstance(parsed, list) else parsed.get("records", [parsed])
        yield from values
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return slug or "discussion"


def adapt_normalized(path: Path) -> AdapterResult:
    contracts: list[ContractRecord] = []
    quarantine: list[QuarantinedRecord] = []
    status_counts: Counter[str] = Counter()
    discussion_ids: set[str] = set()
    bearing: set[str] = set()
    seen_ids: set[str] = set()
    for position, raw in enumerate(_records(path)):
        record_hash = hashlib.sha256(canonical_json(raw).encode()).hexdigest()
        discussion = _decode(raw.get("discussion"))
        discussion = discussion if isinstance(discussion, Mapping) else {}
        discussion_id = str(raw.get("discussion_id") or raw.get("id") or position)
        discussion_ids.add(discussion_id)
        decision = str(raw.get("decision") or raw.get("normalizer_decision") or "")
        status_counts[decision or "unknown"] += 1
        raw_contracts = _decode(
            raw.get("normalized_contracts", raw.get("contracts", []))
        )
        if decision in {"no_contract", "abstain"}:
            continue
        if not isinstance(raw_contracts, list):
            quarantine.append(
                QuarantinedRecord(
                    discussion_id=discussion_id,
                    reason="contracts field is not a JSON array",
                    source_record_hash=record_hash,
                )
            )
            continue
        lineage_items = _decode(raw.get("normalization_audit"))
        lineage_items = (
            lineage_items.get("lineage", [])
            if isinstance(lineage_items, Mapping)
            else []
        )
        for index, item in enumerate(raw_contracts):
            try:
                if not isinstance(item, Mapping):
                    raise ValueError("contract entry is not an object")
                contract_id = f"{_slug(discussion_id)}_n{index}"
                if contract_id in seen_ids:
                    raise ValueError(f"duplicate stable contract ID {contract_id}")
                lineage = [
                    value
                    for value in lineage_items
                    if value.get("normalized_index") == index
                ]
                source_locator = raw.get("source_locator") or discussion.get("link")
                source_kind = raw.get("source_kind")
                if not source_kind and str(source_locator or "").startswith(
                    "https://stackoverflow.com"
                ):
                    source_kind = "stackoverflow"
                contract = ContractRecord(
                    contract_id=contract_id,
                    discussion_id=discussion_id,
                    normalized_index=index,
                    contract=str(item.get("contract", "")).strip(),
                    apis=[str(api) for api in item.get("apis", [])],
                    framework=raw.get("framework"),
                    source_kind=source_kind,
                    lineage=lineage,
                    source_locator=source_locator,
                    source_record_hash=record_hash,
                )
                contracts.append(contract)
                seen_ids.add(contract_id)
                bearing.add(discussion_id)
            except (ValidationError, TypeError, ValueError) as exc:
                quarantine.append(
                    QuarantinedRecord(
                        discussion_id=discussion_id,
                        normalized_index=index,
                        reason=str(exc),
                        source_record_hash=record_hash,
                    )
                )
    return AdapterResult(
        contracts=contracts,
        discussion_status_counts=dict(sorted(status_counts.items())),
        total_discussions=len(discussion_ids),
        contract_bearing_discussions=len(bearing),
        quarantine=quarantine,
    )
