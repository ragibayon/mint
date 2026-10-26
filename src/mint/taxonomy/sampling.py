from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict

from .models import ContractRecord


def stratified_pilot(
    records: list[ContractRecord], *, fraction: float, cap: int, seed: int
) -> tuple[list[ContractRecord], dict[str, object]]:
    if not records:
        return [], {"rng_seed": seed, "selected_contract_ids": [], "strata": {}}
    target = min(cap, max(1, math.ceil(len(records) * fraction)))
    strata: dict[str, list[ContractRecord]] = defaultdict(list)
    for record in records:
        strata[record.framework or "unknown"].append(record)
    rng = random.Random(seed)
    for values in strata.values():
        values.sort(key=lambda item: item.contract_id)
        rng.shuffle(values)
    selected: list[ContractRecord] = []
    keys = sorted(strata)
    while len(selected) < target and keys:
        remaining = []
        for key in keys:
            if len(selected) >= target:
                break
            if strata[key]:
                selected.append(strata[key].pop())
            if strata[key]:
                remaining.append(key)
        keys = remaining
    selected.sort(key=lambda item: item.contract_id)
    return selected, {
        "rng_seed": seed,
        "fraction": fraction,
        "cap": cap,
        "population_size": len(records),
        "selected_contract_ids": [item.contract_id for item in selected],
        "selected_sha256": hashlib.sha256(
            "\n".join(item.contract_id for item in selected).encode()
        ).hexdigest(),
        "strata": dict(
            sorted(
                (framework, sum(item.framework == framework for item in selected))
                for framework in {item.framework for item in selected}
            )
        ),
    }
