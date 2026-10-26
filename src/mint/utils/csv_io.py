from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO


def iter_csv_records(
    path: Path,
    *,
    required_fields: Iterable[str] = (),
) -> Iterator[dict[str, str]]:
    csv.field_size_limit(10_000_000)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(required_fields) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        yield from reader


def csv_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


@contextmanager
def csv_dict_writer(
    path: Path,
    fieldnames: list[str],
    *,
    append: bool,
) -> Iterator[csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not append or not path.exists() or path.stat().st_size == 0
    mode = "a" if append else "w"
    handle: TextIO
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
            handle.flush()
        yield writer


def csv_safe_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Encode nested values because CSV cells can contain only scalar text."""
    return {
        key: csv_json(value) if isinstance(value, (Mapping, list, tuple)) else value
        for key, value in row.items()
    }
