from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_json(path: Path, value: object) -> None:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_jsonl(path: Path, values: Iterable[object]) -> None:
    lines = []
    for value in values:
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        lines.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
    atomic_text(path, "\n".join(lines) + ("\n" if lines else ""))
