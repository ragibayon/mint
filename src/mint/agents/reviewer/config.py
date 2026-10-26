from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReviewerConfig:
    reviewer: str
    model: str
    enabled: bool
    execution: str
    reasoning_effort: str | None = None


REVIEWERS = {
    "qwen": ReviewerConfig(
        reviewer="qwen",
        model="qwen/qwen3.8-max-0902",
        enabled=True,
        execution="direct",
        reasoning_effort="low",
    ),
    "deepseek_flash": ReviewerConfig(
        reviewer="deepseek_flash",
        model="deepseek/deepseek-v4.1-flash:batch",
        enabled=True,
        execution="batch",
        reasoning_effort="low",
    ),
    "deepseek_pro": ReviewerConfig(
        reviewer="deepseek_pro",
        model="~deepseek/deepseek-pro-latest",
        enabled=False,
        execution="direct",
    ),
}


def get_reviewer_config(name: str, *, allow_disabled: bool = False) -> ReviewerConfig:
    try:
        config = REVIEWERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown reviewer: {name}") from exc
    if not config.enabled and not allow_disabled:
        raise ValueError(f"Reviewer {name} is disabled and requires explicit opt-in")
    return config
