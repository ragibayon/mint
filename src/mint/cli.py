from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .agents.Inferrer.agent import ContractInferrer
from .configs import DEFAULT_MODELS, PROMPT_VERSION
from .utils.csv_io import csv_safe_row

LOGGER = logging.getLogger(__name__)
CONSOLE = Console(stderr=True)

INPUT_FIELDS = ("title", "body", "accepted_answer")
METADATA_FIELDS = (
    "bug_type",
    "language",
    "component",
    "framework",
    "root_cause",
    "effect",
    "rational",
)
PROMPT_METADATA_FIELDS = (
    "bug_type",
    "language",
    "component",
    "framework",
    "root_cause",
    "effect",
)
OUTPUT_FIELDS = (
    "id",
    "model",
    "prompt_version",
    "bug_metadata_included",
    *METADATA_FIELDS,
    "decision",
    "contracts",
    "inference_rationale",
    "prompt_tokens",
    "completion_tokens",
    "inference_duration_seconds",
    "status",
    "error",
    "timestamp",
)


def default_output_path() -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("results") / f"contract_inference_{timestamp}.csv"


def configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]",
        handlers=[
            RichHandler(
                console=CONSOLE,
                show_time=True,
                show_level=True,
                show_path=False,
                rich_tracebacks=True,
                markup=False,
            ),
            file_handler,
        ],
        force=True,
    )


def show_run_panel(
    args: argparse.Namespace, questions: int, pending: int, skipped: int
) -> None:
    details = Table.grid(padding=(0, 2))
    details.add_column(style="bold cyan")
    details.add_column(style="white")
    details.add_row("Dataset", str(args.dataset_dir))
    details.add_row("Metadata", str(args.metadata_dir))
    details.add_row("Sheet", args.sheet)
    details.add_row("Output", str(args.output))
    details.add_row("Log", str(args.log_file))
    details.add_row("Models", ", ".join(args.models))
    details.add_row("Questions", str(questions))
    details.add_row("Pending", str(pending))
    details.add_row("Resumed", str(skipped))
    CONSOLE.print(
        Panel(
            details, title="[bold magenta]Contract inference[/]", border_style="magenta"
        )
    )


def show_summary(*, succeeded: int, failed: int, skipped: int, output: Path) -> None:
    details = Table.grid(padding=(0, 2))
    details.add_column(style="bold")
    details.add_column()
    details.add_row("Succeeded", f"[green]{succeeded}[/]")
    details.add_row("Failed", f"[red]{failed}[/]")
    details.add_row("Resumed", f"[yellow]{skipped}[/]")
    details.add_row("Results", f"[bold cyan]{output}[/]")
    style = "red" if failed else "green"
    CONSOLE.print(Panel(details, title="[bold]Run complete[/]", border_style=style))


def successful_runs(path: Path) -> set[tuple[str, str]]:
    """Return saved post/model pairs so a resumed run does not repeat them."""
    if not path.exists():
        return set()

    results = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"id", "model", "status"} - set(results.columns)
    if missing:
        raise ValueError(f"Missing result columns: {', '.join(sorted(missing))}")
    successful = results.loc[results["status"].eq("ok"), ["id", "model"]]
    return set(successful.itertuples(index=False, name=None))


def dataset_sheet_path(dataset_dir: Path, sheet: str) -> Path:
    if sheet == "stackOverflow" and dataset_dir.name == "mint_corpus":
        filename = "stackoverflow_corpus.csv"
    elif sheet == "stackOverflow" and dataset_dir.name == "filtered_dataset":
        filename = "filtered_stackoverflow_posts.csv"
    else:
        filename = f"{sheet}.csv"
    return dataset_dir / filename


def dataset_records(
    dataset_dir: Path,
    sheet: str,
    metadata_dir: Path,
) -> pd.DataFrame:
    sheet_path = dataset_sheet_path(dataset_dir, sheet)
    if not sheet_path.is_file():
        raise ValueError(f"Dataset sheet not found: {sheet_path}")

    rows = pd.read_csv(sheet_path, dtype=str, keep_default_na=False)
    if dataset_dir.name == "mint_corpus" and set(INPUT_FIELDS).issubset(rows.columns):
        for field in METADATA_FIELDS:
            if field not in rows.columns:
                rows[field] = ""
        return rows
    if sheet == "stackOverflow" and set(INPUT_FIELDS).issubset(rows.columns):
        labels_path = metadata_dir / "stackOverflow.csv"
        if not labels_path.is_file():
            raise ValueError(f"Metadata sheet not found: {labels_path}")
        labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
        required_labels = {"id", *METADATA_FIELDS}
        missing = required_labels - set(labels.columns)
        if missing:
            raise ValueError(
                f"Missing Stack Overflow metadata columns: {', '.join(sorted(missing))}"
            )
        content = rows[["id", *INPUT_FIELDS]].drop_duplicates("id", keep="first")
        annotations = labels[["id", *METADATA_FIELDS]].copy()

        def merge_unique(values: pd.Series) -> str:
            unique = list(
                dict.fromkeys(value.strip() for value in values if value.strip())
            )
            return " || ".join(unique)

        metadata = (
            annotations.groupby("id", sort=False, dropna=False)
            .agg({field: merge_unique for field in METADATA_FIELDS})
            .reset_index()
        )
        return metadata.merge(
            content,
            on="id",
            how="inner",
            validate="one_to_one",
        )
    if set(INPUT_FIELDS).issubset(rows.columns):
        return rows

    posts_path = dataset_dir / "title_body.csv"
    posts = pd.read_csv(posts_path, dtype=str, keep_default_na=False)
    required_posts = {"id", "title", "body"}
    missing = required_posts - set(posts.columns)
    if missing:
        raise ValueError(f"Missing title/body columns: {', '.join(sorted(missing))}")
    return rows.merge(
        posts[["id", "title", "body"]],
        on="id",
        how="left",
        validate="one_to_one",
    ).fillna("")


def selected_records(
    rows: pd.DataFrame,
    *,
    start: int,
    limit: int | None,
) -> list[tuple[int, dict[str, str]]]:
    missing = set(INPUT_FIELDS) - set(rows.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {', '.join(sorted(missing))}")
    stop = None if limit is None else start + limit
    selected = rows.iloc[start:stop]
    return [(int(index), row.to_dict()) for index, row in selected.iterrows()]


def append_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([csv_safe_row(result)], columns=OUTPUT_FIELDS)
    if path.exists() and path.stat().st_size:
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        same_run = existing["id"].eq(str(result["id"])) & existing["model"].eq(
            str(result["model"])
        )
        row = pd.concat([existing.loc[~same_run], row], ignore_index=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    row.to_csv(temporary, index=False)
    temporary.replace(path)


def empty_result(
    *,
    record_id: str,
    model: str,
    row: dict[str, str],
    include_bug_metadata: bool = False,
) -> dict[str, Any]:
    return {
        "id": record_id,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "bug_metadata_included": include_bug_metadata,
        **{field: row.get(field, "") for field in METADATA_FIELDS},
        "decision": "",
        "contracts": [],
        "inference_rationale": "",
        "prompt_tokens": "",
        "completion_tokens": "",
        "inference_duration_seconds": "",
        "status": "error",
        "error": "",
        "timestamp": "",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer natural-language contracts from developer discussions."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/mint_corpus"),
        help="Folder containing the dataset CSV sheets.",
    )
    parser.add_argument(
        "--sheet",
        default="stackOverflow",
        help="CSV sheet name without the .csv extension.",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("data/mint_corpus"),
        help="Folder containing dataset labels and rationales.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to results/contract_inference_<timestamp>.csv.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Defaults to logs/<output_stem>.log and appends on resume.",
    )
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODELS[0]])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--include-bug-metadata",
        action="store_true",
        help="Include bug metadata in the model prompt for an explicit ablation.",
    )
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = default_output_path()
    if args.log_file is None:
        args.log_file = Path("logs") / f"{args.output.stem}.log"

    if args.start < 0 or (args.limit is not None and args.limit < 1):
        parser.error("--start must be >= 0 and --limit must be >= 1")
    if len(args.models) != 1:
        parser.error("Run exactly one model at a time; provide one value to --models")
    if not args.dataset_dir.is_dir():
        parser.error(f"Dataset folder not found: {args.dataset_dir}")
    if not args.metadata_dir.is_dir():
        parser.error(f"Metadata folder not found: {args.metadata_dir}")
    sheet_path = dataset_sheet_path(args.dataset_dir, args.sheet)
    if not sheet_path.is_file():
        parser.error(f"Dataset sheet not found: {sheet_path}")
    if args.output.exists() and not args.resume:
        parser.error(
            f"Output already exists: {args.output}. Use --resume or a new --output."
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file)

    completed = successful_runs(args.output) if args.resume else set()
    dataset = dataset_records(args.dataset_dir, args.sheet, args.metadata_dir)
    rows = selected_records(dataset, start=args.start, limit=args.limit)
    pending = [
        (index, row, model)
        for index, row in rows
        for model in args.models
        if ((row.get("id") or row.get("url") or str(index)).strip(), model)
        not in completed
    ]
    skipped = len(rows) * len(args.models) - len(pending)
    agents: dict[str, ContractInferrer] = {}
    succeeded = failed = 0

    show_run_panel(args, len(rows), len(pending), skipped)
    LOGGER.info(
        "Run dataset=%s sheet=%s models=%s output=%s log=%s questions=%d pending=%d resumed=%d",
        args.dataset_dir,
        args.sheet,
        ",".join(args.models),
        args.output,
        args.log_file,
        len(rows),
        len(pending),
        skipped,
    )

    with Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold blue]Inferring contracts"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("[cyan]{task.fields[current]}"),
        console=CONSOLE,
    ) as progress:
        task = progress.add_task("inference", total=len(pending), current="waiting")
        for position, (index, row, model) in enumerate(pending, start=1):
            record_id = (row.get("id") or row.get("url") or str(index)).strip()
            title = (row.get("title") or "").strip()
            progress.update(task, current=f"id={record_id} model={model}")
            LOGGER.info(
                "Starting id=%s model=%s remaining=%d question=%r",
                record_id,
                model,
                len(pending) - position,
                title,
            )

            discussion = {key: (row.get(key) or "").strip() for key in INPUT_FIELDS}
            if args.include_bug_metadata:
                discussion.update(
                    {
                        "id": record_id,
                        "bug_context": True,
                        **{
                            field: (row.get(field) or "unknown").strip() or "unknown"
                            for field in PROMPT_METADATA_FIELDS
                        },
                    }
                )
            output = empty_result(
                record_id=record_id,
                model=model,
                row=row,
                include_bug_metadata=args.include_bug_metadata,
            )
            try:
                missing = [key for key, value in discussion.items() if not value]
                if missing:
                    raise ValueError(
                        f"Missing discussion content: {', '.join(missing)}"
                    )
                agent = agents.get(model)
                if agent is None:
                    agent = ContractInferrer(model=model)
                    agents[model] = agent
                inference = agent.infer(discussion)
                metrics = inference.get("metrics") or {}
                output.update(
                    model=metrics.get("model", model),
                    decision=inference["output"]["decision"],
                    contracts=inference["output"]["contracts"],
                    inference_rationale=inference["output"]["rationale"],
                    prompt_tokens=metrics.get("prompt_tokens", ""),
                    completion_tokens=metrics.get("completion_tokens", ""),
                    inference_duration_seconds=(
                        metrics["total_duration_ns"] / 1_000_000_000
                        if metrics.get("total_duration_ns") is not None
                        else ""
                    ),
                    status="ok",
                    timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
                )
                succeeded += 1
                LOGGER.info("Completed id=%s model=%s", record_id, model)
            except Exception as exc:
                failed += 1
                output.update(
                    error=f"{type(exc).__name__}: {exc}",
                    timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
                )
                LOGGER.exception("Failed id=%s model=%s", record_id, model)

            append_result(args.output, output)

            progress.advance(task)

    for model, agent in agents.items():
        unload = getattr(agent, "unload", None)
        if not callable(unload):
            continue
        try:
            unload()
            LOGGER.info("Unloaded model=%s", model)
        except Exception as exc:
            LOGGER.warning(
                "Could not unload model=%s error=%s: %s",
                model,
                type(exc).__name__,
                exc,
            )

    show_summary(
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
        output=args.output,
    )
    LOGGER.info(
        "Finished succeeded=%d failed=%d resumed=%d results=%s",
        succeeded,
        failed,
        skipped,
        args.output,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
