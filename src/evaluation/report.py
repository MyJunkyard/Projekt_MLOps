"""evaluation/report.py — cross-run comparison table (WS6 D10, D12).

``make report`` reads the MLflow tracking store of record and prints a markdown
comparison of runs — the WS6 payoff that makes logged runs *readable back* as
experiments. It is a pure reader: it writes no MLflow data and no files, it only
prints to stdout.

Selection is **explicit-args-first**: a ``run_id`` (one run) wins over the tag
filters; ``series`` / ``exp_id`` become an MLflow filter string on the
``mlflow.series`` / ``mlflow.exp_id`` tags written via ``mlflow.run_tags`` (WS6
D4). No filter at all → the whole experiment, capped (newest-first) so an
accidental full dump stays readable.

``--dm`` is accepted and **rejected** (D12): the Diebold–Mariano hook is one
command family with this table, but DM needs the ``predictions_{split}.parquet``
artifacts and the loss binding that arrive with WS7. Reserving the flag here —
and failing loudly — stops a second ``compare-dm`` command being bolted on
later. See ``docs/contracts.md`` §6.
"""

import argparse
import logging
from collections.abc import Sequence

import mlflow

from src.config import load_config
from src.config.models import PipelineConfig

MODULE_LOGGER_NAME = "src.evaluation.report"
logger = logging.getLogger(MODULE_LOGGER_NAME)

#: Experiment params shown as columns (§2 of the WS6 plan), in order. These are
#: the flat scalars ``training.registry`` logs with every run — the settings
#: that define an experiment, readable back in one table.
REPORT_PARAM_COLUMNS: tuple[str, ...] = (
    "temporal_strategy",
    "derivative_order",
    "derivative_smooth_windows",
    "weather_derivative_order",
    "weather_derivative_smooth_windows",
    "select_groups",
    "select_columns_n",
    "superset_groups",
    "feature_groups",
    "n_features",
    "n_rows_train",
    "n_rows_val",
    "n_rows_test",
)

#: Cap on runs returned when no explicit filter is given, so an accidental full
#: experiment dump stays readable. Newest-first.
DEFAULT_MAX_RESULTS = 50


def _build_filter_string(series: str | None, exp_id: str | None) -> str | None:
    """Build an MLflow ``search_runs`` filter string from the tag selectors.

    ``series`` / ``exp_id`` map to the ``mlflow.series`` / ``mlflow.exp_id``
    tags that ``mlflow.run_tags`` writes (WS6 D4). Both may be combined with
    ``and``.

    Args:
        series: Optional ``series`` tag value to match.
        exp_id: Optional ``exp_id`` tag value to match.

    Returns:
        The filter string, or ``None`` when neither selector is given (no
        filter → the whole experiment).
    """
    clauses = []
    if exp_id is not None:
        clauses.append(f"tags.`mlflow.exp_id` = '{exp_id}'")
    if series is not None:
        clauses.append(f"tags.`mlflow.series` = '{series}'")
    return " and ".join(clauses) if clauses else None


def _select_runs(
    client: mlflow.MlflowClient,
    experiment_id: str,
    *,
    run_id: str | None,
    series: str | None,
    exp_id: str | None,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> list:
    """Select runs from the tracking store, explicit-args-first.

    A ``run_id`` addresses exactly one run and wins over any tag filter. With no
    ``run_id`` the ``series``/``exp_id`` tags filter the experiment (see
    :func:`_build_filter_string`); with no filter at all the whole experiment is
    returned, newest-first and capped.

    Args:
        client: An MLflow tracking client.
        experiment_id: The experiment to search.
        run_id: Optional single run to select (wins over the tag filters).
        series: Optional ``series`` tag value.
        exp_id: Optional ``exp_id`` tag value.
        max_results: Cap when no explicit ``run_id`` is given.

    Returns:
        The selected MLflow run objects, newest-first.
    """
    if run_id is not None:
        run = client.get_run(run_id)
        return [run]

    filter_string = _build_filter_string(series, exp_id)
    return client.search_runs(
        experiment_ids=[experiment_id],
        # MLflow treats an empty filter string as "no filter" (whole
        # experiment); _build_filter_string returns None for that case.
        filter_string=filter_string or "",
        max_results=max_results,
        order_by=["attributes.start_time DESC"],
    )


def _format_row(run, metric_names: Sequence[str]) -> dict[str, str]:
    """Flatten one run into a row of display strings (blank when absent).

    Columns are the run name, the §2 experiment params, and the evaluation
    metrics. A run with no evaluation metrics yet shows blanks for those — the
    table is what *shows* which arms still need ``make evaluate RUN_ID=``.

    Args:
        run: An MLflow run object.
        metric_names: Metric column names to include (from ``evaluation.metrics``).

    Returns:
        A dict of column name → display string.
    """
    row: dict[str, str] = {
        "run_name": run.data.tags.get("mlflow.runName", run.info.run_id)
    }
    for key in REPORT_PARAM_COLUMNS:
        row[key] = run.data.params.get(key, "")
    for metric in metric_names:
        value = run.data.metrics.get(metric)
        row[metric] = "" if value is None else f"{value:.4f}"
    return row


def _render_markdown(rows: list[dict[str, str]], columns: list[str]) -> str:
    """Render the rows as a GitHub-flavoured markdown table.

    Args:
        rows: One dict per run (column name → display string).
        columns: Ordered column names.

    Returns:
        The markdown table as a single string (empty rows → a note).
    """
    if not rows:
        return "_No runs matched the selection._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(col, "")) for col in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def build_report(
    cfg: PipelineConfig,
    run_id: str | None = None,
    series: str | None = None,
    exp_id: str | None = None,
    dm: bool | None = None,
) -> str:
    """Build the cross-run comparison table as a markdown string.

    Args:
        cfg: The validated ``PipelineConfig`` (uses ``mlflow`` for the tracking
            URI/experiment and ``evaluation.metrics`` for the metric columns).
        run_id: Optional single run to report on (wins over the tag filters).
        series: Optional ``series`` tag value to filter on.
        exp_id: Optional ``exp_id`` tag value to filter on.
        dm: Diebold–Mariano request. Reserved for WS7 — any truthy value raises.

    Returns:
        The markdown table.

    Raises:
        NotImplementedError: If ``dm`` is requested (WS7 owns the DM test).
    """
    if dm:
        raise NotImplementedError(
            "--dm (Diebold–Mariano vs a reference run) arrives with WS7: it "
            "needs the predictions_{split}.parquet artifacts and the loss "
            "binding that WS7 introduces. Use `make report` for the run "
            "comparison table in the meantime."
        )

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(cfg.mlflow.experiment_name)
    if experiment is None:
        return (
            f"_Experiment {cfg.mlflow.experiment_name!r} not found — "
            "run `make train` first._"
        )

    runs = _select_runs(
        client,
        experiment.experiment_id,
        run_id=run_id,
        series=series,
        exp_id=exp_id,
    )

    metric_names = list(cfg.evaluation.metrics)
    columns = ["run_name", *REPORT_PARAM_COLUMNS, *metric_names]
    rows = [_format_row(run, metric_names) for run in runs]
    return _render_markdown(rows, columns)


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the ``report`` subcommand."""
    parser = argparse.ArgumentParser(
        prog="src report",
        description="Compare MLflow runs as a markdown table (WS6).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Report a single run by ID (wins over --series/--exp-id).",
    )
    parser.add_argument(
        "--series", default=None, help="Filter runs by the mlflow.series tag."
    )
    parser.add_argument(
        "--exp-id", default=None, help="Filter runs by the mlflow.exp_id tag."
    )
    parser.add_argument(
        "--dm",
        action="store_true",
        default=None,
        help="Reserved for WS7 (Diebold–Mariano). Errors if used.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse args and print the comparison table to stdout."""
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config()
    table = build_report(
        cfg,
        run_id=args.run_id,
        series=args.series,
        exp_id=args.exp_id,
        dm=args.dm,
    )
    print(table)


if __name__ == "__main__":
    main()

