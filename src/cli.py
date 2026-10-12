"""
cli.py — Single entry point for all pipeline commands.

Usage: ``python -m src <command>`` with one of:

- ``ingest``    — download/validate/save raw data (src.ingestion.main)
- ``featurise`` — engineer features and save splits (src.features.main)
- ``train``     — train model + baselines, log to MLflow (src.training.main)
- ``evaluate``  — evaluate the champion model on the test split (or a
                  specific run via --run-id) (src.evaluation.main)
- ``report``    — compare MLflow runs as a markdown table (src.evaluation.report)

Pipeline order is expressed here (and in the Makefile), not in the
directory tree. Extensible in later stages (``promote``).
"""

import argparse

from src.evaluation.main import main as evaluate_main
from src.evaluation.report import main as report_main
from src.features.main import main as featurise_main
from src.ingestion.main import main as ingest_main
from src.training.main import main as train_main


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with one subcommand per pipeline stage."""
    parser = argparse.ArgumentParser(
        prog="src",
        description="Energy price forecasting pipeline (MLOps project).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "ingest", help="Download (or synthesize), validate, and save raw data."
    )
    subparsers.add_parser(
        "featurise", help="Engineer calendar/lag/rolling/derivative features."
    )
    subparsers.add_parser(
        "train", help="Train the model plus baselines and log to MLflow."
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate the champion model (or a given run) on the test split.",
    )
    evaluate_parser.add_argument(
        "--run-id",
        default=None,
        help="Score this MLflow run on the test split instead of the champion.",
    )

    report_parser = subparsers.add_parser(
        "report", help="Compare MLflow runs as a markdown table (WS6)."
    )
    report_parser.add_argument(
        "--run-id", default=None, help="Report a single run (wins over filters)."
    )
    report_parser.add_argument(
        "--series", default=None, help="Filter runs by the mlflow.series tag."
    )
    report_parser.add_argument(
        "--exp-id", default=None, help="Filter runs by the mlflow.exp_id tag."
    )
    report_parser.add_argument(
        "--dm",
        action="store_true",
        default=None,
        help="Reserved for WS7 (Diebold–Mariano). Errors if used.",
    )

    return parser


COMMANDS = {
    "ingest": ingest_main,
    "featurise": featurise_main,
    "train": train_main,
    "evaluate": evaluate_main,
    "report": report_main,
}


def main() -> None:
    """Parse the command line and dispatch to the selected stage's main().

    ``evaluate`` and ``report`` are the only mains that take an optional
    ``run_id``; it is passed through as a keyword so the other stages keep
    their zero-arg signatures.
    """
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate_main(run_id=args.run_id)
    elif args.command == "report":
        report_main(
            argv=[
                *( ["--run-id", args.run_id] if args.run_id else [] ),
                *(["--series", args.series] if args.series else []),
                *(["--exp-id", args.exp_id] if args.exp_id else []),
                *(["--dm"] if args.dm else []),
            ]
        )
    else:
        COMMANDS[args.command]()


if __name__ == "__main__":
    main()
