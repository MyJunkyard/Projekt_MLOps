"""
cli.py — Single entry point for all pipeline commands.

Usage: ``python -m src <command>`` with one of:

- ``ingest``    — download/validate/save raw data (src.ingestion.main)
- ``featurise`` — engineer features and save splits (src.features.main)
- ``train``     — train model + baselines, log to MLflow (src.training.main)
- ``evaluate``  — evaluate champion model on the test split
                  (src.evaluation.main)

Pipeline order is expressed here (and in the Makefile), not in the
directory tree. Extensible in later stages (``report``, ``promote``).
"""

import argparse

from src.evaluation.main import main as evaluate_main
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
    subparsers.add_parser(
        "evaluate", help="Evaluate the champion model on the test split."
    )

    return parser


COMMANDS = {
    "ingest": ingest_main,
    "featurise": featurise_main,
    "train": train_main,
    "evaluate": evaluate_main,
}


def main() -> None:
    """Parse the command line and dispatch to the selected stage's main()."""
    parser = build_parser()
    args = parser.parse_args()
    COMMANDS[args.command]()


if __name__ == "__main__":
    main()
