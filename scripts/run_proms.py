"""
Module: args_parser

Defines command-line argument parsing for model testing scripts.

This module provides:
- A reproducible entry point for selecting UCI datasets by name.
- Flexible configuration of data preprocessing (scaling bounds).
- Optional hyperparameter tuning via grid search.

Usage (from project root):
    python model_test/<script_name>.py \
        --name <dataset_name> \
        [--scale --l <lower> --r <upper>] \
        [--gridsearch]

Supported flags:
  --name         Required. One of: breast_cancer, parkinsons,
                 connectionist_bench, spambase
  --scale        Optional. Enable MinMax scaling of features.
  --l            Lower bound for MinMax scaling (default: 0.0)
  --r            Upper bound for MinMax scaling (default: 1.0)
  --gridsearch   Optional. Perform grid search for hyperparameters.

Author: Shuai Zhao
Date: 2025-06-12
"""

import os
import sys
import argparse

# ------------------------------------------------------------------------------
# Project Root Path Setup
# ------------------------------------------------------------------------------
# Prepend the project root to sys.path to allow imports from other modules.
ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.append(ROOT_PATH)


def parse_args() -> argparse.Namespace:
    """
    Parse and validate command-line arguments for dataset selection and model configuration.

    Returns
    -------
    args : argparse.Namespace
        Parsed arguments with attributes:
          name (str):   Key identifying the UCI dataset
                         (choices: breast_cancer, parkinsons,
                                   connectionist_bench, spambase)
          scale (bool):  Flag to enable MinMax scaling.
          l (float):     Lower bound for scaling. Default = 0.0
          r (float):     Upper bound for scaling. Default = 1.0
          gridsearch (bool): Flag to toggle hyperparameter grid search.

    Raises
    ------
    SystemExit
        If required arguments are missing or invalid values are provided.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Configure dataset selection and preprocessing/model tuning options "
            "for UCI benchmark experiments."
        )
    )

    # Dataset selection
    parser.add_argument(
        "--name",
        type=str,
        choices=[
            "breast_cancer",
            "parkinsons",
            "connectionist_bench",
            "spambase",
        ],
        required=True,
        help=(
            "Name of the UCI dataset to fetch and evaluate. "
            "Valid options: breast_cancer, parkinsons, "
            "connectionist_bench, spambase."
        ),
    )

    # Scaling options
    parser.add_argument(
        "--scale",
        action="store_true",
        help=(
            "Enable MinMax scaling of feature values to [l, r] range. "
            "Requires --l and --r bounds if specified."
        ),
    )

    parser.add_argument(
        "--l",
        type=float,
        default=0.0,
        help="Lower bound for MinMax scaling (default: 0.0).",
    )

    parser.add_argument(
        "--r",
        type=float,
        default=1.0,
        help="Upper bound for MinMax scaling (default: 1.0).",
    )

    parser.add_argument(
        "--gridsearch",
        action="store_true",
        help=(
            "Perform exhaustive grid search over predefined hyperparameter grid."
        ),
    )

    return parser.parse_args()