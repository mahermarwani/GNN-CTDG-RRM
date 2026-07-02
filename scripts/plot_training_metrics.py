"""Save plots from TGNN-RRM training metrics CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from tgnn_rrm.plotting import plot_training_metrics


def main() -> None:
    args = parse_args()
    output_paths = plot_training_metrics(args.metrics_csv, args.output_dir)
    for path in output_paths:
        print(path, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/plots"))
    return parser.parse_args()


if __name__ == "__main__":
    main()
