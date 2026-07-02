"""Train and evaluate TGNN-RRM on a SUMO FCD mobility trace."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from tgnn_rrm.config import RadioConfig
from tgnn_rrm.objectives import RRMObjectiveConfig
from tgnn_rrm.simulation import DynamicNetworkConfig
from tgnn_rrm.sumo import load_sumo_fcd_trace
from tgnn_rrm.sumo_training import SumoTrainingMetrics, train_sumo_unsupervised
from tgnn_rrm.tgnn import TGNNConfig


def main() -> None:
    args = parse_args()
    trace = load_sumo_fcd_trace(args.trace)
    print(
        "epoch,phase,avg_loss,avg_mean_rate_bps,avg_qos_penalty,"
        "avg_qos_violation_fraction,avg_active_links,slots,elapsed_seconds,slots_per_second",
        flush=True,
    )
    metrics_writer = _StreamingMetricsWriter(args.metrics_csv)
    result = train_sumo_unsupervised(
        trace=trace,
        network_config=DynamicNetworkConfig(
            max_pair_distance_m=args.max_pair_distance_m,
            link_creation_probability=args.link_creation_probability,
            mean_link_duration_s=args.mean_link_duration_s,
            num_rbs=args.num_rbs,
            shadowing_std_db=args.shadowing_std_db,
            fading_correlation=args.fading_correlation,
            csi_error_std=args.csi_error_std,
            csi_error_correlation=args.csi_error_correlation,
            max_interference_neighbors=None
            if args.max_interference_neighbors <= 0
            else args.max_interference_neighbors,
        ),
        radio_config=RadioConfig(
            num_rbs=args.num_rbs,
            max_rbs_per_link=args.max_rbs_per_link,
            min_rate_bps=args.min_rate_bps,
        ),
        objective_config=RRMObjectiveConfig(
            qos_weight=args.qos_weight,
            rate_scale_bps=args.rate_scale_bps,
        ),
        model_config=TGNNConfig(
            num_rbs=args.num_rbs,
            memory_dim=args.memory_dim,
            message_dim=args.memory_dim,
            embedding_dim=args.memory_dim,
            hidden_dim=args.hidden_dim,
            max_rbs_per_link=args.max_rbs_per_link,
        ),
        epochs=args.epochs,
        train_steps=args.train_steps,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer,
        grad_clip_norm=None if args.grad_clip_norm <= 0.0 else args.grad_clip_norm,
        p_max_watt=args.p_max_watt,
        seed=args.seed,
        device=args.device,
        checkpoint_path=args.checkpoint,
        metrics_callback=metrics_writer.write,
    )
    metrics_writer.close()

    print(f"device={result.device}", file=sys.stderr, flush=True)
    if result.checkpoint_path is not None:
        print(f"checkpoint={result.checkpoint_path}", file=sys.stderr, flush=True)
    if args.metrics_csv is not None:
        print(f"metrics_csv={args.metrics_csv}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Path to a SUMO FCD XML file.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--metrics-csv", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="adam")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-pair-distance-m", type=float, default=200.0)
    parser.add_argument("--link-creation-probability", type=float, default=0.15)
    parser.add_argument("--mean-link-duration-s", type=float, default=10.0)
    parser.add_argument("--num-rbs", type=int, default=5)
    parser.add_argument("--max-rbs-per-link", type=int, default=1)
    parser.add_argument("--p-max-watt", type=float, default=0.0025)
    parser.add_argument("--min-rate-bps", type=float, default=7_000.0)
    parser.add_argument("--qos-weight", type=float, default=1.0)
    parser.add_argument("--rate-scale-bps", type=float, default=1_000.0)
    parser.add_argument("--memory-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--shadowing-std-db", type=float, default=8.0)
    parser.add_argument("--fading-correlation", type=float, default=0.8)
    parser.add_argument("--csi-error-std", type=float, default=0.15)
    parser.add_argument("--csi-error-correlation", type=float, default=0.8)
    parser.add_argument(
        "--max-interference-neighbors",
        type=int,
        default=4,
        help="Keep only the strongest CTDG interference neighbors per link; use 0 for dense all-to-all events.",
    )
    return parser.parse_args()


def _format_metrics(metrics: SumoTrainingMetrics) -> str:
    return (
        f"{metrics.epoch},"
        f"{metrics.phase},"
        f"{metrics.avg_loss:.6f},"
        f"{metrics.avg_mean_rate_bps:.2f},"
        f"{metrics.avg_qos_penalty:.6f},"
        f"{metrics.avg_qos_violation_fraction:.6f},"
        f"{metrics.avg_active_links:.2f},"
        f"{metrics.slots},"
        f"{metrics.elapsed_seconds:.3f},"
        f"{metrics.slots_per_second:.3f}"
    )


class _StreamingMetricsWriter:
    def __init__(self, csv_path: Path | None) -> None:
        self._handle = None
        self._writer = None
        if csv_path is not None:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = csv_path.open("w", newline="")
            self._writer = csv.writer(self._handle)
            self._write_header(self._writer)
            self._handle.flush()

    def write(self, row: SumoTrainingMetrics) -> None:
        print(_format_metrics(row), flush=True)
        if self._writer is not None:
            self._writer.writerow(_metrics_values(row))
            assert self._handle is not None
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _write_header(self, writer: csv.writer) -> None:
        writer.writerow(
            (
                "epoch",
                "phase",
                "avg_loss",
                "avg_mean_rate_bps",
                "avg_qos_penalty",
                "avg_qos_violation_fraction",
                "avg_active_links",
                "slots",
                "elapsed_seconds",
                "slots_per_second",
            )
        )


def _metrics_values(row: SumoTrainingMetrics) -> tuple[object, ...]:
    return (
        row.epoch,
        row.phase,
        row.avg_loss,
        row.avg_mean_rate_bps,
        row.avg_qos_penalty,
        row.avg_qos_violation_fraction,
        row.avg_active_links,
        row.slots,
        row.elapsed_seconds,
        row.slots_per_second,
    )


if __name__ == "__main__":
    main()
