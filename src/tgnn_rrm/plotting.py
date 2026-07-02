"""Plot utilities for TGNN-RRM training metrics."""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


@dataclass(frozen=True)
class TrainingMetricRow:
    """One CSV row emitted by SUMO unsupervised training."""

    epoch: int
    phase: str
    avg_loss: float
    avg_mean_rate_bps: float
    avg_qos_penalty: float
    avg_qos_violation_fraction: float | None
    avg_active_links: float
    slots: int
    elapsed_seconds: float
    slots_per_second: float


def plot_training_metrics(metrics_csv: str | Path, output_dir: str | Path) -> tuple[Path, ...]:
    """Save standard training plots from a metrics CSV file."""

    rows = load_training_metrics(metrics_csv)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("avg_loss", "Loss", "loss.png"),
        ("avg_mean_rate_bps", "Mean rate (bps)", "mean_rate_bps.png"),
        ("avg_qos_penalty", "QoS penalty", "qos_penalty.png"),
        ("avg_active_links", "Active links", "active_links.png"),
        ("slots_per_second", "Slots per second", "slots_per_second.png"),
    ]
    if any(row.avg_qos_violation_fraction is not None for row in rows):
        plot_specs.insert(3, ("avg_qos_violation_fraction", "QoS violation fraction", "qos_violation_fraction.png"))
    return tuple(
        _plot_metric(rows, field_name=field_name, ylabel=ylabel, output_path=target_dir / filename)
        for field_name, ylabel, filename in plot_specs
    )


def load_training_metrics(metrics_csv: str | Path) -> tuple[TrainingMetricRow, ...]:
    """Load metrics rows emitted by ``train_sumo_unsupervised.py``."""

    with Path(metrics_csv).open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = tuple(_row_from_dict(row) for row in reader)
    if not rows:
        raise ValueError("metrics CSV does not contain any rows")
    return rows


def _plot_metric(
    rows: tuple[TrainingMetricRow, ...],
    field_name: str,
    ylabel: str,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for phase in _ordered_phases(rows):
        phase_rows = [row for row in rows if row.phase == phase]
        ax.plot(
            [row.epoch for row in phase_rows],
            [_metric_value(row, field_name) for row in phase_rows],
            marker="o",
            linewidth=1.6,
            label=phase,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _ordered_phases(rows: tuple[TrainingMetricRow, ...]) -> tuple[str, ...]:
    phases = []
    for row in rows:
        if row.phase not in phases:
            phases.append(row.phase)
    return tuple(phases)


def _row_from_dict(row: dict[str, str]) -> TrainingMetricRow:
    return TrainingMetricRow(
        epoch=int(row["epoch"]),
        phase=row["phase"],
        avg_loss=float(row["avg_loss"]),
        avg_mean_rate_bps=float(row["avg_mean_rate_bps"]),
        avg_qos_penalty=float(row["avg_qos_penalty"]),
        avg_qos_violation_fraction=_optional_float(row, "avg_qos_violation_fraction"),
        avg_active_links=float(row["avg_active_links"]),
        slots=int(row["slots"]),
        elapsed_seconds=float(row["elapsed_seconds"]),
        slots_per_second=float(row["slots_per_second"]),
    )


def _metric_value(row: TrainingMetricRow, field_name: str) -> float:
    value = getattr(row, field_name)
    return math.nan if value is None else float(value)


def _optional_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)
