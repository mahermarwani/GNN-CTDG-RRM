"""Unsupervised TGNN training on SUMO-derived mobility traces."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import torch

from tgnn_rrm.config import RadioConfig
from tgnn_rrm.objectives import RRMObjectiveConfig, unsupervised_rrm_loss
from tgnn_rrm.simulation import DynamicNetworkConfig, DynamicNetworkStep
from tgnn_rrm.sumo import SumoD2DSimulator, SumoFCDTrace
from tgnn_rrm.tgnn import TGNNConfig, TGNNResourceAllocator


@dataclass(frozen=True)
class SumoTrainingMetrics:
    """Aggregated metrics for one train or evaluation pass."""

    epoch: int
    phase: str
    avg_loss: float
    avg_mean_rate_bps: float
    avg_qos_penalty: float
    avg_qos_violation_fraction: float
    avg_active_links: float
    slots: int
    elapsed_seconds: float
    slots_per_second: float


@dataclass(frozen=True)
class SumoTrainingResult:
    """Model and metrics produced by SUMO unsupervised training."""

    model: TGNNResourceAllocator
    train_metrics: tuple[SumoTrainingMetrics, ...]
    eval_metrics: tuple[SumoTrainingMetrics, ...]
    device: torch.device
    checkpoint_path: Path | None = None


def train_sumo_unsupervised(
    trace: SumoFCDTrace,
    network_config: DynamicNetworkConfig,
    radio_config: RadioConfig,
    objective_config: RRMObjectiveConfig | None = None,
    model_config: TGNNConfig | None = None,
    epochs: int = 5,
    train_steps: int = 100,
    eval_steps: int = 0,
    learning_rate: float = 1e-3,
    optimizer_name: str = "sgd",
    grad_clip_norm: float | None = None,
    p_max_watt: float = 0.0025,
    seed: int = 7,
    device: str | torch.device = "auto",
    checkpoint_path: str | Path | None = None,
    metrics_callback: Callable[[SumoTrainingMetrics], None] | None = None,
) -> SumoTrainingResult:
    """Train a TGNN with the unsupervised RRM objective on SUMO frames."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    if eval_steps < 0:
        raise ValueError("eval_steps must be non-negative")
    if grad_clip_norm is not None and grad_clip_norm <= 0.0:
        raise ValueError("grad_clip_norm must be positive when set")

    torch.manual_seed(seed)
    resolved_device = _resolve_device(device)
    objective = objective_config or RRMObjectiveConfig()
    model_cfg = model_config or TGNNConfig(num_rbs=radio_config.num_rbs)
    model = TGNNResourceAllocator(model_cfg).to(resolved_device)
    optimizer = _build_optimizer(model, optimizer_name=optimizer_name, learning_rate=learning_rate)
    sequence = _sumo_sequence(trace, network_config, train_steps + eval_steps, seed=seed)
    train_sequence = sequence[:train_steps]
    eval_sequence = sequence[train_steps:] if eval_steps else ()

    train_metrics: list[SumoTrainingMetrics] = []
    eval_metrics: list[SumoTrainingMetrics] = []
    for epoch in range(1, epochs + 1):
        model.reset_memory()
        train_row = _run_sequence(
            model=model,
            sequence=train_sequence,
            radio_config=radio_config,
            objective_config=objective,
            optimizer=optimizer,
            grad_clip_norm=grad_clip_norm,
            p_max_watt=p_max_watt,
            device=resolved_device,
            epoch=epoch,
            phase="train",
        )
        train_metrics.append(train_row)
        if metrics_callback is not None:
            metrics_callback(train_row)
        if eval_sequence:
            model.reset_memory()
            eval_row = _run_sequence(
                model=model,
                sequence=eval_sequence,
                radio_config=radio_config,
                objective_config=objective,
                optimizer=None,
                grad_clip_norm=None,
                p_max_watt=p_max_watt,
                device=resolved_device,
                epoch=epoch,
                phase="eval",
            )
            eval_metrics.append(eval_row)
            if metrics_callback is not None:
                metrics_callback(eval_row)
    normalized_checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if normalized_checkpoint is not None:
        _save_checkpoint(
            path=normalized_checkpoint,
            model=model,
            model_config=model_cfg,
            network_config=network_config,
            radio_config=radio_config,
            objective_config=objective,
            train_metrics=tuple(train_metrics),
            eval_metrics=tuple(eval_metrics),
            device=resolved_device,
            seed=seed,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            grad_clip_norm=grad_clip_norm,
            p_max_watt=p_max_watt,
        )
    return SumoTrainingResult(
        model=model,
        train_metrics=tuple(train_metrics),
        eval_metrics=tuple(eval_metrics),
        device=resolved_device,
        checkpoint_path=normalized_checkpoint,
    )


def _sumo_sequence(
    trace: SumoFCDTrace,
    network_config: DynamicNetworkConfig,
    steps: int,
    seed: int,
) -> tuple[DynamicNetworkStep, ...]:
    simulator = SumoD2DSimulator(trace, config=network_config, seed=seed)
    return simulator.run(steps)


def _run_sequence(
    model: TGNNResourceAllocator,
    sequence: tuple[DynamicNetworkStep, ...],
    radio_config: RadioConfig,
    objective_config: RRMObjectiveConfig,
    optimizer: torch.optim.Optimizer | None,
    grad_clip_norm: float | None,
    p_max_watt: float,
    device: torch.device,
    epoch: int,
    phase: str,
) -> SumoTrainingMetrics:
    losses: list[float] = []
    mean_rates: list[float] = []
    qos_penalties: list[float] = []
    qos_violation_fractions: list[float] = []
    active_links: list[int] = []

    grad_enabled = optimizer is not None
    start = perf_counter()
    with torch.set_grad_enabled(grad_enabled):
        for step in sequence:
            if not step.snapshot.active_links:
                continue
            if optimizer is not None:
                optimizer.zero_grad()
            output = model(step.events, p_max=p_max_watt)
            result = unsupervised_rrm_loss(
                gains=step.snapshot.gains.to(device),
                allocation=output.rb_probabilities,
                power=output.power,
                radio_config=radio_config,
                p_max=p_max_watt,
                objective_config=objective_config,
            )
            if optimizer is not None:
                result.loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                model.detach_memory()

            losses.append(float(result.loss.detach()))
            mean_rates.append(float(result.mean_rate.detach()))
            qos_penalties.append(float(result.qos_penalty.detach()))
            qos_violation_fractions.append(_qos_violation_fraction(result.rates.detach(), radio_config.min_rate_bps))
            active_links.append(len(step.snapshot.active_links))

    elapsed = perf_counter() - start
    slots = len(losses)
    return SumoTrainingMetrics(
        epoch=epoch,
        phase=phase,
        avg_loss=_mean(losses),
        avg_mean_rate_bps=_mean(mean_rates),
        avg_qos_penalty=_mean(qos_penalties),
        avg_qos_violation_fraction=_mean(qos_violation_fractions),
        avg_active_links=_mean(active_links),
        slots=slots,
        elapsed_seconds=elapsed,
        slots_per_second=slots / elapsed if elapsed > 0.0 else 0.0,
    )


def _mean(values: list[float] | list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _qos_violation_fraction(rates: torch.Tensor, min_rate_bps: float) -> float:
    if rates.numel() == 0:
        return 0.0
    return float(torch.mean((rates < min_rate_bps).to(dtype=rates.dtype)))


def _build_optimizer(
    model: TGNNResourceAllocator,
    optimizer_name: str,
    learning_rate: float,
) -> torch.optim.Optimizer:
    normalized = optimizer_name.lower()
    if normalized == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    if normalized == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    raise ValueError("optimizer_name must be 'sgd' or 'adam'")


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        requested = device
    elif device == "auto":
        requested = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return requested


def _save_checkpoint(
    path: Path,
    model: TGNNResourceAllocator,
    model_config: TGNNConfig,
    network_config: DynamicNetworkConfig,
    radio_config: RadioConfig,
    objective_config: RRMObjectiveConfig,
    train_metrics: tuple[SumoTrainingMetrics, ...],
    eval_metrics: tuple[SumoTrainingMetrics, ...],
    device: torch.device,
    seed: int,
    optimizer_name: str,
    learning_rate: float,
    grad_clip_norm: float | None,
    p_max_watt: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "network_config": network_config,
            "radio_config": radio_config,
            "objective_config": objective_config,
            "train_metrics": tuple(asdict(metrics) for metrics in train_metrics),
            "eval_metrics": tuple(asdict(metrics) for metrics in eval_metrics),
            "device": str(device),
            "seed": seed,
            "optimizer_name": optimizer_name,
            "learning_rate": learning_rate,
            "grad_clip_norm": grad_clip_norm,
            "p_max_watt": p_max_watt,
        },
        path,
    )
