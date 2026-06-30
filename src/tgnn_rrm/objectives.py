"""Differentiable RRM objectives for TGNN training."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import torch

from tgnn_rrm.config import RadioConfig


@dataclass(frozen=True)
class RRMObjectiveConfig:
    """Weights for the unsupervised TGNN-RRM objective."""

    rate_weight: float = 1.0
    qos_weight: float = 1.0
    rb_budget_weight: float = 1.0
    power_budget_weight: float = 1.0
    inactive_power_weight: float = 1.0
    negative_power_weight: float = 1.0
    allocation_bound_weight: float = 0.1
    rate_scale_bps: float = 1_000.0

    def validate(self) -> None:
        weights = (
            self.rate_weight,
            self.qos_weight,
            self.rb_budget_weight,
            self.power_budget_weight,
            self.inactive_power_weight,
            self.negative_power_weight,
            self.allocation_bound_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("objective weights must be non-negative")
        if self.rate_scale_bps <= 0.0:
            raise ValueError("rate_scale_bps must be positive")


@dataclass(frozen=True)
class RRMObjectiveResult:
    """Loss value and diagnostics for one allocation decision."""

    loss: torch.Tensor
    rates: torch.Tensor
    mean_rate: torch.Tensor
    qos_penalty: torch.Tensor
    rb_budget_penalty: torch.Tensor
    power_budget_penalty: torch.Tensor
    inactive_power_penalty: torch.Tensor
    negative_power_penalty: torch.Tensor
    allocation_bound_penalty: torch.Tensor


def torch_link_rates(
    gains: torch.Tensor,
    allocation: torch.Tensor,
    power: torch.Tensor,
    config: RadioConfig,
) -> torch.Tensor:
    """Compute differentiable per-link rates from Eq. (rate-multiRB).

    ``gains[i, j, k]`` is the gain from transmitter ``j`` to receiver ``i`` on
    RB ``k``. ``allocation`` may contain hard binary decisions or soft
    probabilities during training.
    """

    config.validate()
    gains, allocation, power = _validate_rate_tensors(gains, allocation, power, config)
    num_links = gains.shape[0]
    if num_links == 0:
        return torch.empty(0, device=gains.device, dtype=gains.dtype)

    link_indices = torch.arange(num_links, device=gains.device)
    direct_gains = gains[link_indices, link_indices, :]
    transmitted_power = allocation * power
    signal = transmitted_power * direct_gains
    total_received = torch.sum(gains * transmitted_power.unsqueeze(0), dim=1)
    interference = (total_received - signal).clamp_min(0.0)
    sinr = signal / (config.noise_power_watt + interference)
    log2 = torch.log(torch.tensor(2.0, device=gains.device, dtype=gains.dtype))
    return config.rb_bandwidth_hz * torch.sum(torch.log1p(sinr.clamp_min(0.0)) / log2, dim=1)


def torch_mean_rate(
    gains: torch.Tensor,
    allocation: torch.Tensor,
    power: torch.Tensor,
    config: RadioConfig,
) -> torch.Tensor:
    """Return the differentiable mean active-link rate."""

    rates = torch_link_rates(gains, allocation, power, config)
    if rates.numel() == 0:
        return torch.zeros((), device=gains.device, dtype=gains.dtype)
    return rates.mean()


def unsupervised_rrm_loss(
    gains: torch.Tensor,
    allocation: torch.Tensor,
    power: torch.Tensor,
    radio_config: RadioConfig,
    p_max: torch.Tensor | float | None = None,
    objective_config: RRMObjectiveConfig | None = None,
) -> RRMObjectiveResult:
    """Compute an unsupervised objective from rate maximization and constraints."""

    objective = objective_config or RRMObjectiveConfig()
    objective.validate()
    rates = torch_link_rates(gains, allocation, power, radio_config)
    gains, allocation, power = _validate_rate_tensors(gains, allocation, power, radio_config)
    p_max_matrix = _p_max_matrix(p_max, allocation.shape[0], allocation.shape[1], allocation.device, allocation.dtype)
    p_max_vector = (
        p_max_matrix.max(dim=1).values
        if p_max_matrix.numel()
        else torch.empty(0, device=allocation.device, dtype=allocation.dtype)
    )
    zero = torch.zeros((), device=allocation.device, dtype=allocation.dtype)

    if rates.numel() == 0:
        mean_rate = zero
        qos_penalty = zero
    else:
        mean_rate = rates.mean()
        qos_gap = torch.relu(radio_config.min_rate_bps - rates) / objective.rate_scale_bps
        qos_penalty = torch.mean(qos_gap.pow(2))

    rb_budget_penalty = _mean_square_relu(allocation.sum(dim=1) - radio_config.max_rbs_per_link)
    power_budget_penalty = _mean_square_relu(power.sum(dim=1) - p_max_vector)
    inactive_power_penalty = _mean_square_relu(power - allocation * p_max_matrix)
    negative_power_penalty = _mean_square_relu(-power)
    allocation_bound_penalty = _mean_square_relu(-allocation) + _mean_square_relu(allocation - 1.0)

    normalized_rate = mean_rate / objective.rate_scale_bps
    loss = (
        -objective.rate_weight * normalized_rate
        + objective.qos_weight * qos_penalty
        + objective.rb_budget_weight * rb_budget_penalty
        + objective.power_budget_weight * power_budget_penalty
        + objective.inactive_power_weight * inactive_power_penalty
        + objective.negative_power_weight * negative_power_penalty
        + objective.allocation_bound_weight * allocation_bound_penalty
    )
    return RRMObjectiveResult(
        loss=loss,
        rates=rates,
        mean_rate=mean_rate,
        qos_penalty=qos_penalty,
        rb_budget_penalty=rb_budget_penalty,
        power_budget_penalty=power_budget_penalty,
        inactive_power_penalty=inactive_power_penalty,
        negative_power_penalty=negative_power_penalty,
        allocation_bound_penalty=allocation_bound_penalty,
    )


def _validate_rate_tensors(
    gains: torch.Tensor,
    allocation: torch.Tensor,
    power: torch.Tensor,
    config: RadioConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(gains, torch.Tensor):
        raise TypeError("gains must be a torch.Tensor")
    if not isinstance(allocation, torch.Tensor):
        raise TypeError("allocation must be a torch.Tensor")
    if not isinstance(power, torch.Tensor):
        raise TypeError("power must be a torch.Tensor")
    if gains.ndim != 3:
        raise ValueError("gains must have shape [num_links, num_links, num_rbs]")
    if allocation.ndim != 2 or power.ndim != 2:
        raise ValueError("allocation and power must have shape [num_links, num_rbs]")
    if gains.shape[0] != gains.shape[1]:
        raise ValueError("gains must have square link dimensions")
    expected_shape = (gains.shape[0], config.num_rbs)
    if allocation.shape != expected_shape or power.shape != expected_shape:
        raise ValueError("allocation and power shapes must match gains and num_rbs")
    if gains.shape[2] != config.num_rbs:
        raise ValueError("gains RB dimension must match num_rbs")
    dtype = gains.dtype if gains.is_floating_point() else torch.float32
    return (
        gains.to(dtype=dtype),
        allocation.to(device=gains.device, dtype=dtype),
        power.to(device=gains.device, dtype=dtype),
    )


def _p_max_matrix(
    p_max: torch.Tensor | float | None,
    num_links: int,
    num_rbs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if p_max is None:
        return torch.ones((num_links, num_rbs), device=device, dtype=dtype)
    if isinstance(p_max, Real):
        return torch.full((num_links, num_rbs), float(p_max), device=device, dtype=dtype)

    tensor = p_max.to(device=device, dtype=dtype)
    if tensor.ndim == 0:
        return tensor.expand(num_links, num_rbs)
    if tensor.ndim == 1 and tensor.shape[0] == num_links:
        return tensor.unsqueeze(1).expand(num_links, num_rbs)
    if tensor.shape == (num_links, num_rbs):
        return tensor
    raise ValueError("p_max must be scalar, [num_links], or [num_links, num_rbs]")


def _mean_square_relu(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    return torch.mean(torch.relu(values).pow(2))
