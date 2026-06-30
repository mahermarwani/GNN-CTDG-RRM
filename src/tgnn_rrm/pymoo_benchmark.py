"""PyMOO-based benchmark label generation for RRM snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from numbers import Real
from typing import Sequence

import torch

from tgnn_rrm.config import RadioConfig
from tgnn_rrm.objectives import torch_link_rates


@dataclass(frozen=True)
class PymooBenchmarkConfig:
    """Genetic-search settings for one CSI snapshot."""

    population_size: int = 64
    generations: int = 80
    seed: int | None = None
    p_max_watt: float = 10 ** ((4.0 - 30.0) / 10.0)
    qos_penalty_weight: float = 1.0
    rate_scale_bps: float = 1_000.0

    def validate(self) -> None:
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if self.generations <= 0:
            raise ValueError("generations must be positive")
        if self.p_max_watt <= 0:
            raise ValueError("p_max_watt must be positive")
        if self.qos_penalty_weight < 0:
            raise ValueError("qos_penalty_weight must be non-negative")
        if self.rate_scale_bps <= 0:
            raise ValueError("rate_scale_bps must be positive")


@dataclass(frozen=True)
class PymooRRMResult:
    """Near-optimal allocation label for one CSI snapshot."""

    allocation: torch.Tensor
    power: torch.Tensor
    rates: torch.Tensor
    mean_rate: float
    objective: float
    qos_violation_fraction: float


def is_pymoo_available() -> bool:
    """Return whether the optional PyMOO dependency is importable."""

    return find_spec("pymoo") is not None


def solve_snapshot_with_pymoo(
    gains: torch.Tensor,
    radio_config: RadioConfig,
    benchmark_config: PymooBenchmarkConfig | None = None,
    p_max: torch.Tensor | float | None = None,
) -> PymooRRMResult:
    """Solve one CSI snapshot with a PyMOO genetic algorithm.

    The candidate vector contains RB scores and per-RB power fractions. RB
    scores are converted to top-``L_max`` binary allocations per link, while
    selected power fractions are scaled down only if they exceed per-link
    power budgets.
    This keeps every candidate feasible for constraints C.2-C.4 and lets the
    genetic search focus on rate and QoS.
    """

    _require_pymoo()
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    config = benchmark_config or PymooBenchmarkConfig()
    config.validate()
    radio_config.validate()
    gains = _validate_gains(gains, radio_config)
    num_links = gains.shape[0]
    num_rbs = gains.shape[2]
    if num_links == 0:
        empty = torch.empty((0, num_rbs), dtype=gains.dtype, device=gains.device)
        return PymooRRMResult(
            allocation=empty,
            power=empty,
            rates=torch.empty(0, dtype=gains.dtype, device=gains.device),
            mean_rate=0.0,
            objective=0.0,
            qos_violation_fraction=0.0,
        )

    p_max_tensor = _p_max_vector(p_max if p_max is not None else config.p_max_watt, num_links, gains.device, gains.dtype)
    variable_count = 2 * num_links * num_rbs

    class SnapshotProblem(ElementwiseProblem):
        def __init__(self) -> None:
            super().__init__(n_var=variable_count, n_obj=1, xl=0.0, xu=1.0)

        def _evaluate(self, candidate, out, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            allocation, power = candidate_to_allocation_power(
                candidate=candidate,
                num_links=num_links,
                num_rbs=num_rbs,
                max_rbs_per_link=radio_config.max_rbs_per_link,
                p_max=p_max_tensor,
                dtype=gains.dtype,
                device=gains.device,
            )
            rates = torch_link_rates(gains, allocation, power, radio_config)
            mean_rate = rates.mean()
            qos_gap = torch.relu(radio_config.min_rate_bps - rates) / config.rate_scale_bps
            objective = -mean_rate / config.rate_scale_bps + config.qos_penalty_weight * torch.mean(qos_gap.pow(2))
            out["F"] = float(objective.detach())

    result = minimize(
        SnapshotProblem(),
        GA(pop_size=config.population_size),
        termination=get_termination("n_gen", config.generations),
        seed=config.seed,
        verbose=False,
    )
    allocation, power = candidate_to_allocation_power(
        candidate=result.X,
        num_links=num_links,
        num_rbs=num_rbs,
        max_rbs_per_link=radio_config.max_rbs_per_link,
        p_max=p_max_tensor,
        dtype=gains.dtype,
        device=gains.device,
    )
    rates = torch_link_rates(gains, allocation, power, radio_config)
    mean_rate = float(rates.mean()) if rates.numel() else 0.0
    qos_fraction = float(torch.mean((rates < radio_config.min_rate_bps).to(dtype=gains.dtype))) if rates.numel() else 0.0
    return PymooRRMResult(
        allocation=allocation,
        power=power,
        rates=rates,
        mean_rate=mean_rate,
        objective=float(result.F[0]) if getattr(result, "F", None) is not None else float("nan"),
        qos_violation_fraction=qos_fraction,
    )


def solve_snapshots_with_pymoo(
    gains_sequence: Sequence[torch.Tensor],
    radio_config: RadioConfig,
    benchmark_config: PymooBenchmarkConfig | None = None,
    p_max: torch.Tensor | float | None = None,
) -> tuple[PymooRRMResult, ...]:
    """Solve a chronological sequence of independent CSI snapshots."""

    return tuple(
        solve_snapshot_with_pymoo(
            gains=gains,
            radio_config=radio_config,
            benchmark_config=benchmark_config,
            p_max=p_max,
        )
        for gains in gains_sequence
    )


def candidate_to_allocation_power(
    candidate: Sequence[float],
    num_links: int,
    num_rbs: int,
    max_rbs_per_link: int,
    p_max: torch.Tensor | float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a genetic-search candidate into feasible RB and power matrices."""

    if num_links < 0:
        raise ValueError("num_links must be non-negative")
    if num_rbs <= 0:
        raise ValueError("num_rbs must be positive")
    if not 1 <= max_rbs_per_link <= num_rbs:
        raise ValueError("max_rbs_per_link must be in [1, num_rbs]")
    expected_length = 2 * num_links * num_rbs
    if len(candidate) != expected_length:
        raise ValueError("candidate length must be 2 * num_links * num_rbs")

    tensor = torch.as_tensor(candidate, dtype=dtype, device=device).reshape(2, num_links, num_rbs)
    scores = tensor[0]
    fractions = tensor[1].clamp(0.0, 1.0)
    allocation = torch.zeros((num_links, num_rbs), dtype=dtype, device=tensor.device)
    if num_links == 0:
        return allocation, allocation.clone()

    selected = torch.topk(scores, k=max_rbs_per_link, dim=1).indices
    allocation.scatter_(1, selected, 1.0)
    raw_power = allocation * fractions
    power_sum = raw_power.sum(dim=1, keepdim=True)
    scale = torch.where(power_sum > 1.0, power_sum.clamp_min(1e-12).reciprocal(), torch.ones_like(power_sum))
    normalized_power = raw_power * scale
    p_max_vector = _p_max_vector(p_max, num_links, tensor.device, dtype).unsqueeze(1)
    power = normalized_power * p_max_vector
    return allocation, power


def _require_pymoo() -> None:
    if not is_pymoo_available():
        raise RuntimeError("PyMOO is required for benchmark generation. Install with `pip install .[benchmark]`.")


def _validate_gains(gains: torch.Tensor, config: RadioConfig) -> torch.Tensor:
    if not isinstance(gains, torch.Tensor):
        raise TypeError("gains must be a torch.Tensor")
    if gains.ndim != 3:
        raise ValueError("gains must have shape [num_links, num_links, num_rbs]")
    if gains.shape[0] != gains.shape[1]:
        raise ValueError("gains must have square link dimensions")
    if gains.shape[2] != config.num_rbs:
        raise ValueError("gains RB dimension must match radio_config.num_rbs")
    return gains if gains.is_floating_point() else gains.to(dtype=torch.float32)


def _p_max_vector(
    p_max: torch.Tensor | float,
    num_links: int,
    device: torch.device | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(p_max, Real):
        return torch.full((num_links,), float(p_max), device=device, dtype=dtype)
    tensor = p_max.to(device=device, dtype=dtype)
    if tensor.ndim == 0:
        return tensor.expand(num_links)
    if tensor.ndim == 1 and tensor.shape[0] == num_links:
        return tensor
    raise ValueError("p_max must be scalar or [num_links]")
