"""Core wireless equations from the TGNN-RRM manuscript."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Sequence

from tgnn_rrm.config import RadioConfig

Matrix = Sequence[Sequence[float]]
Tensor3 = Sequence[Sequence[Sequence[float]]]


@dataclass(frozen=True)
class ConstraintViolations:
    """Counts and magnitudes of allocation constraint violations."""

    rb_budget: int
    power_budget: int
    inactive_power: int
    negative_power: int

    @property
    def total(self) -> int:
        return self.rb_budget + self.power_budget + self.inactive_power + self.negative_power


def link_rates(
    gains: Tensor3,
    allocation: Matrix,
    power: Matrix,
    config: RadioConfig,
) -> list[float]:
    """Compute per-link rates using the paper's multi-RB SINR equation.

    ``gains[i][j][k]`` is the channel gain from transmitter ``j`` to receiver
    ``i`` on RB ``k``. The direct channel of link ``i`` is ``gains[i][i][k]``.
    """

    config.validate()
    num_links = _validate_shapes(gains, allocation, power, config)
    rates: list[float] = []

    for i in range(num_links):
        rate = 0.0
        for k in range(config.num_rbs):
            signal = allocation[i][k] * power[i][k] * gains[i][i][k]
            interference = 0.0
            for j in range(num_links):
                if j != i:
                    interference += allocation[j][k] * power[j][k] * gains[i][j][k]
            sinr = signal / (config.noise_power_watt + interference)
            rate += config.rb_bandwidth_hz * log2(1.0 + sinr)
        rates.append(rate)

    return rates


def mean_rate(
    gains: Tensor3,
    allocation: Matrix,
    power: Matrix,
    config: RadioConfig,
) -> float:
    """Return the average active-link rate."""

    rates = link_rates(gains, allocation, power, config)
    return sum(rates) / len(rates) if rates else 0.0


def qos_violation_fraction(rates: Sequence[float], min_rate_bps: float) -> float:
    """Return the fraction of links whose rates are below the QoS target."""

    if not rates:
        return 0.0
    return sum(rate < min_rate_bps for rate in rates) / len(rates)


def constraint_violations(
    allocation: Matrix,
    power: Matrix,
    p_max: Sequence[float],
    config: RadioConfig,
    tol: float = 1e-12,
) -> ConstraintViolations:
    """Check constraints C.2-C.4 from the manuscript."""

    config.validate()
    if len(allocation) != len(power) or len(power) != len(p_max):
        raise ValueError("allocation, power, and p_max must have matching link counts")

    rb_budget = 0
    power_budget = 0
    inactive_power = 0
    negative_power = 0

    for i, (x_row, p_row) in enumerate(zip(allocation, power)):
        if len(x_row) != config.num_rbs or len(p_row) != config.num_rbs:
            raise ValueError("allocation and power rows must match num_rbs")
        if sum(1 for value in x_row if value > 0.5) > config.max_rbs_per_link:
            rb_budget += 1
        if sum(p_row) > p_max[i] + tol:
            power_budget += 1
        for x_ik, p_ik in zip(x_row, p_row):
            if p_ik < -tol:
                negative_power += 1
            if x_ik <= 0.5 and p_ik > tol:
                inactive_power += 1
            if p_ik > p_max[i] + tol:
                power_budget += 1

    return ConstraintViolations(
        rb_budget=rb_budget,
        power_budget=power_budget,
        inactive_power=inactive_power,
        negative_power=negative_power,
    )


def _validate_shapes(
    gains: Tensor3,
    allocation: Matrix,
    power: Matrix,
    config: RadioConfig,
) -> int:
    num_links = len(gains)
    if num_links == 0:
        return 0
    if len(allocation) != num_links or len(power) != num_links:
        raise ValueError("gains, allocation, and power must have matching link counts")

    for i in range(num_links):
        if len(gains[i]) != num_links:
            raise ValueError("gains must have shape [num_links][num_links][num_rbs]")
        if len(allocation[i]) != config.num_rbs or len(power[i]) != config.num_rbs:
            raise ValueError("allocation and power must have shape [num_links][num_rbs]")
        for j in range(num_links):
            if len(gains[i][j]) != config.num_rbs:
                raise ValueError("gains must have shape [num_links][num_links][num_rbs]")
    return num_links
