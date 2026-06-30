"""Configuration objects for wireless resource-management experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RadioConfig:
    """Radio parameters used by the D2D rate and constraint equations."""

    num_rbs: int = 5
    rb_bandwidth_hz: float = 500.0
    noise_power_watt: float = 10 ** ((-142.0 - 30.0) / 10.0)
    max_rbs_per_link: int = 1
    min_rate_bps: float = 7_000.0

    def validate(self) -> None:
        """Raise ``ValueError`` when the configuration is physically invalid."""

        if self.num_rbs <= 0:
            raise ValueError("num_rbs must be positive")
        if self.rb_bandwidth_hz <= 0:
            raise ValueError("rb_bandwidth_hz must be positive")
        if self.noise_power_watt <= 0:
            raise ValueError("noise_power_watt must be positive")
        if not 1 <= self.max_rbs_per_link <= self.num_rbs:
            raise ValueError("max_rbs_per_link must be in [1, num_rbs]")
        if self.min_rate_bps < 0:
            raise ValueError("min_rate_bps must be non-negative")
