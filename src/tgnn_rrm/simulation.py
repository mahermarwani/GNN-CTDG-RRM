"""Dynamic D2D network simulation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi
from typing import Iterable

import torch

from tgnn_rrm.ctdg import CTDGEventBatch, build_ctdg_events


@dataclass(frozen=True)
class DynamicNetworkConfig:
    """Parameters for the lightweight dynamic D2D simulator."""

    num_entities: int = 50
    area_width_m: float = 10_000.0
    area_height_m: float = 5_000.0
    time_step_s: float = 0.01
    min_speed_mps: float = 1.0
    max_speed_mps: float = 7.0
    max_pair_distance_m: float = 200.0
    link_creation_probability: float = 0.15
    mean_link_duration_s: float = 10.0
    num_rbs: int = 5
    path_loss_exponent: float = 2.5
    shadowing_std_db: float = 8.0
    csi_error_std: float = 0.1
    min_distance_m: float = 1.0

    def validate(self) -> None:
        if self.num_entities < 0:
            raise ValueError("num_entities must be non-negative")
        if self.area_width_m <= 0 or self.area_height_m <= 0:
            raise ValueError("area dimensions must be positive")
        if self.time_step_s <= 0:
            raise ValueError("time_step_s must be positive")
        if self.min_speed_mps < 0 or self.max_speed_mps < self.min_speed_mps:
            raise ValueError("speed range must be non-negative and ordered")
        if self.max_pair_distance_m <= 0:
            raise ValueError("max_pair_distance_m must be positive")
        if not 0.0 <= self.link_creation_probability <= 1.0:
            raise ValueError("link_creation_probability must be in [0, 1]")
        if self.mean_link_duration_s <= 0:
            raise ValueError("mean_link_duration_s must be positive")
        if self.num_rbs <= 0:
            raise ValueError("num_rbs must be positive")
        if self.path_loss_exponent <= 0:
            raise ValueError("path_loss_exponent must be positive")
        if self.shadowing_std_db < 0:
            raise ValueError("shadowing_std_db must be non-negative")
        if not 0.0 <= self.csi_error_std < 1.0:
            raise ValueError("csi_error_std must be in [0, 1)")
        if self.min_distance_m <= 0:
            raise ValueError("min_distance_m must be positive")


@dataclass(frozen=True)
class EntityState:
    """Position and velocity of one mobile entity."""

    entity_id: int
    position_m: torch.Tensor
    velocity_mps: torch.Tensor


@dataclass(frozen=True)
class D2DLink:
    """One directed D2D communication pair."""

    link_id: int
    transmitter_id: int
    receiver_id: int
    start_time: float
    end_time: float


@dataclass(frozen=True)
class DynamicNetworkSnapshot:
    """Active links and CSI for one simulated time instant."""

    time: float
    entities: tuple[EntityState, ...]
    active_links: tuple[D2DLink, ...]
    gains: torch.Tensor

    @property
    def active_ids(self) -> tuple[int, ...]:
        return tuple(link.link_id for link in self.active_links)


@dataclass(frozen=True)
class DynamicNetworkStep:
    """Simulator output for one time instant."""

    snapshot: DynamicNetworkSnapshot
    events: CTDGEventBatch


class DynamicD2DSimulator:
    """Generate mobility-driven D2D links, CSI tensors, and CTDG events."""

    def __init__(self, config: DynamicNetworkConfig | None = None, seed: int | None = None) -> None:
        self.config = config or DynamicNetworkConfig()
        self.config.validate()
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

        self._time = 0.0
        self._next_link_id = 0
        self._active_links: dict[int, D2DLink] = {}
        self._previous_active_ids: tuple[int, ...] = ()
        self._previous_gains: torch.Tensor | None = None

        self._positions = self._initial_positions()
        self._velocities = self._initial_velocities()

    @property
    def time(self) -> float:
        return self._time

    def step(self) -> DynamicNetworkStep:
        """Advance one simulator tick and return snapshot plus CTDG events."""

        self._drop_inactive_links()
        self._add_new_links()

        active_links = tuple(self._active_links[link_id] for link_id in sorted(self._active_links))
        active_ids = tuple(link.link_id for link in active_links)
        gains = self._build_gains(active_links)
        events = build_ctdg_events(
            previous_active_ids=self._previous_active_ids,
            current_active_ids=active_ids,
            previous_gains=self._previous_gains,
            current_gains=gains,
            time=self._time,
        )
        snapshot = DynamicNetworkSnapshot(
            time=self._time,
            entities=self._entity_states(),
            active_links=active_links,
            gains=gains,
        )

        self._previous_active_ids = active_ids
        self._previous_gains = gains
        self._advance_mobility()
        self._time += self.config.time_step_s

        return DynamicNetworkStep(snapshot=snapshot, events=events)

    def run(self, num_steps: int) -> tuple[DynamicNetworkStep, ...]:
        """Return a chronological sequence of simulator steps."""

        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        return tuple(self.step() for _ in range(num_steps))

    def _initial_positions(self) -> torch.Tensor:
        if self.config.num_entities == 0:
            return torch.empty((0, 2))
        scale = torch.tensor([self.config.area_width_m, self.config.area_height_m])
        return torch.rand((self.config.num_entities, 2), generator=self.generator) * scale

    def _initial_velocities(self) -> torch.Tensor:
        if self.config.num_entities == 0:
            return torch.empty((0, 2))
        speeds = self._uniform(self.config.min_speed_mps, self.config.max_speed_mps, (self.config.num_entities,))
        angles = self._uniform(0.0, 2.0 * pi, (self.config.num_entities,))
        return torch.stack((speeds * torch.cos(angles), speeds * torch.sin(angles)), dim=1)

    def _drop_inactive_links(self) -> None:
        kept: dict[int, D2DLink] = {}
        for link_id, link in self._active_links.items():
            distance = self._entity_distance(link.transmitter_id, link.receiver_id)
            if self._time < link.end_time and distance <= self.config.max_pair_distance_m:
                kept[link_id] = link
        self._active_links = kept

    def _add_new_links(self) -> None:
        paired_entities = {
            entity_id
            for link in self._active_links.values()
            for entity_id in (link.transmitter_id, link.receiver_id)
        }
        available = [entity_id for entity_id in range(self.config.num_entities) if entity_id not in paired_entities]
        candidates = self._candidate_pairs(available)
        if not candidates:
            return

        order = torch.randperm(len(candidates), generator=self.generator).tolist()
        for candidate_index in order:
            tx_id, rx_id = candidates[candidate_index]
            if tx_id in paired_entities or rx_id in paired_entities:
                continue
            if self._random_float() > self.config.link_creation_probability:
                continue
            if self._random_float() < 0.5:
                tx_id, rx_id = rx_id, tx_id
            duration = self._exponential(self.config.mean_link_duration_s)
            link = D2DLink(
                link_id=self._next_link_id,
                transmitter_id=tx_id,
                receiver_id=rx_id,
                start_time=self._time,
                end_time=self._time + duration,
            )
            self._active_links[link.link_id] = link
            self._next_link_id += 1
            paired_entities.add(tx_id)
            paired_entities.add(rx_id)

    def _candidate_pairs(self, entity_ids: Iterable[int]) -> list[tuple[int, int]]:
        ids = tuple(entity_ids)
        candidates: list[tuple[int, int]] = []
        for left_index, left_id in enumerate(ids):
            for right_id in ids[left_index + 1 :]:
                if self._entity_distance(left_id, right_id) <= self.config.max_pair_distance_m:
                    candidates.append((left_id, right_id))
        return candidates

    def _build_gains(self, active_links: tuple[D2DLink, ...]) -> torch.Tensor:
        num_links = len(active_links)
        if num_links == 0:
            return torch.empty((0, 0, self.config.num_rbs))

        gains = torch.empty((num_links, num_links, self.config.num_rbs))
        for receiver_index, receiver_link in enumerate(active_links):
            receiver_position = self._positions[receiver_link.receiver_id]
            for transmitter_index, transmitter_link in enumerate(active_links):
                transmitter_position = self._positions[transmitter_link.transmitter_id]
                distance = torch.linalg.vector_norm(receiver_position - transmitter_position).clamp_min(
                    self.config.min_distance_m
                )
                path_loss = distance.pow(-self.config.path_loss_exponent)
                shadowing_db = torch.randn((self.config.num_rbs,), generator=self.generator) * self.config.shadowing_std_db
                shadowing = torch.pow(torch.tensor(10.0), shadowing_db / 10.0)
                fading_amplitude = torch.sqrt(self._exponential_tensor((self.config.num_rbs,)))
                csi_error = torch.randn((self.config.num_rbs,), generator=self.generator)
                observed_amplitude = (
                    (1.0 - self.config.csi_error_std**2) ** 0.5 * fading_amplitude
                    + self.config.csi_error_std * csi_error
                ).abs()
                gains[receiver_index, transmitter_index, :] = path_loss * shadowing * observed_amplitude.pow(2)
        return gains

    def _entity_states(self) -> tuple[EntityState, ...]:
        return tuple(
            EntityState(
                entity_id=entity_id,
                position_m=self._positions[entity_id].clone(),
                velocity_mps=self._velocities[entity_id].clone(),
            )
            for entity_id in range(self.config.num_entities)
        )

    def _advance_mobility(self) -> None:
        if self.config.num_entities == 0:
            return
        self._positions = self._positions + self._velocities * self.config.time_step_s
        self._reflect_dimension(0, self.config.area_width_m)
        self._reflect_dimension(1, self.config.area_height_m)

    def _reflect_dimension(self, dimension: int, limit: float) -> None:
        below = self._positions[:, dimension] < 0.0
        above = self._positions[:, dimension] > limit
        self._positions[below, dimension] = -self._positions[below, dimension]
        self._velocities[below, dimension] *= -1.0
        self._positions[above, dimension] = 2.0 * limit - self._positions[above, dimension]
        self._velocities[above, dimension] *= -1.0
        self._positions[:, dimension].clamp_(0.0, limit)

    def _entity_distance(self, left_id: int, right_id: int) -> float:
        return float(torch.linalg.vector_norm(self._positions[left_id] - self._positions[right_id]))

    def _uniform(self, low: float, high: float, shape: tuple[int, ...]) -> torch.Tensor:
        return low + (high - low) * torch.rand(shape, generator=self.generator)

    def _random_float(self) -> float:
        return float(torch.rand((), generator=self.generator))

    def _exponential(self, mean: float) -> float:
        sample = max(self._random_float(), 1e-12)
        return -mean * log(sample)

    def _exponential_tensor(self, shape: tuple[int, ...]) -> torch.Tensor:
        samples = torch.rand(shape, generator=self.generator).clamp_min(1e-12)
        return -torch.log(samples)
