"""SUMO mobility trace adapters for dynamic D2D benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import log, radians, sin, cos
from pathlib import Path
from xml.etree import ElementTree

import torch

from tgnn_rrm.ctdg import build_ctdg_events
from tgnn_rrm.simulation import (
    D2DLink,
    DynamicD2DSimulator,
    DynamicNetworkConfig,
    DynamicNetworkSnapshot,
    DynamicNetworkStep,
    EntityState,
)


@dataclass(frozen=True)
class SumoMobileEntity:
    """One vehicle or pedestrian state from a SUMO FCD timestep."""

    external_id: str
    entity_id: int
    node_type: str
    position_m: torch.Tensor
    velocity_mps: torch.Tensor


@dataclass(frozen=True)
class SumoMobilityFrame:
    """All SUMO mobile entities present at one time instant."""

    time: float
    entities: tuple[SumoMobileEntity, ...]


@dataclass(frozen=True)
class SumoFCDTrace:
    """Parsed SUMO floating-car-data trace with stable numeric entity IDs."""

    frames: tuple[SumoMobilityFrame, ...]
    entity_ids: tuple[str, ...]

    @property
    def num_entities(self) -> int:
        return len(self.entity_ids)


def load_sumo_fcd_trace(path: str | Path) -> SumoFCDTrace:
    """Load vehicles and pedestrians from a SUMO FCD XML file."""

    source = Path(path)
    root = ElementTree.parse(source).getroot()
    raw_frames: list[tuple[float, list[tuple[str, str, float, float, float, float]]]] = []
    external_ids: set[str] = set()

    for timestep in root.iter("timestep"):
        time = _required_float(timestep.attrib, "time", "timestep")
        raw_entities: list[tuple[str, str, float, float, float, float]] = []
        seen_in_frame: set[str] = set()
        for element in timestep:
            if element.tag not in {"vehicle", "person"}:
                continue
            external_id = _required_text(element.attrib, "id", element.tag)
            if external_id in seen_in_frame:
                raise ValueError(f"duplicate SUMO entity ID in timestep: {external_id}")
            seen_in_frame.add(external_id)
            external_ids.add(external_id)
            raw_entities.append(
                (
                    external_id,
                    element.tag,
                    _required_float(element.attrib, "x", element.tag),
                    _required_float(element.attrib, "y", element.tag),
                    _optional_float(element.attrib, "speed", 0.0),
                    _optional_float(element.attrib, "angle", 0.0),
                )
            )
        raw_frames.append((time, raw_entities))

    entity_ids = tuple(sorted(external_ids))
    entity_index = {external_id: index for index, external_id in enumerate(entity_ids)}
    frames = tuple(_build_frame(time, raw_entities, entity_index) for time, raw_entities in raw_frames)
    return SumoFCDTrace(frames=frames, entity_ids=entity_ids)


class SumoD2DSimulator:
    """Replay SUMO mobility as D2D snapshots and CTDG events."""

    def __init__(
        self,
        trace: SumoFCDTrace,
        config: DynamicNetworkConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.trace = trace
        base_config = config or DynamicNetworkConfig()
        self.config = replace(base_config, num_entities=trace.num_entities)
        self.config.validate()
        self._channel_simulator = DynamicD2DSimulator(self.config, seed=seed)
        self._frame_index = 0
        self._next_link_id = 0
        self._active_links: dict[int, D2DLink] = {}
        self._previous_active_ids: tuple[int, ...] = ()
        self._previous_gains: torch.Tensor | None = None

    def step(self) -> DynamicNetworkStep:
        """Return the next SUMO-backed network step.

        Raises ``StopIteration`` when all trace frames have been consumed.
        """

        if self._frame_index >= len(self.trace.frames):
            raise StopIteration("SUMO trace is exhausted")

        frame = self.trace.frames[self._frame_index]
        present_entities = {entity.entity_id: entity for entity in frame.entities}
        positions, velocities = self._dense_state_tensors(present_entities)
        self._channel_simulator._positions = positions
        self._channel_simulator._velocities = velocities
        self._channel_simulator._time = frame.time

        self._drop_inactive_links(present_entities)
        self._add_new_links(present_entities)

        active_links = tuple(self._active_links[link_id] for link_id in sorted(self._active_links))
        active_ids = tuple(link.link_id for link in active_links)
        gains = self._channel_simulator._build_gains(active_links)
        events = build_ctdg_events(
            previous_active_ids=self._previous_active_ids,
            current_active_ids=active_ids,
            previous_gains=self._previous_gains,
            current_gains=gains,
            time=frame.time,
        )
        snapshot = DynamicNetworkSnapshot(
            time=frame.time,
            entities=tuple(
                EntityState(
                    entity_id=entity.entity_id,
                    position_m=entity.position_m.clone(),
                    velocity_mps=entity.velocity_mps.clone(),
                )
                for entity in sorted(frame.entities, key=lambda item: item.entity_id)
            ),
            active_links=active_links,
            gains=gains,
        )

        self._previous_active_ids = active_ids
        self._previous_gains = gains
        self._frame_index += 1
        return DynamicNetworkStep(snapshot=snapshot, events=events)

    def run(self, num_steps: int) -> tuple[DynamicNetworkStep, ...]:
        """Return up to ``num_steps`` chronological SUMO-backed steps."""

        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        steps: list[DynamicNetworkStep] = []
        for _ in range(num_steps):
            try:
                steps.append(self.step())
            except StopIteration:
                break
        return tuple(steps)

    def _dense_state_tensors(
        self,
        present_entities: dict[int, SumoMobileEntity],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.zeros((self.trace.num_entities, 2))
        velocities = torch.zeros((self.trace.num_entities, 2))
        for entity_id, entity in present_entities.items():
            positions[entity_id] = entity.position_m
            velocities[entity_id] = entity.velocity_mps
        return positions, velocities

    def _drop_inactive_links(self, present_entities: dict[int, SumoMobileEntity]) -> None:
        kept: dict[int, D2DLink] = {}
        for link_id, link in self._active_links.items():
            if link.transmitter_id not in present_entities or link.receiver_id not in present_entities:
                continue
            distance = self._entity_distance(link.transmitter_id, link.receiver_id)
            if self._channel_simulator._time < link.end_time and distance <= self.config.max_pair_distance_m:
                kept[link_id] = link
        self._active_links = kept

    def _add_new_links(self, present_entities: dict[int, SumoMobileEntity]) -> None:
        paired_entities = {
            entity_id
            for link in self._active_links.values()
            for entity_id in (link.transmitter_id, link.receiver_id)
        }
        available = tuple(entity_id for entity_id in sorted(present_entities) if entity_id not in paired_entities)
        candidates = self._candidate_pairs(available)
        if not candidates:
            return

        order = torch.randperm(len(candidates), generator=self._channel_simulator.generator).tolist()
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
                start_time=self._channel_simulator._time,
                end_time=self._channel_simulator._time + duration,
            )
            self._active_links[link.link_id] = link
            self._next_link_id += 1
            paired_entities.add(tx_id)
            paired_entities.add(rx_id)

    def _candidate_pairs(self, entity_ids: tuple[int, ...]) -> list[tuple[int, int]]:
        candidates: list[tuple[int, int]] = []
        for left_index, left_id in enumerate(entity_ids):
            for right_id in entity_ids[left_index + 1 :]:
                if self._entity_distance(left_id, right_id) <= self.config.max_pair_distance_m:
                    candidates.append((left_id, right_id))
        return candidates

    def _entity_distance(self, left_id: int, right_id: int) -> float:
        positions = self._channel_simulator._positions
        return float(torch.linalg.vector_norm(positions[left_id] - positions[right_id]))

    def _random_float(self) -> float:
        return float(torch.rand((), generator=self._channel_simulator.generator))

    def _exponential(self, mean: float) -> float:
        sample = max(self._random_float(), 1e-12)
        return -mean * log(sample)


def _build_frame(
    time: float,
    raw_entities: list[tuple[str, str, float, float, float, float]],
    entity_index: dict[str, int],
) -> SumoMobilityFrame:
    entities = []
    for external_id, node_type, x, y, speed, angle in raw_entities:
        angle_rad = radians(angle)
        entities.append(
            SumoMobileEntity(
                external_id=external_id,
                entity_id=entity_index[external_id],
                node_type=node_type,
                position_m=torch.tensor([x, y], dtype=torch.float32),
                velocity_mps=torch.tensor([speed * sin(angle_rad), speed * cos(angle_rad)], dtype=torch.float32),
            )
        )
    return SumoMobilityFrame(time=float(time), entities=tuple(sorted(entities, key=lambda item: item.entity_id)))


def _required_text(attributes: dict[str, str], key: str, element_name: str) -> str:
    value = attributes.get(key)
    if value is None:
        raise ValueError(f"{element_name} is missing required attribute {key!r}")
    return value


def _required_float(attributes: dict[str, str], key: str, element_name: str) -> float:
    return float(_required_text(attributes, key, element_name))


def _optional_float(attributes: dict[str, str], key: str, default: float) -> float:
    value = attributes.get(key)
    return default if value is None else float(value)
