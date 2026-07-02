"""Continuous-time dynamic graph event construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import torch


class EventType(IntEnum):
    """Event labels used by the CTDG representation."""

    ADD = 0
    UPDATE = 1
    DELETE = 2


@dataclass(frozen=True)
class InteractionEvent:
    """A directed interaction event between two D2D links."""

    source_id: int
    destination_id: int
    source_features: torch.Tensor
    destination_features: torch.Tensor
    edge_features: torch.Tensor
    time: float
    event_type: EventType


@dataclass(frozen=True)
class CTDGEventBatch:
    """Events and link-state changes for one time slot."""

    time: float
    active_ids: tuple[int, ...]
    added_ids: tuple[int, ...]
    updated_ids: tuple[int, ...]
    deleted_ids: tuple[int, ...]
    events: tuple[InteractionEvent, ...]


def build_ctdg_events(
    previous_active_ids: Sequence[int],
    current_active_ids: Sequence[int],
    previous_gains: torch.Tensor | None,
    current_gains: torch.Tensor,
    time: float,
    max_interference_neighbors: int | None = None,
) -> CTDGEventBatch:
    """Build CTDG events from active-link IDs and CSI snapshots.

    ``gains[i, j, k]`` is the gain from transmitter ``j`` to receiver ``i`` on
    RB ``k``. Direct CSI ``gains[i, i, :]`` becomes node features, and directed
    interference CSI ``gains[src, dst, :]`` becomes edge features. When
    ``max_interference_neighbors`` is set, each source link only emits events
    to the strongest interferers measured by max CSI across RBs.
    """

    previous_ids = _validate_ids(previous_active_ids, "previous_active_ids")
    current_ids = _validate_ids(current_active_ids, "current_active_ids")
    _validate_max_interference_neighbors(max_interference_neighbors)
    _validate_gains(current_gains, len(current_ids), "current_gains")

    previous_id_set = set(previous_ids)
    current_id_set = set(current_ids)
    added_ids = tuple(link_id for link_id in current_ids if link_id not in previous_id_set)
    updated_ids = tuple(link_id for link_id in current_ids if link_id in previous_id_set)
    deleted_ids = tuple(link_id for link_id in previous_ids if link_id not in current_id_set)

    if deleted_ids:
        if previous_gains is None:
            raise ValueError("previous_gains is required when deletions exist")
        _validate_gains(previous_gains, len(previous_ids), "previous_gains")
    elif previous_gains is not None:
        _validate_gains(previous_gains, len(previous_ids), "previous_gains")

    current_index = {link_id: index for index, link_id in enumerate(current_ids)}
    previous_index = {link_id: index for index, link_id in enumerate(previous_ids)}

    events: list[InteractionEvent] = []
    for source_id in current_ids:
        event_type = EventType.ADD if source_id in added_ids else EventType.UPDATE
        destination_ids = tuple(link_id for link_id in current_ids if link_id != source_id)
        destination_ids = _strongest_interference_ids(
            source_id=source_id,
            destination_ids=destination_ids,
            gains=current_gains,
            indices=current_index,
            max_interference_neighbors=max_interference_neighbors,
        )
        if destination_ids:
            for destination_id in destination_ids:
                events.append(
                    _event_from_snapshot(
                        source_id=source_id,
                        destination_id=destination_id,
                        gains=current_gains,
                        indices=current_index,
                        time=time,
                        event_type=event_type,
                    )
                )
        else:
            events.append(
                _self_event_from_snapshot(
                    link_id=source_id,
                    gains=current_gains,
                    indices=current_index,
                    time=time,
                    event_type=event_type,
                )
            )

    if deleted_ids:
        assert previous_gains is not None
        surviving_ids = tuple(link_id for link_id in current_ids if link_id in previous_id_set)
        for source_id in deleted_ids:
            destination_ids = surviving_ids if surviving_ids else (source_id,)
            if surviving_ids:
                destination_ids = _strongest_interference_ids(
                    source_id=source_id,
                    destination_ids=destination_ids,
                    gains=previous_gains,
                    indices=previous_index,
                    max_interference_neighbors=max_interference_neighbors,
                )
            for destination_id in destination_ids:
                if destination_id == source_id:
                    events.append(
                        _self_event_from_snapshot(
                            link_id=source_id,
                            gains=previous_gains,
                            indices=previous_index,
                            time=time,
                            event_type=EventType.DELETE,
                        )
                    )
                else:
                    events.append(
                        _event_from_snapshot(
                            source_id=source_id,
                            destination_id=destination_id,
                            gains=previous_gains,
                            indices=previous_index,
                            time=time,
                            event_type=EventType.DELETE,
                        )
                    )

    return CTDGEventBatch(
        time=float(time),
        active_ids=current_ids,
        added_ids=added_ids,
        updated_ids=updated_ids,
        deleted_ids=deleted_ids,
        events=tuple(events),
    )


def _event_from_snapshot(
    source_id: int,
    destination_id: int,
    gains: torch.Tensor,
    indices: dict[int, int],
    time: float,
    event_type: EventType,
) -> InteractionEvent:
    source_index = indices[source_id]
    destination_index = indices[destination_id]
    return InteractionEvent(
        source_id=source_id,
        destination_id=destination_id,
        source_features=gains[source_index, source_index, :],
        destination_features=gains[destination_index, destination_index, :],
        edge_features=gains[source_index, destination_index, :],
        time=float(time),
        event_type=event_type,
    )


def _self_event_from_snapshot(
    link_id: int,
    gains: torch.Tensor,
    indices: dict[int, int],
    time: float,
    event_type: EventType,
) -> InteractionEvent:
    index = indices[link_id]
    features = gains[index, index, :]
    return InteractionEvent(
        source_id=link_id,
        destination_id=link_id,
        source_features=features,
        destination_features=features,
        edge_features=torch.zeros_like(features),
        time=float(time),
        event_type=event_type,
    )


def _validate_ids(ids: Sequence[int], name: str) -> tuple[int, ...]:
    normalized = tuple(int(link_id) for link_id in ids)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate IDs")
    return normalized


def _strongest_interference_ids(
    source_id: int,
    destination_ids: tuple[int, ...],
    gains: torch.Tensor,
    indices: dict[int, int],
    max_interference_neighbors: int | None,
) -> tuple[int, ...]:
    if max_interference_neighbors is None or len(destination_ids) <= max_interference_neighbors:
        return destination_ids
    source_index = indices[source_id]
    ranked = sorted(
        destination_ids,
        key=lambda destination_id: float(torch.max(gains[source_index, indices[destination_id], :]).detach()),
        reverse=True,
    )
    selected = set(ranked[:max_interference_neighbors])
    return tuple(destination_id for destination_id in destination_ids if destination_id in selected)


def _validate_max_interference_neighbors(max_interference_neighbors: int | None) -> None:
    if max_interference_neighbors is not None and max_interference_neighbors <= 0:
        raise ValueError("max_interference_neighbors must be positive when set")


def _validate_gains(gains: torch.Tensor, expected_links: int, name: str) -> None:
    if not isinstance(gains, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if gains.ndim != 3:
        raise ValueError(f"{name} must have shape [num_links, num_links, num_rbs]")
    if gains.shape[0] != expected_links or gains.shape[1] != expected_links:
        raise ValueError(f"{name} link dimensions must match active ID count")
    if gains.shape[2] <= 0:
        raise ValueError(f"{name} must contain at least one RB")
