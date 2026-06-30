"""Temporal GNN modules for CTDG-based radio resource management."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import torch
from torch import nn

from tgnn_rrm.ctdg import CTDGEventBatch, EventType, InteractionEvent


@dataclass(frozen=True)
class TGNNConfig:
    """Model dimensions for the first TGNN resource-allocation core."""

    num_rbs: int = 5
    memory_dim: int = 32
    message_dim: int = 32
    embedding_dim: int = 32
    hidden_dim: int = 64
    time_dim: int = 8
    attention_heads: int = 4
    max_rbs_per_link: int = 1
    memory_capacity: int | None = 512
    initial_memory_std: float = 1.0

    def validate(self) -> None:
        if self.num_rbs <= 0:
            raise ValueError("num_rbs must be positive")
        if self.memory_dim <= 0 or self.message_dim <= 0 or self.embedding_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if self.hidden_dim <= 0 or self.time_dim <= 0:
            raise ValueError("hidden_dim and time_dim must be positive")
        if self.attention_heads <= 0 or self.memory_dim % self.attention_heads != 0:
            raise ValueError("attention_heads must divide memory_dim")
        if not 1 <= self.max_rbs_per_link <= self.num_rbs:
            raise ValueError("max_rbs_per_link must be in [1, num_rbs]")
        if self.memory_capacity is not None and self.memory_capacity <= 0:
            raise ValueError("memory_capacity must be positive when set")
        if self.initial_memory_std < 0:
            raise ValueError("initial_memory_std must be non-negative")


@dataclass(frozen=True)
class TGNNOutput:
    """Resource-allocation outputs for one CTDG event batch."""

    active_ids: tuple[int, ...]
    embeddings: torch.Tensor
    rb_probabilities: torch.Tensor
    rb_allocation: torch.Tensor
    power: torch.Tensor
    graph_memory: torch.Tensor


class TemporalEncoding(nn.Module):
    """Learnable cosine temporal encoding from the manuscript."""

    def __init__(self, time_dim: int) -> None:
        super().__init__()
        self.omega_time = nn.Parameter(torch.randn(time_dim))
        self.omega_delta = nn.Parameter(torch.randn(time_dim))
        self.omega_duration = nn.Parameter(torch.randn(time_dim))

    def forward(self, time: torch.Tensor, delta_time: torch.Tensor, duration: torch.Tensor) -> torch.Tensor:
        return torch.cos(
            self.omega_time * time + self.omega_delta * delta_time + self.omega_duration * duration
        )


class TGNNResourceAllocator(nn.Module):
    """Stateful TGNN encoder-decoder for CTDG event batches.

    The module keeps node memories between calls. Use ``reset_memory`` at the
    beginning of a new sequence and ``detach_memory`` between truncated
    backpropagation windows.
    """

    def __init__(self, config: TGNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or TGNNConfig()
        self.config.validate()

        message_input_dim = 2 * self.config.memory_dim + self.config.time_dim + 3 * self.config.num_rbs
        self.temporal_encoding = TemporalEncoding(self.config.time_dim)
        self.source_message_mlps = nn.ModuleList(
            _mlp(message_input_dim, self.config.hidden_dim, self.config.message_dim) for _ in EventType
        )
        self.destination_message_mlps = nn.ModuleList(
            _mlp(message_input_dim, self.config.hidden_dim, self.config.message_dim) for _ in EventType
        )
        self.message_score = _mlp(self.config.message_dim, self.config.hidden_dim, 1)
        self.node_updater = nn.GRUCell(self.config.message_dim, self.config.memory_dim)
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=self.config.memory_dim,
            num_heads=self.config.attention_heads,
            batch_first=True,
        )
        self.graph_summary = _mlp(self.config.memory_dim, self.config.hidden_dim, self.config.memory_dim)
        self.graph_updater = nn.GRUCell(self.config.memory_dim, self.config.memory_dim)
        self.embedding_mlp = _mlp(
            2 * self.config.memory_dim + self.config.num_rbs,
            self.config.hidden_dim,
            self.config.embedding_dim,
        )
        self.rb_head = nn.Linear(self.config.embedding_dim, self.config.num_rbs)
        self.power_head = nn.Linear(self.config.embedding_dim, self.config.num_rbs)
        self.register_buffer("graph_memory", torch.zeros(self.config.memory_dim))

        self._node_memory: dict[int, torch.Tensor] = {}
        self._node_features: dict[int, torch.Tensor] = {}
        self._last_node_time: dict[int, float] = {}
        self._last_pair_time: dict[tuple[int, int], float] = {}
        self._pair_duration: dict[tuple[int, int], float] = {}

    def reset_memory(self) -> None:
        """Clear node and graph memories before a new event sequence."""

        self._node_memory.clear()
        self._node_features.clear()
        self._last_node_time.clear()
        self._last_pair_time.clear()
        self._pair_duration.clear()
        self.graph_memory.zero_()

    def detach_memory(self) -> None:
        """Detach recurrent memories from the current autograd graph."""

        self._node_memory = {node_id: memory.detach() for node_id, memory in self._node_memory.items()}
        self.graph_memory = self.graph_memory.detach()

    def forward(self, batch: CTDGEventBatch, p_max: torch.Tensor | float | None = None) -> TGNNOutput:
        """Process one chronological CTDG event batch."""

        device = self.graph_memory.device
        dtype = self.graph_memory.dtype
        messages = self._compute_messages(batch, device=device, dtype=dtype)
        self._update_node_memories(messages, time=batch.time)
        self._update_graph_memory(batch.active_ids)
        self._enforce_memory_capacity(batch.active_ids)
        return self._decode(batch.active_ids, p_max=p_max, device=device, dtype=dtype)

    def _compute_messages(
        self,
        batch: CTDGEventBatch,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[int, list[torch.Tensor]]:
        messages: dict[int, list[torch.Tensor]] = {}
        for event in batch.events:
            source_memory = self._memory_for(event.source_id, device=device, dtype=dtype)
            destination_memory = self._memory_for(event.destination_id, device=device, dtype=dtype)
            source_features = self._event_features(event.source_features, device=device, dtype=dtype)
            destination_features = self._event_features(event.destination_features, device=device, dtype=dtype)
            edge_features = self._event_features(event.edge_features, device=device, dtype=dtype)
            self._node_features[event.source_id] = source_features
            self._node_features[event.destination_id] = destination_features

            time_features = self._time_features(event)
            event_type_index = int(event.event_type)
            source_input = torch.cat(
                (
                    source_memory,
                    destination_memory,
                    time_features,
                    edge_features,
                    source_features,
                    destination_features,
                )
            )
            source_message = self.source_message_mlps[event_type_index](source_input)
            messages.setdefault(event.source_id, []).append(source_message)

            if event.destination_id != event.source_id:
                destination_input = torch.cat(
                    (
                        destination_memory,
                        source_memory,
                        time_features,
                        edge_features,
                        destination_features,
                        source_features,
                    )
                )
                destination_message = self.destination_message_mlps[event_type_index](destination_input)
                messages.setdefault(event.destination_id, []).append(destination_message)

            self._record_event_time(event)
        return messages

    def _update_node_memories(self, messages: dict[int, list[torch.Tensor]], time: float) -> None:
        for node_id, node_messages in messages.items():
            stacked_messages = torch.stack(node_messages)
            weights = torch.softmax(self.message_score(stacked_messages).squeeze(-1), dim=0)
            aggregated = torch.sum(weights.unsqueeze(-1) * stacked_messages, dim=0)
            previous = self._node_memory[node_id]
            updated = self.node_updater(aggregated.unsqueeze(0), previous.unsqueeze(0)).squeeze(0)
            self._node_memory[node_id] = updated
            self._last_node_time[node_id] = float(time)

    def _update_graph_memory(self, active_ids: tuple[int, ...]) -> None:
        active_memories = [self._node_memory[node_id] for node_id in active_ids if node_id in self._node_memory]
        if not active_memories:
            return
        memory_matrix = torch.stack(active_memories).unsqueeze(0)
        attended, _ = self.memory_attention(memory_matrix, memory_matrix, memory_matrix, need_weights=False)
        summary = self.graph_summary(attended.mean(dim=1).squeeze(0))
        updated = self.graph_updater(summary.unsqueeze(0), self.graph_memory.unsqueeze(0)).squeeze(0)
        self.graph_memory = updated

    def _decode(
        self,
        active_ids: tuple[int, ...],
        p_max: torch.Tensor | float | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TGNNOutput:
        if not active_ids:
            empty = torch.empty((0, self.config.num_rbs), device=device, dtype=dtype)
            return TGNNOutput(
                active_ids=active_ids,
                embeddings=torch.empty((0, self.config.embedding_dim), device=device, dtype=dtype),
                rb_probabilities=empty,
                rb_allocation=empty,
                power=empty,
                graph_memory=self.graph_memory,
            )

        embeddings = []
        zero_features = torch.zeros(self.config.num_rbs, device=device, dtype=dtype)
        for node_id in active_ids:
            node_memory = self._memory_for(node_id, device=device, dtype=dtype)
            node_features = self._node_features.get(node_id, zero_features)
            embeddings.append(self.embedding_mlp(torch.cat((node_memory, node_features, self.graph_memory))))
        embedding_tensor = torch.stack(embeddings)
        rb_probabilities = torch.softmax(self.rb_head(embedding_tensor), dim=1)
        rb_allocation = _topk_allocation(rb_probabilities, self.config.max_rbs_per_link)
        p_max_tensor = _p_max_tensor(p_max, len(active_ids), self.config.num_rbs, device=device, dtype=dtype)
        power = torch.sigmoid(self.power_head(embedding_tensor)) * p_max_tensor * rb_allocation
        return TGNNOutput(
            active_ids=active_ids,
            embeddings=embedding_tensor,
            rb_probabilities=rb_probabilities,
            rb_allocation=rb_allocation,
            power=power,
            graph_memory=self.graph_memory,
        )

    def _memory_for(self, node_id: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        memory = self._node_memory.get(node_id)
        if memory is None:
            memory = torch.randn(self.config.memory_dim, device=device, dtype=dtype) * self.config.initial_memory_std
            self._node_memory[node_id] = memory
        elif memory.device != device or memory.dtype != dtype:
            memory = memory.to(device=device, dtype=dtype)
            self._node_memory[node_id] = memory
        return memory

    def _event_features(self, features: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        normalized = features.to(device=device, dtype=dtype).reshape(-1)
        if normalized.numel() != self.config.num_rbs:
            raise ValueError("event feature length must match num_rbs")
        return normalized

    def _time_features(self, event: InteractionEvent) -> torch.Tensor:
        key = (event.source_id, event.destination_id)
        previous_time = self._last_pair_time.get(key)
        delta = 0.0 if previous_time is None else max(float(event.time) - previous_time, 0.0)
        duration = self._pair_duration.get(key, 0.0) + delta
        device = self.graph_memory.device
        dtype = self.graph_memory.dtype
        return self.temporal_encoding(
            torch.tensor(float(event.time), device=device, dtype=dtype),
            torch.tensor(delta, device=device, dtype=dtype),
            torch.tensor(duration, device=device, dtype=dtype),
        )

    def _record_event_time(self, event: InteractionEvent) -> None:
        key = (event.source_id, event.destination_id)
        previous_time = self._last_pair_time.get(key)
        delta = 0.0 if previous_time is None else max(float(event.time) - previous_time, 0.0)
        self._pair_duration[key] = self._pair_duration.get(key, 0.0) + delta
        self._last_pair_time[key] = float(event.time)
        if event.event_type == EventType.DELETE:
            self._last_pair_time.pop(key, None)

    def _enforce_memory_capacity(self, active_ids: tuple[int, ...]) -> None:
        capacity = self.config.memory_capacity
        if capacity is None or len(self._node_memory) <= capacity:
            return
        active_id_set = set(active_ids)
        inactive_ids = sorted(
            (node_id for node_id in self._node_memory if node_id not in active_id_set),
            key=lambda node_id: self._last_node_time.get(node_id, float("-inf")),
        )
        for node_id in inactive_ids:
            if len(self._node_memory) <= capacity:
                break
            self._node_memory.pop(node_id, None)
            self._node_features.pop(node_id, None)
            self._last_node_time.pop(node_id, None)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _topk_allocation(probabilities: torch.Tensor, max_rbs_per_link: int) -> torch.Tensor:
    allocation = torch.zeros_like(probabilities)
    if probabilities.numel() == 0:
        return allocation
    rb_count = min(max_rbs_per_link, probabilities.shape[1])
    indices = torch.topk(probabilities, k=rb_count, dim=1).indices
    allocation.scatter_(1, indices, 1.0)
    return allocation


def _p_max_tensor(
    p_max: torch.Tensor | float | None,
    num_links: int,
    num_rbs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if p_max is None:
        return torch.ones((num_links, num_rbs), device=device, dtype=dtype)
    if isinstance(p_max, Real):
        return torch.full((num_links, num_rbs), p_max, device=device, dtype=dtype)

    tensor = p_max.to(device=device, dtype=dtype)
    if tensor.ndim == 0:
        return tensor.expand(num_links, num_rbs)
    if tensor.ndim == 1 and tensor.shape[0] == num_links:
        return tensor.unsqueeze(1).expand(num_links, num_rbs)
    if tensor.shape == (num_links, num_rbs):
        return tensor
    raise ValueError("p_max must be scalar, [num_links], or [num_links, num_rbs]")
