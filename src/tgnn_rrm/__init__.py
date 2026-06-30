"""Utilities and models for TGNN-based radio resource management."""

from tgnn_rrm.config import RadioConfig
from tgnn_rrm.ctdg import CTDGEventBatch, EventType, InteractionEvent, build_ctdg_events
from tgnn_rrm.objectives import (
    RRMObjectiveConfig,
    RRMObjectiveResult,
    torch_link_rates,
    torch_mean_rate,
    unsupervised_rrm_loss,
)
from tgnn_rrm.pymoo_benchmark import (
    PymooBenchmarkConfig,
    PymooRRMResult,
    candidate_to_allocation_power,
    candidate_to_relaxed_allocation_power,
    is_pymoo_available,
    solve_snapshot_with_pymoo,
    solve_snapshots_with_pymoo,
)
from tgnn_rrm.radio import (
    constraint_violations,
    link_rates,
    mean_rate,
    qos_violation_fraction,
)
from tgnn_rrm.simulation import (
    D2DLink,
    DynamicD2DSimulator,
    DynamicNetworkConfig,
    DynamicNetworkSnapshot,
    DynamicNetworkStep,
    EntityState,
)
from tgnn_rrm.sumo import (
    SumoD2DSimulator,
    SumoFCDTrace,
    SumoMobileEntity,
    SumoMobilityFrame,
    load_sumo_fcd_trace,
)
from tgnn_rrm.tgnn import (
    TGNNConfig,
    TGNNOutput,
    TGNNResourceAllocator,
    TemporalEncoding,
)

__all__ = [
    "CTDGEventBatch",
    "D2DLink",
    "DynamicD2DSimulator",
    "DynamicNetworkConfig",
    "DynamicNetworkSnapshot",
    "DynamicNetworkStep",
    "EntityState",
    "EventType",
    "InteractionEvent",
    "PymooBenchmarkConfig",
    "PymooRRMResult",
    "RRMObjectiveConfig",
    "RRMObjectiveResult",
    "RadioConfig",
    "SumoD2DSimulator",
    "SumoFCDTrace",
    "SumoMobileEntity",
    "SumoMobilityFrame",
    "TGNNConfig",
    "TGNNOutput",
    "TGNNResourceAllocator",
    "TemporalEncoding",
    "build_ctdg_events",
    "candidate_to_allocation_power",
    "candidate_to_relaxed_allocation_power",
    "constraint_violations",
    "is_pymoo_available",
    "link_rates",
    "load_sumo_fcd_trace",
    "mean_rate",
    "qos_violation_fraction",
    "solve_snapshot_with_pymoo",
    "solve_snapshots_with_pymoo",
    "torch_link_rates",
    "torch_mean_rate",
    "unsupervised_rrm_loss",
]
