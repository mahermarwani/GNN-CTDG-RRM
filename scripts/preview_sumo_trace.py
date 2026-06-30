"""Preview a SUMO FCD trace as D2D snapshots and CTDG events."""

from __future__ import annotations

import argparse
from pathlib import Path

from tgnn_rrm.simulation import DynamicNetworkConfig
from tgnn_rrm.sumo import SumoD2DSimulator, load_sumo_fcd_trace


def main() -> None:
    args = parse_args()
    trace = load_sumo_fcd_trace(args.trace)
    simulator = SumoD2DSimulator(
        trace,
        config=DynamicNetworkConfig(
            max_pair_distance_m=args.max_pair_distance_m,
            link_creation_probability=args.link_creation_probability,
            mean_link_duration_s=args.mean_link_duration_s,
            num_rbs=args.num_rbs,
            shadowing_std_db=args.shadowing_std_db,
            fading_correlation=args.fading_correlation,
            csi_error_std=args.csi_error_std,
            csi_error_correlation=args.csi_error_correlation,
        ),
        seed=args.seed,
    )

    print(
        f"trace={args.trace} frames={len(trace.frames)} entities={trace.num_entities}",
        flush=True,
    )
    for slot_index, step in enumerate(simulator.run(args.steps), start=1):
        print(
            f"slot={slot_index} "
            f"time={step.snapshot.time:.3f} "
            f"entities={len(step.snapshot.entities)} "
            f"links={len(step.snapshot.active_links)} "
            f"events={len(step.events.events)} "
            f"added={len(step.events.added_ids)} "
            f"updated={len(step.events.updated_ids)} "
            f"deleted={len(step.events.deleted_ids)}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Path to a SUMO FCD XML file.")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-pair-distance-m", type=float, default=60.0)
    parser.add_argument("--link-creation-probability", type=float, default=1.0)
    parser.add_argument("--mean-link-duration-s", type=float, default=6.0)
    parser.add_argument("--num-rbs", type=int, default=5)
    parser.add_argument("--shadowing-std-db", type=float, default=8.0)
    parser.add_argument("--fading-correlation", type=float, default=0.8)
    parser.add_argument("--csi-error-std", type=float, default=0.15)
    parser.add_argument("--csi-error-correlation", type=float, default=0.8)
    return parser.parse_args()


if __name__ == "__main__":
    main()
