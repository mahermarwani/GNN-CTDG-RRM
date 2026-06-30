"""Generate PyMOO benchmark labels from simulated D2D snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tgnn_rrm.config import RadioConfig
from tgnn_rrm.pymoo_benchmark import PymooBenchmarkConfig, solve_snapshot_with_pymoo
from tgnn_rrm.simulation import DynamicD2DSimulator, DynamicNetworkConfig


def main() -> None:
    args = parse_args()
    simulator = DynamicD2DSimulator(
        DynamicNetworkConfig(
            num_entities=args.num_entities,
            area_width_m=args.area_width_m,
            area_height_m=args.area_height_m,
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
    radio_config = RadioConfig(
        num_rbs=args.num_rbs,
        max_rbs_per_link=args.max_rbs_per_link,
        min_rate_bps=args.min_rate_bps,
    )
    benchmark_config = PymooBenchmarkConfig(
        population_size=args.population_size,
        generations=args.generations,
        seed=args.seed,
        p_max_watt=args.p_max_watt,
        qos_penalty_weight=args.qos_penalty_weight,
    )

    rows = []
    for step in simulator.run(args.steps):
        if not step.snapshot.active_links:
            continue
        result = solve_snapshot_with_pymoo(
            gains=step.snapshot.gains,
            radio_config=radio_config,
            benchmark_config=benchmark_config,
            p_max=args.p_max_watt,
        )
        rows.append(
            {
                "time": step.snapshot.time,
                "active_ids": step.snapshot.active_ids,
                "gains": step.snapshot.gains,
                "allocation": result.allocation,
                "power": result.power,
                "rates": result.rates,
                "mean_rate": result.mean_rate,
                "qos_violation_fraction": result.qos_violation_fraction,
                "objective": result.objective,
            }
        )
        print(
            f"time={step.snapshot.time:.4f} "
            f"links={len(step.snapshot.active_links)} "
            f"mean_rate={result.mean_rate:.2f} "
            f"qos={result.qos_violation_fraction:.3f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "radio_config": radio_config,
            "benchmark_config": benchmark_config,
            "rows": rows,
        },
        args.output,
    )
    print(f"saved={args.output} snapshots={len(rows)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/pymoo_labels.pt"))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-entities", type=int, default=12)
    parser.add_argument("--area-width-m", type=float, default=60.0)
    parser.add_argument("--area-height-m", type=float, default=60.0)
    parser.add_argument("--max-pair-distance-m", type=float, default=60.0)
    parser.add_argument("--link-creation-probability", type=float, default=1.0)
    parser.add_argument("--mean-link-duration-s", type=float, default=6.0)
    parser.add_argument("--num-rbs", type=int, default=5)
    parser.add_argument("--max-rbs-per-link", type=int, default=1)
    parser.add_argument("--min-rate-bps", type=float, default=12_000.0)
    parser.add_argument("--p-max-watt", type=float, default=10 ** ((0.0 - 30.0) / 10.0))
    parser.add_argument("--shadowing-std-db", type=float, default=8.0)
    parser.add_argument("--fading-correlation", type=float, default=0.8)
    parser.add_argument("--csi-error-std", type=float, default=0.15)
    parser.add_argument("--csi-error-correlation", type=float, default=0.8)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--qos-penalty-weight", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
