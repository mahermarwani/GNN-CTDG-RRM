"""Run a small unsupervised TGNN-RRM training loop and print progress."""

from __future__ import annotations

import argparse

import torch

from tgnn_rrm.config import RadioConfig
from tgnn_rrm.objectives import RRMObjectiveConfig, unsupervised_rrm_loss
from tgnn_rrm.simulation import DynamicD2DSimulator, DynamicNetworkConfig
from tgnn_rrm.tgnn import TGNNConfig, TGNNResourceAllocator


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

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
            csi_error_std=args.csi_error_std,
        ),
        seed=args.seed,
    )
    sequence = simulator.run(args.steps)

    model = TGNNResourceAllocator(
        TGNNConfig(
            num_rbs=args.num_rbs,
            memory_dim=args.memory_dim,
            message_dim=args.memory_dim,
            embedding_dim=args.memory_dim,
            hidden_dim=args.hidden_dim,
            max_rbs_per_link=args.max_rbs_per_link,
        )
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    radio_config = RadioConfig(
        num_rbs=args.num_rbs,
        max_rbs_per_link=args.max_rbs_per_link,
        min_rate_bps=args.min_rate_bps,
    )
    objective_config = RRMObjectiveConfig(
        qos_weight=args.qos_weight,
        rate_scale_bps=args.rate_scale_bps,
    )

    print("epoch,avg_loss,avg_mean_rate_bps,avg_qos_penalty,avg_active_links", flush=True)
    for epoch in range(1, args.epochs + 1):
        torch.manual_seed(args.seed)
        model.reset_memory()
        losses: list[float] = []
        mean_rates: list[float] = []
        qos_penalties: list[float] = []
        active_links: list[int] = []

        for step in sequence:
            if not step.snapshot.active_links:
                continue

            optimizer.zero_grad()
            output = model(step.events, p_max=args.p_max_watt)
            result = unsupervised_rrm_loss(
                gains=step.snapshot.gains,
                allocation=output.rb_probabilities,
                power=output.power,
                radio_config=radio_config,
                p_max=args.p_max_watt,
                objective_config=objective_config,
            )
            result.loss.backward()
            optimizer.step()
            model.detach_memory()

            losses.append(float(result.loss.detach()))
            mean_rates.append(float(result.mean_rate.detach()))
            qos_penalties.append(float(result.qos_penalty.detach()))
            active_links.append(len(step.snapshot.active_links))

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"{epoch},"
                f"{_mean(losses):.6f},"
                f"{_mean(mean_rates):.2f},"
                f"{_mean(qos_penalties):.6f},"
                f"{_mean(active_links):.2f}",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-entities", type=int, default=4)
    parser.add_argument("--area-width-m", type=float, default=10.0)
    parser.add_argument("--area-height-m", type=float, default=10.0)
    parser.add_argument("--max-pair-distance-m", type=float, default=100.0)
    parser.add_argument("--link-creation-probability", type=float, default=1.0)
    parser.add_argument("--mean-link-duration-s", type=float, default=10.0)
    parser.add_argument("--num-rbs", type=int, default=5)
    parser.add_argument("--max-rbs-per-link", type=int, default=1)
    parser.add_argument("--p-max-watt", type=float, default=0.0025)
    parser.add_argument("--min-rate-bps", type=float, default=7_000.0)
    parser.add_argument("--qos-weight", type=float, default=1.0)
    parser.add_argument("--rate-scale-bps", type=float, default=1_000.0)
    parser.add_argument("--memory-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--shadowing-std-db", type=float, default=0.0)
    parser.add_argument("--csi-error-std", type=float, default=0.0)
    return parser.parse_args()


def _mean(values: list[float] | list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
