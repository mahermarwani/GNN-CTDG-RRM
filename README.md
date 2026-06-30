# GNN-CTDG-RRM

Reimplementation of an event-based Temporal Graph Neural Network (TGNN) for radio resource management in dynamic D2D wireless networks.

The implementation is being built step by step from the accompanying TGNN-RRM manuscript. The current milestone provides wireless rate utilities, a PyTorch-based CTDG event layer, a lightweight dynamic D2D simulator, an initial TGNN resource-allocation core, and differentiable RRM objectives; later milestones will add training and evaluation scripts.

## Current Status

- Core system configuration dataclasses.
- Rate/SINR computation for multi-RB D2D links.
- Allocation constraint checks.
- CTDG add/update/delete event construction from active link IDs and CSI tensors.
- Dynamic D2D event generation from bounded mobility, distance-based pairing, and temporally correlated CSI snapshots.
- TGNN memory/message core with RB-allocation probabilities and power outputs.
- Differentiable unsupervised loss for rate maximization with QoS and constraint penalties.
- Optional PyMOO benchmark generation for near-optimal RB and power labels.
- Unit tests using Python's built-in `unittest`.

## Minimal Event Stream Example

```python
from tgnn_rrm.simulation import DynamicD2DSimulator, DynamicNetworkConfig

config = DynamicNetworkConfig(num_entities=50, num_rbs=5)
simulator = DynamicD2DSimulator(config, seed=7)

step = simulator.step()
print(step.snapshot.active_ids)
print(step.snapshot.gains.shape)
print(len(step.events.events))
```

## Minimal TGNN Forward Pass

```python
from tgnn_rrm.tgnn import TGNNConfig, TGNNResourceAllocator

model = TGNNResourceAllocator(TGNNConfig(num_rbs=5))
output = model(step.events)

print(output.rb_probabilities.shape)
print(output.rb_allocation.shape)
print(output.power.shape)
```

## Minimal Unsupervised Objective

```python
from tgnn_rrm.config import RadioConfig
from tgnn_rrm.objectives import unsupervised_rrm_loss

result = unsupervised_rrm_loss(
    gains=step.snapshot.gains,
    allocation=output.rb_probabilities,
    power=output.power,
    radio_config=RadioConfig(num_rbs=5),
    p_max=0.1,
)
result.loss.backward()
```

## Quick Check

```bash
python -m unittest discover -s tests
```

## Loss Progress Example

```bash
python scripts/train_unsupervised.py
```

## PyMOO Benchmark Labels

Install the optional benchmark dependency, then generate labels:

```bash
pip install ".[benchmark]"
python scripts/generate_pymoo_labels.py --steps 5
```

The default benchmark profile is intentionally dense and interference-heavy.
Use `--num-entities`, `--max-pair-distance-m`, or `--min-rate-bps` to tune
hardness for smaller or larger experiments.
