# GNN-CTDG-RRM

Reimplementation of an event-based Temporal Graph Neural Network (TGNN) for radio resource management in dynamic D2D wireless networks.

The implementation is being built step by step from the accompanying TGNN-RRM manuscript. The current milestone provides wireless rate utilities, a PyTorch-based CTDG event layer, a lightweight dynamic D2D simulator, and an initial TGNN resource-allocation core; later milestones will add training and evaluation scripts.

## Current Status

- Core system configuration dataclasses.
- Rate/SINR computation for multi-RB D2D links.
- Allocation constraint checks.
- CTDG add/update/delete event construction from active link IDs and CSI tensors.
- Dynamic D2D event generation from bounded mobility, distance-based pairing, and CSI snapshots.
- TGNN memory/message core with RB-allocation probabilities and power outputs.
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

## Quick Check

```bash
python -m unittest discover -s tests
```
