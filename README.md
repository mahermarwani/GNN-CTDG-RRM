# GNN-CTDG-RRM

Reimplementation of an event-based Temporal Graph Neural Network (TGNN) for radio resource management in dynamic D2D wireless networks.

The implementation is being built step by step from the accompanying TGNN-RRM manuscript. The current milestone provides wireless rate utilities and a PyTorch-based CTDG event layer; later milestones will add data generation, TGNN models, training, and evaluation scripts.

## Current Status

- Core system configuration dataclasses.
- Rate/SINR computation for multi-RB D2D links.
- Allocation constraint checks.
- CTDG add/update/delete event construction from active link IDs and CSI tensors.
- Unit tests using Python's built-in `unittest`.

## Quick Check

```bash
python -m unittest discover -s tests
```
