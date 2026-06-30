# GNN-CTDG-RRM

Reimplementation of an event-based Temporal Graph Neural Network (TGNN) for radio resource management in dynamic D2D wireless networks.

The implementation is being built step by step from the accompanying TGNN-RRM manuscript. The first milestone provides dependency-free wireless rate and constraint utilities; later milestones will add data generation, CTDG event construction, PyTorch models, training, and evaluation scripts.

## Current Status

- Core system configuration dataclasses.
- Rate/SINR computation for multi-RB D2D links.
- Allocation constraint checks.
- Unit tests using Python's built-in `unittest`.

## Quick Check

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
