# Phase 1 Report: Core Substrate

## Date: 2026-04-30

## Goal
Build fundamental data structures and basic propagation using latpy.

## Implementation
Created the following modules:
- `src/lrn/node.py` - Node dataclass with all fields (name, x/y/z, activation, pinned, role_counts, tau, flavors)
- `src/lrn/spring.py` - Spring dataclass with energy property, stiffness, tau, precedence
- `src/lrn/lattice.py` - LatticeNN container with nodes dict, springs dict, add_node, add_spring, _key
- `src/lrn/propagate.py` - Relaxation kernel: A_i(t+1) = clamp((4*A_i + 6*Σω·A_j)/10, 0, 100)

## Test
Two-node spring propagation: node 'a' (pinned, activation=100) → node 'b'

## Results

| Metric | Value |
|--------|-------|
| Nodes | 2 |
| Springs | 1 |
| Node 'a' activation | 100 (pinned) |
| Node 'b' activation | 60 |
| Test Status | ✓ PASS |

## Files Created
```
lrn/src/lrn/
├── __init__.py
├── node.py
├── spring.py
├── lattice.py
└── propagate.py
```

## Scripts
- `scripts/train_phase1.py` - ✓ PASSED
- `scripts/eval_phase1.py` - ✓ PASSED

## Checkpoint
`checkpoints/phase1.pkl`

## Next Phase
Phase 2: Training Pipeline - CorpusExpander, add_sentence(), n-gram table

## Status: APPROVED