#!/usr/bin/env python3
"""
Eval Phase 1 - Core Substrate
Verifies: Basic propagation works, two nodes connected by spring
"""
import sys
import os
import pickle
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, propagate


def main():
    print("=" * 60)
    print("Phase 1: Core Substrate - Evaluation")
    print("=" * 60)

    checkpoint_path = "/Users/tyarc/github/lrn/checkpoints/phase1.pkl"

    if not os.path.exists(checkpoint_path):
        print(f"\n✗ ERROR: Checkpoint not found: {checkpoint_path}")
        print("    Run train_phase1.py first")
        return 1

    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)

    lnn = data['lnn']
    test_data = data['test']

    print("\n[1] Loaded checkpoint")
    print(f"    Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")

    print("\n[2] Re-running propagation test...")

    lnn.reset()
    lnn.add_node("a")
    lnn.add_node("b")
    lnn.add_spring("a", "b", stiffness=5)
    lnn.nodes["a"].activation = 100
    lnn.nodes["a"].pinned = True

    propagate(lnn, n_steps=1)

    b_activation = lnn.nodes["b"].activation
    passed = b_activation > 0

    print("\n[3] Test Results:")
    print(f"    - Node 'a' pinned: {lnn.nodes['a'].pinned}")
    print(f"    - Node 'a' activation: {lnn.nodes['a'].activation}")
    print(f"    - Node 'b' activation: {b_activation}")
    print(f"    - Spring 'a-b' stiffness: {lnn.springs[lnn._key('a','b')].stiffness}")

    print("\n" + "=" * 60)
    if passed:
        print("✓ PASS: Activation flowed from node 'a' to node 'b'")
        print("  - Two-node spring propagation working correctly")
        print("  - Ready to proceed to Phase 2 (Training Pipeline)")
    else:
        print("✗ FAIL: No activation flow detected")
    print("=" * 60)

    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())