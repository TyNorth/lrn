#!/usr/bin/env python3
"""
Train Phase 1 - Core Substrate
Goal: Create basic LatticeNN, test propagation between two nodes
"""
import sys
import os
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, propagate


def main():
    print("=" * 60)
    print("Phase 1: Core Substrate - Training")
    print("=" * 60)

    lnn = LatticeNN()
    print("\n[1] Created empty LatticeNN")

    lnn.add_node("a")
    lnn.add_node("b")
    lnn.add_spring("a", "b", stiffness=5)
    print("[2] Added nodes 'a' and 'b' with spring (k=5)")

    lnn.nodes["a"].activation = 100
    lnn.nodes["a"].pinned = True
    print(f"[3] Pinned node 'a' with activation=100")

    print("\n[4] Before propagation:")
    print(f"    a: activation={lnn.nodes['a'].activation}, pinned={lnn.nodes['a'].pinned}")
    print(f"    b: activation={lnn.nodes['b'].activation}")

    print("\n[5] Running propagation (1 step)...")
    propagate(lnn, n_steps=1)

    print("\n[6] After propagation:")
    print(f"    a: activation={lnn.nodes['a'].activation} (pinned, unchanged)")
    print(f"    b: activation={lnn.nodes['b'].activation} (should be > 0)")

    b_activation = lnn.nodes["b"].activation
    print(f"\n[7] RESULT: Node b activation = {b_activation}")

    if b_activation > 0:
        print("    ✓ PASS - Activation flowed through spring")
    else:
        print("    ✗ FAIL - No activation flow")

    os.makedirs("/Users/tyarc/github/lrn/checkpoints", exist_ok=True)
    checkpoint_path = "/Users/tyarc/github/lrn/checkpoints/phase1.pkl"

    import pickle
    with open(checkpoint_path, 'wb') as f:
        pickle.dump({
            'lnn': lnn,
            'test': {
                'b_activation': b_activation,
                'passed': b_activation > 0
            }
        }, f)

    print(f"\n[8] Checkpoint saved: {checkpoint_path}")
    print(f"\n{'='*60}")
    print(f"Phase 1 Complete: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    print(f"Test Result: {'PASS' if b_activation > 0 else 'FAIL'}")
    print(f"{'='*60}")

    return 0 if b_activation > 0 else 1


if __name__ == '__main__':
    sys.exit(main())