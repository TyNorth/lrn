#!/usr/bin/env python3
"""
Eval Phase 2 - Training Pipeline
Verifies: Spring topology, n-gram counts, corpus processing
"""
import sys
import os
import pickle
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, propagate
from lrn.training import add_sentence


def main():
    print("=" * 60)
    print("Phase 2: Training Pipeline - Evaluation")
    print("=" * 60)

    checkpoint_path = "/Users/tyarc/github/lrn/checkpoints/phase2.pkl"

    if not os.path.exists(checkpoint_path):
        print(f"\n✗ ERROR: Checkpoint not found: {checkpoint_path}")
        print("    Run train_phase2.py first")
        return 1

    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)

    lnn = data['lnn']
    stats = data['stats']

    print("\n[1] Loaded checkpoint")
    print(f"    Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}, Trigrams: {len(lnn.trigrams)}")

    print("\n[2] Verifying spring topology...")
    positive_springs = sum(1 for sp in lnn.springs.values() if sp.stiffness > 0)
    negative_springs = sum(1 for sp in lnn.springs.values() if sp.stiffness < 0)
    print(f"    - Positive springs (attraction): {positive_springs}")
    print(f"    - Negative springs (repulsion): {negative_springs}")

    print("\n[3] Testing propagation with learned springs...")
    test_sentence = "birds fly in the sky"
    tokens = test_sentence.split()

    lnn.reset()
    for tok in tokens:
        lnn.add_node(tok)
        lnn.nodes[tok].activation = 100
        lnn.nodes[tok].pinned = True

    lnn.add_node("identity:self")
    lnn.nodes["identity:self"].activation = 100
    lnn.nodes["identity:self"].pinned = True

    propagate(lnn, n_steps=5)

    print(f"    - Test sentence: '{test_sentence}'")
    active_after = sum(1 for n in lnn.nodes.values() if n.activation > 0 and not n.pinned)
    print(f"    - Active non-pinned nodes after propagation: {active_after}")

    print("\n[4] Verifying n-gram table...")
    test_trigram = ("birds", "fly", "in")
    trigram_count = lnn.trigrams.get(test_trigram, 0)
    print(f"    - Trigram '{test_trigram}': {trigram_count}")

    top_trigrams = sorted(lnn.trigrams.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"    - Top 5 trigrams:")
    for gram, count in top_trigrams:
        print(f"      {gram}: {count}")

    print("\n[5] Checking negative/repulsion springs...")
    neg_springs = [(k, sp.stiffness) for k, sp in lnn.springs.items() if sp.stiffness < 0]
    if neg_springs:
        print(f"    - Found {len(neg_springs)} negative springs:")
        for (a, b), k in neg_springs[:5]:
            print(f"      {a} <-> {b}: k={k}")
    else:
        print(f"    - No negative springs found")

    passed = (
        len(lnn.nodes) >= 50 and
        len(lnn.springs) >= 100 and
        len(lnn.trigrams) >= 50 and
        positive_springs > 0
    )

    print("\n" + "=" * 60)
    if passed:
        print("✓ PASS: Training pipeline working correctly")
        print(f"  - {len(lnn.nodes)} nodes created from corpus")
        print(f"  - {len(lnn.springs)} springs formed (Hebbian co-activation)")
        print(f"  - {len(lnn.trigrams)} n-grams stored")
        print(f"  - {positive_springs} positive, {negative_springs} negative springs")
        print("  - Ready to proceed to Phase 3 (Generation)")
    else:
        print("✗ FAIL: Training pipeline issues detected")
    print("=" * 60)

    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())