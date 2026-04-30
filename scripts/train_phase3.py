#!/usr/bin/env python3
"""
Train Phase 3 - Generation
Uses checkpoint from Phase 2, tests generation
"""
import sys
import os
import pickle
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, generate


def main():
    print("=" * 60)
    print("Phase 3: Generation - Training")
    print("=" * 60)

    checkpoint_path = "/Users/tyarc/github/lrn/checkpoints/phase2.pkl"
    
    if not os.path.exists(checkpoint_path):
        print(f"\n✗ ERROR: Phase 2 checkpoint not found: {checkpoint_path}")
        print("    Run train_phase2.py first")
        return 1

    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)

    lnn = data['lnn']
    print(f"\n[1] Loaded checkpoint from Phase 2")
    print(f"    Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")

    print("\n[2] Testing generation...")

    test_prompts = [
        (["birds", "fly", "in", "the"], "sky"),
        (["fish", "swim", "in", "the"], "river"),
        (["fire", "burns", "hot", "and"], "bright"),
        (["the", "sun", "shines", "in"], "the"),
    ]

    results = []
    for prompt, expected in test_prompts:
        candidates = generate(lnn, prompt, top_k=5)
        top_word = candidates[0]["word"] if candidates else "NONE"
        match = top_word == expected
        results.append({
            "prompt": " ".join(prompt),
            "expected": expected,
            "top": top_word,
            "score": candidates[0]["score"] if candidates else 0,
            "match": match
        })
        status = "✓" if match else "✗"
        print(f"    {status} '{' '.join(prompt)}' -> '{top_word}' (expected: '{expected}')")

    matches = sum(1 for r in results if r["match"])
    total = len(results)

    print(f"\n[3] Results: {matches}/{total} prompts matched expected")

    os.makedirs("/Users/tyarc/github/lrn/checkpoints", exist_ok=True)
    out_path = "/Users/tyarc/github/lrn/checkpoints/phase3.pkl"
    
    with open(out_path, 'wb') as f:
        pickle.dump({
            'lnn': lnn,
            'test_results': results,
            'matches': matches,
            'total': total
        }, f)

    print(f"\n[4] Checkpoint saved: {out_path}")
    print(f"\n{'='*60}")
    print(f"Phase 3 Complete: {matches}/{total} generation tests passed")
    print(f"{'='*60}")

    return 0 if matches > 0 else 1


if __name__ == '__main__':
    sys.exit(main())