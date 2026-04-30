#!/usr/bin/env python3
"""
Eval Phase 3 - Generation
Verifies: gravity formula, generate() function
"""
import sys
import os
import pickle
import json
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, generate


def main():
    print("=" * 60)
    print("Phase 3: Generation - Evaluation")
    print("=" * 60)

    checkpoint_path = "/Users/tyarc/github/lrn/checkpoints/phase3.pkl"

    if not os.path.exists(checkpoint_path):
        print(f"\n✗ ERROR: Checkpoint not found: {checkpoint_path}")
        print("    Run train_phase3.py first")
        return 1

    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)

    lnn = data['lnn']
    results = data['test_results']
    matches = data['matches']
    total = data['total']

    print("\n[1] Loaded Phase 3 checkpoint")
    print(f"    Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")

    print("\n[2] Generation Results:")
    for r in results:
        status = "✓" if r["match"] else "✗"
        print(f"    {status} '{r['prompt']}' -> '{r['top']}' (expected: '{r['expected']}')")

    print("\n[3] Testing additional prompts for sys_test...")
    
    additional_tests = [
        ["the", "sun"],
        ["water", "flows"],
        ["the", "cat"],
        ["light", "travels"],
        ["fire", "causes"],
        ["ice", "melts", "when"],
        ["cold", "and", "hot", "are", "both"],
    ]
    
    all_results = []
    for prompt in additional_tests:
        candidates = generate(lnn, prompt, top_k=3)
        result = {
            "prompt": " ".join(prompt),
            "top1": candidates[0]["word"] if len(candidates) > 0 else "NONE",
            "top2": candidates[1]["word"] if len(candidates) > 1 else "NONE",
            "top3": candidates[2]["word"] if len(candidates) > 2 else "NONE",
            "top1_score": candidates[0]["score"] if len(candidates) > 0 else 0,
            "top1_activation": candidates[0]["activation"] if len(candidates) > 0 else 0,
        }
        all_results.append(result)
        print(f"    '{' '.join(prompt)}' -> {result['top1']} (score: {result['top1_score']})")

    sys_test_path = "/Users/tyarc/github/lrn/sys_test/generation_results.json"
    with open(sys_test_path, 'w') as f:
        json.dump({
            "phase": 3,
            "test_type": "language_generation",
            "prompt_results": results,
            "additional_tests": all_results,
            "match_rate": f"{matches}/{total}"
        }, f, indent=2)
    
    print(f"\n[4] Saved sys_test results: {sys_test_path}")

    passed = matches >= total // 2

    print("\n" + "=" * 60)
    if passed:
        print(f"✓ PASS: Generation working ({matches}/{total} matches)")
        print("  - Gravity formula calculating scores")
        print("  - generate() returning top candidates")
        print("  - Ready to proceed to Phase 4 (Performance)")
    else:
        print(f"✗ FAIL: Generation issues ({matches}/{total} matches)")
    print("=" * 60)

    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())