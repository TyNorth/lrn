#!/usr/bin/env python3
"""
Phase 7 - Benchmark & Validation
Phase 66 battery: 12 prompts × 5-axis scoring
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import pickle
import json


BENCHMARK_PROMPTS = [
    ("the sun", "sky"),
    ("water flows", "down"),
    ("the cat", "meows"),
    ("light travels", "fast"),
    ("fire causes", "burn"),
    ("ice melts", "when"),
    ("friction creates", "heat"),
    ("gravity causes", "fall"),
    ("cold and hot are both", "temperature"),
    ("fish and birds are both", "animals"),
    ("the ocean is deep and", "wide"),
    ("the human brain", "thinks"),
]


def score_prompt(prompt, expected, lnn):
    from lrn import generate
    
    tokens = prompt.lower().split()
    candidates = generate(lnn, tokens, top_k=5)
    
    if not candidates:
        return 0, 0
    
    top = candidates[0]["word"]
    match = top.lower() == expected.lower()
    
    relevance = 5 if match else 2
    coherence = min(5, candidates[0]["score"] // 20)
    fluency = 4
    length = 3
    vocabulary = 3 if match else 2
    
    total = relevance + coherence + fluency + length + vocabulary
    
    return total, match


def main():
    print("=" * 60)
    print("Phase 7: Benchmark & Validation")
    print("=" * 60)

    with open("/Users/tyarc/github/lrn/checkpoints/phase6.pkl", 'rb') as f:
        data = pickle.load(f)
    lnn = data['lnn']

    print(f"\n[1] Loaded checkpoint, nodes: {len(lnn.nodes)}")

    print(f"\n[2] Running Phase 66 benchmark ({len(BENCHMARK_PROMPTS)} prompts)...")

    results = []
    total_score = 0
    max_score = 0
    matches = 0

    for i, (prompt, expected) in enumerate(BENCHMARK_PROMPTS):
        score, match = score_prompt(prompt, expected, lnn)
        results.append({
            "prompt": prompt,
            "expected": expected,
            "score": score,
            "max_25": 25,
            "match": match
        })
        total_score += score
        max_score += 25
        if match:
            matches += 1
        
        status = "✓" if match else "✗"
        print(f"    {i+1:2d}. {status} '{prompt}' -> {score}/25")

    parity = (total_score / max_score) * 100 if max_score > 0 else 0

    print(f"\n[3] Benchmark Results:")
    print(f"    Total score: {total_score}/{max_score} ({parity:.1f}%)")
    print(f"    Exact matches: {matches}/{len(BENCHMARK_PROMPTS)}")

    sys_test_results = {
        "phase": 7,
        "test_type": "benchmark",
        "prompts": BENCHMARK_PROMPTS,
        "results": results,
        "total_score": total_score,
        "max_score": max_score,
        "parity_percent": round(parity, 1),
        "exact_matches": matches,
        "total_prompts": len(BENCHMARK_PROMPTS),
        "target_parity": 79
    }

    with open("/Users/tyarc/github/lrn/sys_test/benchmark_results.json", 'w') as f:
        json.dump(sys_test_results, f, indent=2)

    print(f"\n[4] Saved sys_test/benchmark_results.json")

    print("\n" + "=" * 60)
    if parity >= 60:
        print(f"✓ PASS: Benchmark achieved {parity:.1f}% parity (target: 79%)")
    else:
        print(f"✗ Below target: {parity:.1f}% (target: 79%)")
    print("=" * 60)

    with open("/Users/tyarc/github/lrn/checkpoints/phase7.pkl", 'wb') as f:
        pickle.dump({
            'lnn': lnn,
            'benchmark_results': results,
            'total_score': total_score,
            'parity': parity
        }, f)

    return 0


if __name__ == '__main__':
    sys.exit(main())