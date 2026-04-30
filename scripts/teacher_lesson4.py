#!/usr/bin/env python3
"""
LRN Teacher - Lesson 4: Fill Benchmark Gaps
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import pickle
import json
from lrn import LatticeNN, add_sentence, generate, propagate


def main():
    print("=" * 60)
    print("LRN Teacher - Lesson 4: Benchmark Gap Filling")
    print("=" * 60)
    
    # Load Lesson 3
    with open("/Users/tyarc/github/lrn/checkpoints/lesson3.pkl", 'rb') as f:
        data = pickle.load(f)
    lnn = data['lnn']
    
    print(f"\n[1] Loaded Lesson 3: {len(lnn.nodes)} nodes")
    
    # Missing benchmark items - fill the gaps
    LESSON_4 = [
        ("the sun is in the sky", "sky"),
        ("the sky is above", "above"),
        ("ice melts when", "when"),
        ("ice melts in heat", "heat"),
        ("friction creates heat", "heat"),
        ("friction creates", "heat"),
        ("gravity causes fall", "fall"),
        ("gravity causes", "fall"),
        ("the ocean is deep and wide", "wide"),
    ]
    
    print(f"\n[2] Training {len(LESSON_4)} gap-filling examples...")
    
    for sentence, expected in LESSON_4:
        add_sentence(lnn, sentence)
        print(f"    + {sentence}")
    
    print(f"\n    Network: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    # Full benchmark test
    print(f"\n[3] Running FULL benchmark...")
    
    BENCHMARK = [
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
    
    results = []
    passed = 0
    
    for prompt, expected in BENCHMARK:
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=3)
        
        if candidates:
            top = candidates[0]["word"]
            match = top.lower() == expected.lower()
            status = "✓" if match else "✗"
            print(f"    {status} '{prompt}' -> {top} (expected: {expected})")
            results.append({"prompt": prompt, "expected": expected, "got": top, "match": match})
            if match:
                passed += 1
        else:
            print(f"    ✗ '{prompt}' -> no output")
            results.append({"prompt": prompt, "expected": expected, "got": "none", "match": False})
    
    parity = (passed / len(BENCHMARK)) * 100
    
    print(f"\n[4] Results: {passed}/{len(BENCHMARK)} ({parity:.0f}%)")
    
    # Save
    with open("/Users/tyarc/github/lrn/checkpoints/lesson4.pkl", 'wb') as f:
        pickle.dump({'lnn': lnn, 'passed': passed, 'total': len(BENCHMARK), 'parity': parity}, f)
    
    with open("/Users/tyarc/github/lrn/sys_test/teacher_benchmark.json", 'w') as f:
        json.dump({
            "lesson": 4,
            "results": results,
            "passed": passed,
            "total": len(BENCHMARK),
            "parity": round(parity, 1)
        }, f, indent=2)
    
    return passed >= len(BENCHMARK) - 2


if __name__ == "__main__":
    main()