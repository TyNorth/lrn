#!/usr/bin/env python3
"""
LRN Teacher - Lesson 9: Different phrasings for temperature
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import pickle
import json
from lrn import LatticeNN, add_sentence, generate, propagate


def main():
    print("=" * 60)
    print("LRN Teacher - Lesson 9: Temperature via Different Structure")
    print("=" * 60)
    
    # Load Lesson 7 (92%)
    with open("/Users/tyarc/github/lrn/checkpoints/lesson7.pkl", 'rb') as f:
        data = pickle.load(f)
    lnn = data['lnn']
    
    print(f"\n[1] Loaded: {len(lnn.nodes)} nodes")
    
    # Try DIFFERENT structure - avoid "both" which links to animals
    # Instead use explicit definition structure
    DIFFERENT_STRUCTURE = [
        # Avoid "both" - use "is a type of" instead
        ("cold is a type of temperature", "temperature"),
        ("hot is a type of temperature", "temperature"),
        ("temperature describes cold", "cold"),
        ("temperature describes hot", "hot"),
        
        # Bridge via weather (neither animal nor both)
        ("weather has temperature", "temperature"),
        ("temperature affects weather", "weather"),
        
        # Use "measure" instead of "both"
        ("cold measures temperature", "temperature"),
        ("hot measures temperature", "temperature"),
        
        # More explicit
        ("cold equals temperature", "temperature"),
        ("hot equals temperature", "temperature"),
    ]
    
    print(f"\n[2] Training different structures...")
    
    for sentence, expected in DIFFERENT_STRUCTURE:
        for rep in range(6):
            add_sentence(lnn, sentence)
    
    print(f"    (6x each = {len(DIFFERENT_STRUCTURE)*6} total)")
    print(f"\n    Network: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    # Benchmark
    print(f"\n[3] Running benchmark...")
    
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
        candidates = generate(lnn, tokens, top_k=5)
        
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
    
    with open("/Users/tyarc/github/lrn/checkpoints/lesson9.pkl", 'wb') as f:
        pickle.dump({'lnn': lnn, 'passed': passed, 'parity': parity}, f)
    
    with open("/Users/tyarc/github/lrn/sys_test/teacher_lesson9.json", 'w') as f:
        json.dump({
            "lesson": 9,
            "results": results,
            "passed": passed,
            "total": len(BENCHMARK),
            "parity": round(parity, 1)
        }, f, indent=2)
    
    return passed >= len(BENCHMARK)


if __name__ == "__main__":
    main()