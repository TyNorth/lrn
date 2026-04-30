#!/usr/bin/env python3
"""
LRN Teacher - Experiment B: Combine Exact + Diverse in Same Network
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import pickle
import json
from lrn import LatticeNN, add_sentence, generate, propagate


def main():
    print("=" * 60)
    print("Experiment B: Exact + Diverse Combined")
    print("=" * 60)
    
    lnn = LatticeNN()
    
    # PHASE 1: Exact training (like Lesson 10)
    print(f"\n[Phase 1] Exact training...")
    
    EXACT = [
        ("the sun is in the sky", "sky"),
        ("water flows down", "down"),
        ("the cat meows", "meows"),
        ("light travels fast", "fast"),
        ("fire causes burn", "burn"),
        ("ice melts when", "when"),
        ("friction creates heat", "heat"),
        ("gravity causes fall", "fall"),
        ("cold and hot are both temperature", "temperature"),
        ("fish and birds are both animals", "animals"),
        ("the ocean is deep and wide", "wide"),
        ("the human brain thinks", "thinks"),
    ]
    
    # Train exact 15x each
    for sentence, expected in EXACT:
        for _ in range(15):
            add_sentence(lnn, sentence)
    
    print(f"    After exact: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    # PHASE 2: Diverse training (generalization examples)
    print(f"\n[Phase 2] Diverse training...")
    
    DIVERSE = [
        # Variations of known concepts
        ("dogs bark", "barks"),
        ("cats meow", "meows"),
        ("a bird flies", "flies"),
        ("many fish swim", "swims"),
        ("the sun shines bright", "bright"),
        ("rain falls from clouds", "clouds"),
        
        # New concepts for bridging
        ("the moon glows", "glows"),
        ("the stars twinkle", "twinkle"),
        ("clouds form in sky", "sky"),
        
        # More variations
        ("birds and fish are animals", "animals"),
        ("hot and cold are temperature", "temperature"),
    ]
    
    # Train diverse 3x each
    for sentence, expected in DIVERSE:
        for _ in range(3):
            add_sentence(lnn, sentence)
    
    print(f"    After diverse: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    # PHASE 3: More exact epochs (refine)
    print(f"\n[Phase 3] Refine with more epochs...")
    
    for sentence, expected in EXACT[:6]:  # First half
        for _ in range(5):
            add_sentence(lnn, sentence)
    
    print(f"    Final: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    # Test
    print(f"\n[Testing]")
    
    TESTS = [
        # Original benchmark
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
        
        # Generalization (from diverse)
        ("dogs", "barks"),
        ("cats", "meows"),
        ("the moon", "glows"),
        ("clouds form in", "sky"),
    ]
    
    results = []
    passed = 0
    
    for prompt, expected in TESTS:
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
    
    parity = (passed / len(TESTS)) * 100
    print(f"\n[Results] {passed}/{len(TESTS)} ({parity:.0f}%)")
    
    with open("/Users/tyarc/github/lrn/checkpoints/experiment_b.pkl", 'wb') as f:
        pickle.dump({'lnn': lnn}, f)
    
    with open("/Users/tyarc/github/lrn/sys_test/experiment_b_combined.json", 'w') as f:
        json.dump({
            "experiment": "b_combined",
            "passed": passed,
            "total": len(TESTS),
            "parity": round(parity, 1),
            "results": results
        }, f, indent=2)


if __name__ == "__main__":
    main()