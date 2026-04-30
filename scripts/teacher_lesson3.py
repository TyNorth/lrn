#!/usr/bin/env python3
"""
LRN Teacher Curriculum - Lesson 3: Categories & Properties
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import pickle
from lrn import LatticeNN, add_sentence, generate, propagate


def main():
    print("=" * 60)
    print("LRN Teacher - Lesson 3: Categories & Properties")
    print("=" * 60)
    
    # Load Lesson 2
    with open("/Users/tyarc/github/lrn/checkpoints/lesson2.pkl", 'rb') as f:
        data = pickle.load(f)
    lnn = data['lnn']
    
    print(f"\n[1] Loaded Lesson 2: {len(lnn.nodes)} nodes")
    
    # Categories and properties - more abstract
    LESSON_3 = [
        # Categories (both X are Y)
        ("dog and cat are both animals", "animals"),
        ("fish and birds are both animals", "animals"),
        ("sun and moon are both celestial", "celestial"),
        ("hot and cold are both temperature", "temperature"),
        ("big and small are both size", "size"),
        
        # Properties
        ("the ocean is deep", "deep"),
        ("the ocean is wide", "wide"),
        ("the mountain is tall", "tall"),
        ("the mountain is high", "high"),
        ("the human brain thinks", "thinks"),
        
        # Comparisons
        ("hot is the opposite of cold", "cold"),
        ("fast is the opposite of slow", "slow"),
        ("light is the opposite of dark", "dark"),
        ("big is the opposite of small", "small"),
    ]
    
    print(f"\n[2] Training {len(LESSON_3)} new examples...")
    
    for sentence, expected in LESSON_3:
        add_sentence(lnn, sentence)
        print(f"    + {sentence}")
    
    print(f"\n    Network: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    print(f"\n[3] Testing Lesson 3 concepts...")
    
    test_prompts = [
        ("dog and cat are both", "animals"),
        ("fish and birds are both", "animals"),
        ("the ocean is", "deep"),
        ("hot is the opposite of", "cold"),
        ("the human brain", "thinks"),
    ]
    
    passed = 0
    for prompt, expected in test_prompts:
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=3)
        
        if candidates:
            top = candidates[0]["word"]
            match = top.lower() == expected.lower()
            status = "✓" if match else "✗"
            print(f"    {status} '{prompt}' -> {top} (expected: {expected})")
            if match:
                passed += 1
        else:
            print(f"    ✗ '{prompt}' -> no output")
    
    # Test benchmark prompts
    print(f"\n[4] Testing benchmark concepts...")
    
    benchmark_prompts = [
        ("the sun", "sky"),
        ("water flows", "down"),
        ("the cat", "meows"),
        ("light travels", "fast"),
        ("fire causes", "burn"),
        ("friction creates", "heat"),
        ("gravity causes", "fall"),
        ("cold and hot are both", "temperature"),
    ]
    
    bench_passed = 0
    for prompt, expected in benchmark_prompts:
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=3)
        
        if candidates:
            top = candidates[0]["word"]
            match = top.lower() == expected.lower()
            status = "✓" if match else "✗"
            print(f"    {status} '{prompt}' -> {top}")
            if match:
                bench_passed += 1
    
    print(f"\n[5] Results: Lesson 3: {passed}/{len(test_prompts)}, Benchmark: {bench_passed}/{len(benchmark_prompts)}")
    
    with open("/Users/tyarc/github/lrn/checkpoints/lesson3.pkl", 'wb') as f:
        pickle.dump({'lnn': lnn, 'lesson3_passed': passed, 'benchmark_passed': bench_passed}, f)
    
    return passed >= len(test_prompts) - 1


if __name__ == "__main__":
    main()