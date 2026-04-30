#!/usr/bin/env python3
"""
Test complex generalization - phrases not seen in training
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import pickle
import json
from lrn import generate


def main():
    print("=" * 60)
    print("Generalization Test - Novel Combinations")
    print("=" * 60)
    
    # Load best (lesson 10 - 100% on original)
    with open("/Users/tyarc/github/lrn/checkpoints/lesson10.pkl", 'rb') as f:
        data = pickle.load(f)
    lnn = data['lnn']
    
    print(f"\n[Network] {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    # Test NOVEL combinations - never trained but should work via generalization
    print(f"\n[Testing novel combinations...]")
    
    NOVEL_TESTS = [
        # Combine known concepts in new ways
        ("the dog", "meows"),  # dog doesn't meow, should fail
        ("the dog", "barks"),  # dog barks - TRAINED
        ("the bird", "swims"),  # bird doesn't swim, should fail  
        ("the bird", "flies"),  # bird flies - TRAINED
        ("the fish", "flies"),  # fish doesn't fly, should fail
        ("the fish", "swims"),  # fish swims - TRAINED
        
        # Novel subjects with known actions
        ("the moon", "shines"),  # moon shines - similar to sun
        ("the stars", "shine"),  # stars shine - similar
        ("the wind", "burns"),  # wind doesn't burn - should fail
        ("the wind", "blows"),  # wind blows - TRAINED
        
        # Novel combinations
        ("the bee", "flies"),  # bee flies
        ("the snake", "crawls"),  # snake crawls
        ("the bear", "walks"),  # bear walks
    ]
    
    results = []
    passed = 0
    
    for prompt, expected in NOVEL_TESTS:
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=5)
        
        if candidates:
            top = candidates[0]["word"]
            match = top.lower() == expected.lower()
            status = "✓" if match else "✗"
            
            # Also show alternatives
            alts = [c["word"] for c in candidates[:3]]
            print(f"    {status} '{prompt}' -> {top} (expected: {expected})")
            print(f"        alternatives: {', '.join(alts)}")
            
            results.append({"prompt": prompt, "expected": expected, "got": top, "match": match})
            if match:
                passed += 1
    
    print(f"\n[Results] {passed}/{len(NOVEL_TESTS)}")
    
    # Now test with lesson 12 (more concepts)
    print("\n" + "="*50)
    print("With Lesson 12 (more concepts)...")
    
    with open("/Users/tyarc/github/lrn/checkpoints/lesson12.pkl", 'rb') as f:
        data = pickle.load(f)
    lnn2 = data['lnn']
    
    print(f"[Network] {len(lnn2.nodes)} nodes, {len(lnn2.springs)} springs")
    
    results2 = []
    passed2 = 0
    
    for prompt, expected in NOVEL_TESTS:
        tokens = prompt.lower().split()
        candidates = generate(lnn2, tokens, top_k=5)
        
        if candidates:
            top = candidates[0]["word"]
            match = top.lower() == expected.lower()
            status = "✓" if match else "✗"
            print(f"    {status} '{prompt}' -> {top} (expected: {expected})")
            results2.append({"prompt": prompt, "expected": expected, "got": top, "match": match})
            if match:
                passed2 += 1
    
    print(f"\n[Results L12] {passed2}/{len(NOVEL_TESTS)}")
    
    # Save results
    with open("/Users/tyarc/github/lrn/sys_test/generalization_test.json", 'w') as f:
        json.dump({
            "lesson10": {"passed": passed, "total": len(NOVEL_TESTS), "results": results},
            "lesson12": {"passed": passed2, "total": len(NOVEL_TESTS), "results": results2}
        }, f, indent=2)


if __name__ == "__main__":
    main()