#!/usr/bin/env python3
"""
LRN Teacher - Logical Reasoning Lesson 1: Basic Syllogisms
"All X are Y, Z is X → Z is Y"
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import pickle
import json
from lrn import LatticeNN, add_sentence, generate, propagate


def main():
    print("=" * 60)
    print("Logical Reasoning - Syllogisms")
    print("=" * 60)
    
    lnn = LatticeNN()
    
    # SYLLOGISMS: All A are B, X is A → X is B
    # Format: "all dogs are animals" + "the dog is a dog" → "the dog is an animal"
    
    SYLLOGISMS = [
        # All dogs are animals
        ("all dogs are animals", "animals"),
        ("dogs are animals", "animals"),
        ("a dog is an animal", "animal"),
        
        # All birds are animals  
        ("all birds are animals", "animals"),
        ("birds are animals", "animals"),
        ("a bird is an animal", "animal"),
        
        # All fish are animals
        ("all fish are animals", "animals"),
        
        # All cats are animals
        ("all cats are animals", "animals"),
        
        # All humans are animals
        ("all humans are animals", "animals"),
        ("humans are animals", "animals"),
        
        # All roses are flowers
        ("all roses are flowers", "flowers"),
        ("roses are flowers", "flowers"),
        
        # All cars are vehicles
        ("all cars are vehicles", "vehicles"),
        
        # All apples are fruit
        ("all apples are fruit", "fruit"),
    ]
    
    print(f"\n[1] Training {len(SYLLOGISMS)} syllogism examples...")
    
    for sentence, expected in SYLLOGISMS:
        for rep in range(5):
            add_sentence(lnn, sentence)
    
    print(f"    Network: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    # Test - syllogistic reasoning
    print(f"\n[2] Testing syllogisms...")
    
    TESTS = [
        # Direct training
        ("all dogs are", "animals"),
        ("birds are", "animals"),
        
        # Should infer
        ("the dog is an", "animal"),  # dog → animal via "dogs are animals"
        ("a bird is an", "animal"),
        ("humans are", "animals"),
        ("roses are", "flowers"),
        ("apples are", "fruit"),
        
        # Novel
        ("all elephants are", None),  # what?
        ("the elephant is an", None),
    ]
    
    results = []
    for prompt, expected in TESTS:
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=5)
        
        if candidates:
            top = candidates[0]["word"]
            if expected:
                match = top.lower() == expected.lower()
                status = "✓" if match else "✗"
                print(f"    {status} '{prompt}' -> {top} (expected: {expected})")
                results.append({"prompt": prompt, "got": top, "expected": expected, "match": match})
            else:
                print(f"    ? '{prompt}' -> {top}")
                results.append({"prompt": prompt, "got": top, "expected": "unknown"})
    
    # Count matches
    matches = sum(1 for r in results if r.get("match", False))
    print(f"\n[Results] {matches}/{len([r for r in results if 'expected' in r and r['expected']])}")
    
    with open("/Users/tyarc/github/lrn/checkpoints/logic1.pkl", 'wb') as f:
        pickle.dump({'lnn': lnn, 'results': results}, f)
    
    with open("/Users/tyarc/github/lrn/sys_test/logic_syllogisms.json", 'w') as f:
        json.dump({
            "lesson": "logic_syllogisms",
            "training": len(SYLLOGISMS) * 5,
            "results": results,
            "network": {"nodes": len(lnn.nodes), "springs": len(lnn.springs)}
        }, f, indent=2)


if __name__ == "__main__":
    main()