#!/usr/bin/env python3
"""
LRN Teacher Curriculum - Lesson 2: Simple Relationships
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import pickle
from lrn import LatticeNN, add_sentence, generate, propagate


def main():
    print("=" * 60)
    print("LRN Teacher - Lesson 2: Simple Relationships")
    print("=" * 60)
    
    # Load Lesson 1
    with open("/Users/tyarc/github/lrn/checkpoints/lesson1.pkl", 'rb') as f:
        data = pickle.load(f)
    lnn = data['lnn']
    
    print(f"\n[1] Loaded Lesson 1: {len(lnn.nodes)} nodes")
    
    # New concepts: spatial, causal, temporal relationships
    LESSON_2 = [
        # Spatial
        ("the bird is in the sky", "sky"),
        ("the fish is in the water", "water"),
        ("the cat is on the floor", "floor"),
        ("the book is on the table", "table"),
        ("the sun is in the sky", "sky"),
        
        # Causality
        ("fire causes burn", "burn"),
        ("rain causes wet", "wet"),
        ("cold causes freeze", "freeze"),
        ("heat causes melt", "melt"),
        ("sun causes light", "light"),
        
        # Actions
        ("water flows down", "down"),
        ("light travels fast", "fast"),
        ("time passes slow", "slow"),
        ("birds fly south", "south"),
        ("fish swim deep", "deep"),
    ]
    
    print(f"\n[2] Training {len(LESSON_2)} new examples...")
    
    for sentence, expected in LESSON_2:
        add_sentence(lnn, sentence)
        print(f"    + {sentence}")
    
    print(f"\n    Network: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    print(f"\n[3] Testing Lesson 2 concepts...")
    
    test_prompts = [
        ("the bird is in the", "sky"),
        ("fire causes", "burn"),
        ("water flows", "down"),
        ("light travels", "fast"),
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
    
    # Also test Lesson 1 retention
    print(f"\n[4] Testing Lesson 1 retention...")
    lesson1_prompts = [
        ("the dog", "barks"),
        ("the cat", "meows"),
    ]
    
    retained = 0
    for prompt, expected in lesson1_prompts:
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=3)
        
        if candidates:
            top = candidates[0]["word"]
            match = top.lower() == expected.lower()
            status = "✓" if match else "✗"
            print(f"    {status} '{prompt}' -> {top}")
            if match:
                retained += 1
    
    print(f"\n[5] Results: Lesson 2: {passed}/{len(test_prompts)}, Retention: {retained}/{len(lesson1_prompts)}")
    
    with open("/Users/tyarc/github/lrn/checkpoints/lesson2.pkl", 'wb') as f:
        pickle.dump({'lnn': lnn, 'lesson2_passed': passed, 'lesson1_retained': retained}, f)
    
    return passed >= len(test_prompts) - 1


if __name__ == "__main__":
    main()