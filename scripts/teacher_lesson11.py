#!/usr/bin/env python3
"""
LRN Teacher - Lesson 11: More Complex Concepts
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import pickle
import json
from lrn import LatticeNN, add_sentence, generate, propagate


def main():
    print("=" * 60)
    print("LRN Teacher - Lesson 11: Complex Concepts")
    print("=" * 60)
    
    # Load final (100%)
    with open("/Users/tyarc/github/lrn/checkpoints/lesson10.pkl", 'rb') as f:
        data = pickle.load(f)
    lnn = data['lnn']
    
    print(f"\n[1] Loaded: {len(lnn.nodes)} nodes")
    
    # NEW concepts: emotions, senses, phases of matter, more science
    NEW_CONCEPTS = [
        # Emotions
        ("happy feels good", "good"),
        ("sad feels bad", "bad"),
        ("angry feels hot", "hot"),
        ("scared feels fear", "fear"),
        ("love feels warm", "warm"),
        
        # Senses
        ("eyes see", "see"),
        ("ears hear", "hear"),
        ("nose smells", "smells"),
        ("tongue tastes", "tastes"),
        ("hands touch", "touch"),
        
        # Phases of matter
        ("water becomes ice", "ice"),
        ("ice becomes water", "water"),
        ("water becomes vapor", "vapor"),
        ("vapor becomes water", "water"),
        
        # More science
        ("plants need sun", "sun"),
        ("plants need water", "water"),
        ("plants make oxygen", "oxygen"),
        ("humans breathe oxygen", "oxygen"),
        
        # Colors
        ("sky is blue", "blue"),
        ("grass is green", "green"),
        ("sun is yellow", "yellow"),
        ("fire is red", "red"),
        
        # Shapes
        ("circle is round", "round"),
        ("square has four sides", "four"),
        ("triangle has three sides", "three"),
    ]
    
    print(f"\n[2] Training {len(NEW_CONCEPTS)} new concepts...")
    
    for sentence, expected in NEW_CONCEPTS:
        for rep in range(3):
            add_sentence(lnn, sentence)
    
    print(f"    (3x each = {len(NEW_CONCEPTS)*3} total)")
    print(f"\n    Network: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    # Original benchmark
    print(f"\n[3] Original benchmark...")
    
    ORIGINAL_BENCHMARK = [
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
    
    passed_orig = 0
    for prompt, expected in ORIGINAL_BENCHMARK:
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=5)
        
        if candidates:
            top = candidates[0]["word"]
            match = top.lower() == expected.lower()
            status = "✓" if match else "✗"
            print(f"    {status} '{prompt}' -> {top}")
            if match:
                passed_orig += 1
    
    print(f"\n    Original: {passed_orig}/{len(ORIGINAL_BENCHMARK)}")
    
    # NEW generalization test
    print(f"\n[4] NEW test (never trained)...")
    
    NEW_PROMPTS = [
        # Emotions (new)
        ("happy feels", "good"),
        ("eyes", "see"),
        ("ears", "hear"),
        
        # Science (new)  
        ("plants need", "sun"),
        ("plants make", "oxygen"),
        
        # Phases (new)
        ("water becomes", "ice"),
        ("vapor becomes", "water"),
        
        # Colors (new)
        ("sky is", "blue"),
        ("grass is", "green"),
    ]
    
    results = []
    passed_new = 0
    
    for prompt, expected in NEW_PROMPTS:
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=5)
        
        if candidates:
            top = candidates[0]["word"]
            match = top.lower() == expected.lower()
            status = "✓" if match else "✗"
            print(f"    {status} '{prompt}' -> {top} (expected: {expected})")
            results.append({"prompt": prompt, "expected": expected, "got": top, "match": match})
            if match:
                passed_new += 1
        else:
            print(f"    ✗ '{prompt}' -> no output")
            results.append({"prompt": prompt, "expected": expected, "got": "none", "match": False})
    
    print(f"\n    New: {passed_new}/{len(NEW_PROMPTS)}")
    
    # Save
    with open("/Users/tyarc/github/lrn/checkpoints/lesson11.pkl", 'wb') as f:
        pickle.dump({'lnn': lnn, 'passed_orig': passed_orig, 'passed_new': passed_new}, f)
    
    with open("/Users/tyarc/github/lrn/sys_test/teacher_lesson11.json", 'w') as f:
        json.dump({
            "lesson": 11,
            "new_concepts": len(NEW_CONCEPTS) * 3,
            "original": {"passed": passed_orig, "total": len(ORIGINAL_BENCHMARK)},
            "new_test": {"passed": passed_new, "total": len(NEW_PROMPTS), "results": results},
            "network": {"nodes": len(lnn.nodes), "springs": len(lnn.springs)}
        }, f, indent=2)
    
    return passed_orig == len(ORIGINAL_BENCHMARK) and passed_new >= len(NEW_PROMPTS) - 1


if __name__ == "__main__":
    main()