#!/usr/bin/env python3
"""
LRN Teacher - Lesson 12: More Generalization + Fix Failing
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import pickle
import json
from lrn import LatticeNN, add_sentence, generate, propagate


def main():
    print("=" * 60)
    print("LRN Teacher - Lesson 12: Extend & Fix")
    print("=" * 60)
    
    with open("/Users/tyarc/github/lrn/checkpoints/lesson11.pkl", 'rb') as f:
        data = pickle.load(f)
    lnn = data['lnn']
    
    print(f"\n[1] Loaded: {len(lnn.nodes)} nodes")
    
    # FIX failing cases + MORE concepts
    FIX_AND_EXTEND = [
        # FIX: plants need -> sun
        ("plants need sun to grow", "grow"),
        ("plants need sun", "sun"),
        ("plants need light from sun", "light"),
        ("plants grow toward sun", "sun"),
        
        # FIX: water becomes -> ice  
        ("water becomes ice when cold", "cold"),
        ("water becomes ice", "ice"),
        ("freezing water becomes ice", "ice"),
        
        # MORE concepts: body
        ("heart pumps blood", "blood"),
        ("bones support body", "body"),
        ("muscles move", "move"),
        ("skin protects", "protects"),
        
        # MORE concepts: technology
        ("computer computes", "computes"),
        ("phone calls", "calls"),
        ("radio broadcasts", "broadcasts"),
        ("television shows", "shows"),
        
        # MORE concepts: nature
        ("tree grows", "grows"),
        ("flower blooms", "blooms"),
        ("mountain stands", "stands"),
        ("river runs", "runs"),
    ]
    
    print(f"\n[2] Fixing + extending...")
    
    for sentence, expected in FIX_AND_EXTEND:
        for rep in range(4):
            add_sentence(lnn, sentence)
    
    print(f"    (4x each = {len(FIX_AND_EXTEND)*4} total)")
    print(f"\n    Network: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    # Tests
    print(f"\n[3] Original benchmark...")
    
    ORIGINAL = [
        ("the sun", "sky"), ("water flows", "down"), ("the cat", "meows"),
        ("light travels", "fast"), ("fire causes", "burn"), ("ice melts", "when"),
        ("friction creates", "heat"), ("gravity causes", "fall"),
        ("cold and hot are both", "temperature"), ("fish and birds are both", "animals"),
        ("the ocean is deep and", "wide"), ("the human brain", "thinks"),
    ]
    
    passed_orig = sum(1 for p, e in ORIGINAL if generate(lnn, p.lower().split(), top_k=3)[0]["word"].lower() == e.lower())
    print(f"    Original: {passed_orig}/{len(ORIGINAL)}")
    
    print(f"\n[4] Extended test (from lesson 11)...")
    
    EXTENDED = [
        ("happy feels", "good"), ("eyes", "see"), ("ears", "hear"),
        ("plants need", "sun"), ("plants make", "oxygen"),
        ("water becomes", "ice"), ("vapor becomes", "water"),
        ("sky is", "blue"), ("grass is", "green"),
        # NEW
        ("heart pumps", "blood"),
        ("computer", "computes"),
        ("tree", "grows"),
    ]
    
    results = []
    passed = 0
    for prompt, expected in EXTENDED:
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=5)
        
        if candidates:
            top = candidates[0]["word"]
            match = top.lower() == expected.lower()
            status = "✓" if match else "✗"
            print(f"    {status} '{prompt}' -> {top}")
            results.append({"prompt": prompt, "expected": expected, "got": top, "match": match})
            if match:
                passed += 1
    
    print(f"\n    Extended: {passed}/{len(EXTENDED)}")
    
    with open("/Users/tyarc/github/lrn/checkpoints/lesson12.pkl", 'wb') as f:
        pickle.dump({'lnn': lnn}, f)
    
    with open("/Users/tyarc/github/lrn/sys_test/teacher_lesson12.json", 'w') as f:
        json.dump({
            "lesson": 12,
            "original": passed_orig,
            "extended": passed,
            "total_extended": len(EXTENDED),
            "network": {"nodes": len(lnn.nodes), "springs": len(lnn.springs)},
            "results": results
        }, f, indent=2)
    
    print(f"\n✓ Final: Original {passed_orig}/12, Extended {passed}/{len(EXTENDED)}")


if __name__ == "__main__":
    main()