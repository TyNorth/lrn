#!/usr/bin/env python3
"""
LRN Teacher - Logical Reasoning: Negation, Contradictions, Paradoxes
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import pickle
import json
from lrn import LatticeNN, add_sentence, generate, propagate


def main():
    print("=" * 60)
    print("Logical Reasoning - Advanced: Negation & Paradox")
    print("=" * 60)
    
    lnn = LatticeNN()
    
    # NEGATION - "not" patterns
    NEGATION = [
        # Not statements
        ("the cat is not a dog", "dog"),
        ("birds are not fish", "fish"),
        ("fire is not cold", "cold"),
        ("light is not dark", "dark"),
        ("hot is not cold", "cold"),
        
        # None/no patterns
        ("no birds are fish", "fish"),
        ("no dogs are cats", "cats"),
        ("nothing is hotter than fire", "fire"),
        ("nobody is taller than sky", "sky"),
        
        # Opposites
        ("opposite of hot is cold", "cold"),
        ("opposite of light is dark", "dark"),
        ("opposite of big is small", "small"),
        ("opposite of fast is slow", "slow"),
    ]
    
    # CONTRADICTIONS - "but" / "however"
    CONTRADICT = [
        ("the sun is hot but the moon is cold", "cold"),
        ("birds fly but fish swim", "swim"),
        ("fire burns however ice freezes", "freezes"),
        ("day is bright but night is dark", "dark"),
    ]
    
    # PARADOXES - "all X are Y, but X is not Y"
    PARADOX = [
        ("all squares are rectangles but some squares are not rectangles", "rectangles"),  # actually true
        ("nothing is impossible but everything is possible", "possible"),
    ]
    
    # NECCESSARY vs POSSIBLE
    NECESSARY = [
        ("water is necessary for life", "life"),
        ("air is necessary for breathing", "breathing"),
        ("food is necessary for energy", "energy"),
        ("sun is necessary for plants", "plants"),
    ]
    
    print(f"\n[1] Training negation...")
    for s, e in NEGATION:
        for rep in range(5):
            add_sentence(lnn, s)
    
    print(f"    Training contradictions...")
    for s, e in CONTRADICT:
        for rep in range(5):
            add_sentence(lnn, s)
    
    print(f"    Training necessary...")
    for s, e in NECESSARY:
        for rep in range(5):
            add_sentence(lnn, s)
    
    print(f"    Network: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    # Test negation
    print(f"\n[Testing negation...]")
    
    NEG_TESTS = [
        ("cat is not a", "dog"),
        ("birds are not", "fish"),
        ("opposite of hot is", "cold"),
        ("opposite of big is", "small"),
    ]
    
    neg_passed = 0
    for prompt, expected in NEG_TESTS:
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=5)
        if candidates:
            top = candidates[0]["word"]
            match = top.lower() == expected.lower()
            status = "✓" if match else "✗"
            print(f"    {status} '{prompt}' -> {top}")
            if match:
                neg_passed += 1
    
    # Test necessary
    print(f"\n[Testing necessary...]")
    
    NEC_TESTS = [
        ("water is necessary for", "life"),
        ("air is necessary for", "breathing"),
        ("sun is necessary for", "plants"),
    ]
    
    nec_passed = 0
    for prompt, expected in NEC_TESTS:
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=5)
        if candidates:
            top = candidates[0]["word"]
            match = top.lower() == expected.lower()
            status = "✓" if match else "✗"
            print(f"    {status} '{prompt}' -> {top}")
            if match:
                nec_passed += 1
    
    # Test contradictions
    print(f"\n[Testing contradictions...]")
    
    CONTRA_TESTS = [
        ("sun is hot but moon is", "cold"),
        ("birds fly but fish", "swim"),
    ]
    
    contra_passed = 0
    for prompt, expected in CONTRA_TESTS:
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=5)
        if candidates:
            top = candidates[0]["word"]
            match = top.lower() == expected.lower()
            status = "✓" if match else "✗"
            print(f"    {status} '{prompt}' -> {top}")
            if match:
                contra_passed += 1
    
    print(f"\n[Results] Negation: {neg_passed}/4, Necessary: {nec_passed}/3, Contradictions: {contra_passed}/2")
    
    with open("/Users/tyarc/github/lrn/checkpoints/logic_advanced.pkl", 'wb') as f:
        pickle.dump({'lnn': lnn}, f)
    
    with open("/Users/tyarc/github/lrn/sys_test/logic_advanced.json", 'w') as f:
        json.dump({
            "negation": {"passed": neg_passed, "total": 4},
            "necessary": {"passed": nec_passed, "total": 3},
            "contradictions": {"passed": contra_passed, "total": 2},
            "network": {"nodes": len(lnn.nodes), "springs": len(lnn.springs)}
        }, f, indent=2)


if __name__ == "__main__":
    main()