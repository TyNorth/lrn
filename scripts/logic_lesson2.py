#!/usr/bin/env python3
"""
LRN Teacher - Logical Reasoning: If-Then & Comparisons
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import pickle
import json
from lrn import LatticeNN, add_sentence, generate, propagate


def main():
    print("=" * 60)
    print("Logical Reasoning - If-Then & Comparisons")
    print("=" * 60)
    
    lnn = LatticeNN()
    
    # IF-THEN patterns
    IF_THEN = [
        # If it rains, the ground gets wet
        ("if it rains then ground gets wet", "wet"),
        ("if it rains then wet", "wet"),
        ("it rains causes wet", "wet"),
        ("rain makes ground wet", "wet"),
        
        # If fire, then heat
        ("if fire then heat", "heat"),
        ("fire causes heat", "heat"),
        ("if hot then melt", "melt"),
        
        # If cold then freeze
        ("if cold then freeze", "freeze"),
        ("cold causes freeze", "freeze"),
        
        # If sun then light
        ("if sun then light", "light"),
        ("sun gives light", "light"),
        
        # If study then learn
        ("if study then learn", "learn"),
        ("study causes learn", "learn"),
        ("study leads to learn", "learn"),
        
        # If eat then full
        ("if eat then full", "full"),
        ("eating makes full", "full"),
    ]
    
    # COMPARISONS
    COMPARISONS = [
        # Size
        ("elephant is bigger than mouse", "mouse"),
        ("mouse is smaller than elephant", "elephant"),
        ("big is bigger than small", "small"),
        ("tall is bigger than short", "short"),
        
        # Speed
        ("jet is faster than car", "car"),
        ("car is slower than jet", "jet"),
        ("fast is faster than slow", "slow"),
        
        # Temperature
        ("fire is hotter than ice", "ice"),
        ("hot is hotter than cold", "cold"),
        
        # Value
        ("gold is more than silver", "silver"),
        ("more is greater than less", "less"),
    ]
    
    print(f"\n[1] Training {len(IF_THEN)} if-then examples...")
    for sentence, expected in IF_THEN:
        for rep in range(5):
            add_sentence(lnn, sentence)
    
    print(f"    Training {len(COMPARISONS)} comparison examples...")
    for sentence, expected in COMPARISONS:
        for rep in range(5):
            add_sentence(lnn, sentence)
    
    print(f"    Network: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    # Test
    print(f"\n[2] Testing if-then...")
    
    IF_TESTS = [
        ("if rain then", "wet"),
        ("if fire then", "heat"),
        ("if cold then", "freeze"),
        ("if sun then", "light"),
        ("if study then", "learn"),
        ("it rains causes", "wet"),
    ]
    
    if_passed = 0
    for prompt, expected in IF_TESTS:
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=5)
        
        if candidates:
            top = candidates[0]["word"]
            match = top.lower() == expected.lower()
            status = "✓" if match else "✗"
            print(f"    {status} '{prompt}' -> {top} (expected: {expected})")
            if match:
                if_passed += 1
    
    print(f"\n[3] Testing comparisons...")
    
    COMP_TESTS = [
        ("elephant is bigger than", "mouse"),
        ("fast is faster than", "slow"),
        ("hot is hotter than", "cold"),
    ]
    
    comp_passed = 0
    for prompt, expected in COMP_TESTS:
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=5)
        
        if candidates:
            top = candidates[0]["word"]
            match = top.lower() == expected.lower()
            status = "✓" if match else "✗"
            print(f"    {status} '{prompt}' -> {top} (expected: {expected})")
            if match:
                comp_passed += 1
    
    print(f"\n[Results] If-Then: {if_passed}/{len(IF_TESTS)}, Comparisons: {comp_passed}/{len(COMP_TESTS)}")
    
    with open("/Users/tyarc/github/lrn/checkpoints/logic2.pkl", 'wb') as f:
        pickle.dump({'lnn': lnn}, f)
    
    with open("/Users/tyarc/github/lrn/sys_test/logic_ifthen.json", 'w') as f:
        json.dump({
            "if_then": {"passed": if_passed, "total": len(IF_TESTS)},
            "comparisons": {"passed": comp_passed, "total": len(COMP_TESTS)},
            "network": {"nodes": len(lnn.nodes), "springs": len(lnn.springs)}
        }, f, indent=2)


if __name__ == "__main__":
    main()