#!/usr/bin/env python3
"""
LRN Teacher - Logical Reasoning: Transitivity & Chains
A > B, B > C → A > C (transitivity)
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import pickle
import json
from lrn import LatticeNN, add_sentence, generate, propagate


def main():
    print("=" * 60)
    print("Logical Reasoning - Transitivity & Chains")
    print("=" * 60)
    
    lnn = LatticeNN()
    
    # TRANSITIVITY chains
    TRANSITIVITY = [
        # A > B, B > C chains
        ("bigger than bigger", "bigger"),  # A > B > A (circular but ok)
        
        # Height chain
        ("mountain is taller than hill", "hill"),
        ("hill is taller than tree", "tree"),
        
        # Speed chain  
        ("jet is faster than plane", "plane"),
        ("plane is faster than car", "car"),
        
        # Value chain
        ("diamond is more valuable than gold", "gold"),
        ("gold is more valuable than silver", "silver"),
        
        # Age chain
        ("adult is older than child", "child"),
        ("child is older than baby", "baby"),
    ]
    
    # CAUSAL chains
    CAUSAL = [
        # Rain → wet → cold
        ("rain makes wet", "wet"),
        ("wet makes cold", "cold"),
        
        # Fire → heat → expand
        ("fire makes heat", "heat"),
        ("heat makes expand", "expand"),
        
        # Eat → energy → strong
        ("food gives energy", "energy"),
        ("energy makes strong", "strong"),
        
        # Study → knowledge → smart
        ("study gives knowledge", "knowledge"),
        ("knowledge makes smart", "smart"),
    ]
    
    # SEQUENCE
    SEQUENCE = [
        ("morning comes before noon", "noon"),
        ("noon comes before night", "night"),
        ("day comes before night", "night"),
        ("spring comes before summer", "summer"),
        ("summer comes before fall", "fall"),
    ]
    
    print(f"\n[1] Training transitivity...")
    for s, e in TRANSITIVITY:
        for rep in range(5):
            add_sentence(lnn, s)
    
    print(f"\n[2] Training causal chains...")
    for s, e in CAUSAL:
        for rep in range(5):
            add_sentence(lnn, s)
    
    print(f"\n[3] Training sequences...")
    for s, e in SEQUENCE:
        for rep in range(5):
            add_sentence(lnn, s)
    
    print(f"    Network: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    # Test
    print(f"\n[Testing transitivity...]")
    
    TRANS_TESTS = [
        ("mountain is taller than", "hill"),
        ("jet is faster than", "plane"),
        ("diamond is more valuable than", "gold"),
    ]
    
    trans_passed = 0
    for prompt, expected in TRANS_TESTS:
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=5)
        if candidates:
            top = candidates[0]["word"]
            match = top.lower() == expected.lower()
            status = "✓" if match else "✗"
            print(f"    {status} '{prompt}' -> {top}")
            if match:
                trans_passed += 1
    
    print(f"\n[Testing causal chains...]")
    
    CAUSAL_TESTS = [
        ("rain makes", "wet"),
        ("heat makes", "expand"),
        ("food gives", "energy"),
    ]
    
    causal_passed = 0
    for prompt, expected in CAUSAL_TESTS:
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=5)
        if candidates:
            top = candidates[0]["word"]
            match = top.lower() == expected.lower()
            status = "✓" if match else "✗"
            print(f"    {status} '{prompt}' -> {top}")
            if match:
                causal_passed += 1
    
    print(f"\n[Testing sequences...]")
    
    SEQ_TESTS = [
        ("morning comes before", "noon"),
        ("day comes before", "night"),
    ]
    
    seq_passed = 0
    for prompt, expected in SEQ_TESTS:
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=5)
        if candidates:
            top = candidates[0]["word"]
            match = top.lower() == expected.lower()
            status = "✓" if match else "✗"
            print(f"    {status} '{prompt}' -> {top}")
            if match:
                seq_passed += 1
    
    print(f"\n[Results] Transitivity: {trans_passed}/3, Causal: {causal_passed}/3, Sequence: {seq_passed}/2")
    
    with open("/Users/tyarc/github/lrn/checkpoints/logic_transitivity.pkl", 'wb') as f:
        pickle.dump({'lnn': lnn}, f)
    
    with open("/Users/tyarc/github/lrn/sys_test/logic_transitivity.json", 'w') as f:
        json.dump({
            "transitivity": trans_passed,
            "causal": causal_passed,
            "sequence": seq_passed,
            "network": {"nodes": len(lnn.nodes), "springs": len(lnn.springs)}
        }, f, indent=2)


if __name__ == "__main__":
    main()