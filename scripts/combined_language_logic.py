#!/usr/bin/env python3
"""
LRN Teacher - Combined: Language + Logic
Start with best language model (Exp B), add logical reasoning
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import pickle
import json
from lrn import add_sentence, generate


def main():
    print("=" * 60)
    print("Combined: Language + Logic")
    print("=" * 60)
    
    # Load best language model (Experiment B)
    with open("/Users/tyarc/github/lrn/checkpoints/experiment_b.pkl", 'rb') as f:
        data = pickle.load(f)
    lnn = data['lnn']
    
    print(f"\n[1] Loaded language model: {len(lnn.nodes)} nodes")
    
    # Add logical reasoning
    LOGIC = [
        # Syllogisms
        ("all dogs are animals", "animals"),
        ("all birds are animals", "animals"),
        ("all fish are animals", "animals"),
        ("all humans are animals", "animals"),
        ("all roses are flowers", "flowers"),
        ("all cars are vehicles", "vehicles"),
        
        # If-Then
        ("if rain then wet", "wet"),
        ("if fire then heat", "heat"),
        ("if cold then freeze", "freeze"),
        ("if sun then light", "light"),
        
        # Comparisons
        ("bigger than small", "small"),
        ("faster than slow", "slow"),
        ("hotter than cold", "cold"),
    ]
    
    print(f"\n[2] Training {len(LOGIC)} logic examples...")
    for sentence, expected in LOGIC:
        for rep in range(5):
            add_sentence(lnn, sentence)
    
    print(f"    Network: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    # Test language
    print(f"\n[3] Testing language benchmark...")
    
    LANGUAGE = [
        ("the sun", "sky"),
        ("water flows", "down"),
        ("the cat", "meows"),
        ("fire causes", "burn"),
        ("gravity causes", "fall"),
        ("cold and hot are both", "temperature"),
        ("fish and birds are both", "animals"),
    ]
    
    lang_passed = 0
    for prompt, expected in LANGUAGE:
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=5)
        if candidates:
            top = candidates[0]["word"]
            match = top.lower() == expected.lower()
            status = "✓" if match else "✗"
            print(f"    {status} '{prompt}' -> {top}")
            if match:
                lang_passed += 1
    
    # Test logic
    print(f"\n[4] Testing logical reasoning...")
    
    LOGIC_TESTS = [
        ("all dogs are", "animals"),
        ("all roses are", "flowers"),
        ("if rain then", "wet"),
        ("if fire then", "heat"),
        ("bigger than", "small"),
    ]
    
    logic_passed = 0
    for prompt, expected in LOGIC_TESTS:
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=5)
        if candidates:
            top = candidates[0]["word"]
            match = top.lower() == expected.lower()
            status = "✓" if match else "✗"
            print(f"    {status} '{prompt}' -> {top} (expected: {expected})")
            if match:
                logic_passed += 1
    
    print(f"\n[Results] Language: {lang_passed}/{len(LANGUAGE)}, Logic: {logic_passed}/{len(LOGIC_TESTS)}")
    
    with open("/Users/tyarc/github/lrn/checkpoints/language_plus_logic.pkl", 'wb') as f:
        pickle.dump({'lnn': lnn}, f)
    
    with open("/Users/tyarc/github/lrn/sys_test/language_plus_logic.json", 'w') as f:
        json.dump({
            "language": {"passed": lang_passed, "total": len(LANGUAGE)},
            "logic": {"passed": logic_passed, "total": len(LOGIC_TESTS)},
            "network": {"nodes": len(lnn.nodes), "springs": len(lnn.springs)}
        }, f, indent=2)


if __name__ == "__main__":
    main()