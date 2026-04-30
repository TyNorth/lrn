#!/usr/bin/env python3
"""
LRN Teacher - Experiment A: More Epochs on Diverse Training
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import pickle
import json
from lrn import LatticeNN, add_sentence, generate, propagate


def main():
    print("=" * 60)
    print("Experiment A: More Epochs on Diverse Training")
    print("=" * 60)
    
    lnn = LatticeNN()
    
    DIVERSE_TRAINING = [
        # Animals
        ("the dog barks", "barks"), ("dogs bark", "barks"), ("a dog barks", "barks"),
        ("the cat meows", "meows"), ("cats meow", "meows"), ("a cat meows", "meows"),
        ("the bird flies", "flies"), ("birds fly", "flies"), ("a bird flies", "flies"),
        ("the fish swims", "swims"), ("fishes swim", "swims"), ("a fish swims", "swims"),
        
        # Weather
        ("the sun shines", "shines"), ("sun shines bright", "bright"),
        ("the rain falls", "falls"), ("rain falls down", "down"),
        ("the wind blows", "blows"), ("wind blows hard", "hard"),
        
        # Actions
        ("fire causes burn", "burn"), ("fire causes heat", "heat"),
        ("friction creates heat", "heat"), ("friction creates", "heat"),
        ("gravity causes fall", "fall"), ("gravity causes", "fall"),
        
        # Science
        ("ice melts when", "when"), ("ice melts in heat", "heat"),
        ("cold and hot are both temperature", "temperature"),
        
        # Nature
        ("sky is blue", "blue"), ("the sky is blue", "blue"),
        ("grass is green", "green"), ("the grass is green", "green"),
        ("ocean is wide", "wide"), ("the ocean is wide", "wide"),
        
        # Body
        ("brain thinks", "thinks"), ("human brain thinks", "thinks"),
    ]
    
    # Train with MORE EPOCHS (5x repetition per example)
    EPOCHS = 5
    
    print(f"\n[1] Training {len(DIVERSE_TRAINING)} examples × {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        for sentence, expected in DIVERSE_TRAINING:
            add_sentence(lnn, sentence)
        print(f"    Epoch {epoch+1}/{EPOCHS} done")
    
    print(f"    Network: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    # Test
    print(f"\n[2] Testing...")
    
    TESTS = [
        # Original benchmark
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
        
        # Generalization
        ("dogs", "barks"),
        ("cat", "meows"),
        ("birds", "flies"),
    ]
    
    results = []
    passed = 0
    
    for prompt, expected in TESTS:
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=5)
        
        if candidates:
            top = candidates[0]["word"]
            match = top.lower() == expected.lower()
            status = "✓" if match else "✗"
            print(f"    {status} '{prompt}' -> {top} (expected: {expected})")
            results.append({"prompt": prompt, "expected": expected, "got": top, "match": match})
            if match:
                passed += 1
    
    parity = (passed / len(TESTS)) * 100
    print(f"\n[Results] {passed}/{len(TESTS)} ({parity:.0f}%)")
    
    with open("/Users/tyarc/github/lrn/sys_test/experiment_a_epochs.json", 'w') as f:
        json.dump({
            "experiment": "a_epochs",
            "epochs": EPOCHS,
            "passed": passed,
            "total": len(TESTS),
            "parity": round(parity, 1),
            "results": results
        }, f, indent=2)


if __name__ == "__main__":
    main()