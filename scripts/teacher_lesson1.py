#!/usr/bin/env python3
"""
LRN Teacher Curriculum - Structured Learning
Lesson 1: Basic Vocabulary (nouns, verbs)
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_sentence, generate, propagate


def main():
    print("=" * 60)
    print("LRN Teacher - Lesson 1: Basic Vocabulary")
    print("=" * 60)
    
    lnn = LatticeNN()
    
    # Curated vocabulary - high frequency, clear concepts
    # Each pair: (subject, relationship, object)
    LESSON_1 = [
        # Basic nouns
        ("the dog barks", "dog", "barks"),
        ("the cat meows", "cat", "meows"),
        ("the bird flies", "bird", "flies"),
        ("the fish swims", "fish", "swims"),
        ("the sun shines", "sun", "shines"),
        ("the rain falls", "rain", "falls"),
        ("the wind blows", "wind", "blows"),
        ("the fire burns", "fire", "burns"),
        ("the water flows", "water", "flows"),
        ("the ice freezes", "ice", "freezes"),
    ]
    
    print(f"\n[1] Training {len(LESSON_1)} curated examples...")
    
    for sentence, noun, verb in LESSON_1:
        add_sentence(lnn, sentence)
        print(f"    ✓ {sentence}")
    
    print(f"\n    Network: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    print(f"\n[2] Testing Lesson 1 concepts...")
    
    # Test prompts from training
    test_prompts = [
        ("the dog", "barks"),
        ("the cat", "meows"),
        ("the bird", "flies"),
        ("the fish", "swims"),
        ("the sun", "shines"),
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
    
    print(f"\n[3] Lesson 1 Results: {passed}/{len(test_prompts)}")
    
    # Save checkpoint
    import pickle
    with open("/Users/tyarc/github/lrn/checkpoints/lesson1.pkl", 'wb') as f:
        pickle.dump({'lnn': lnn, 'passed': passed, 'total': len(test_prompts)}, f)
    
    print(f"\n[4] Saved checkpoint: checkpoints/lesson1.pkl")
    
    return passed >= len(test_prompts) - 1


if __name__ == "__main__":
    main()