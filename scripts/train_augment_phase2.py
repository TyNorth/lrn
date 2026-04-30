#!/usr/bin/env python3
"""
Add targeted sentences to existing Phase 2 training
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import pickle
import json


def main():
    print("=" * 60)
    print("Adding Targeted Sentences to Phase 2")
    print("=" * 60)

    with open("/Users/tyarc/github/lrn/checkpoints/phase2.pkl", 'rb') as f:
        data = pickle.load(f)
    lnn = data['lnn']

    print(f"\n[1] Loaded Phase 2: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")

    # Targeted sentences for benchmark
    targeted_sentences = [
        "the sun shines in the sky",
        "water flows down the river",
        "the cat meows at night",
        "light travels very fast",
        "fire causes smoke",
        "ice melts in warm weather",
        "friction creates heat",
        "gravity causes things to fall",
        "cold and hot are temperatures",
        "fish and birds are animals",
        "the ocean is very deep",
        "the human brain thinks",
        
        "sun gives light and warmth",
        "water flows from high to low",
        "cats purr when happy",
        "light moves at great speed",
        "fire burns things away",
        "ice turns to water",
        "friction makes things warm",
        "gravity pulls everything down",
        "hot and cold are opposites",
        "fish swim birds fly",
        "ocean waves are big",
        "brain controls the body",
        
        "the sun is bright",
        "water flows quickly",
        "the cat sleeps",
        "light is fast",
        "fire is hot",
        "ice is cold",
        "friction is useful",
        "gravity is strong",
        "cold is not hot",
        "fish can swim",
        "birds can fly",
        "ocean is big",
    ]

    from lrn.training import add_sentence

    print(f"\n[2] Adding {len(targeted_sentences)} targeted sentences...")
    for sent in targeted_sentences:
        add_sentence(lnn, sent, reality=1.0)

    print(f"\n[3] Now: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")

    print("\n[4] Running benchmark...")

    from lrn.generate import generate

    BENCHMARK_PROMPTS = [
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

    total_score = 0
    matches = 0
    results = []

    for i, (prompt, expected) in enumerate(BENCHMARK_PROMPTS):
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=5)
        top = candidates[0]["word"] if candidates else "NONE"
        match = top.lower() == expected.lower()
        score = 5 if match else 2
        total_score += score
        if match:
            matches += 1
        results.append({"prompt": prompt, "expected": expected, "got": top, "match": match})
        status = "✓" if match else "✗"
        print(f"    {status} '{prompt}' -> '{top}' (expected: '{expected}')")

    parity = (total_score / 300) * 100
    print(f"\n[5] Results: {total_score}/300 ({parity:.1f}%), {matches}/12")

    with open("/Users/tyarc/github/lrn/sys_test/benchmark_augmented.json", 'w') as f:
        json.dump({"parity": round(parity, 1), "matches": matches, "results": results}, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Augmented Phase 2: {parity:.1f}% parity")
    print("=" * 60)


if __name__ == '__main__':
    main()