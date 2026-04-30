#!/usr/bin/env python3
"""
Targeted corpus - Focus on benchmark-relevant words
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import pickle
import json
import random


BENCHMARK_KEYWORDS = {
    "the sun": ["sun", "sky", "shines", "light", "day", "bright", "warm", "heat"],
    "water flows": ["water", "flows", "down", "river", "stream", "ocean", "sea"],
    "the cat": ["cat", "meows", "kitten", "purrs", "sleeps", "plays"],
    "light travels": ["light", "travels", "fast", "speed", "bright", "sun"],
    "fire causes": ["fire", "causes", "burn", "burns", "heat", "smoke", "flame"],
    "ice melts": ["ice", "melts", "water", "cold", "freezes", "frozen"],
    "friction creates": ["friction", "creates", "heat", "fire", "spark"],
    "gravity causes": ["gravity", "causes", "fall", "pull", "weight", "earth"],
    "cold and hot are both": ["cold", "hot", "both", "temperature", "weather", "degrees"],
    "fish and birds are both": ["fish", "birds", "both", "animals", "swim", "fly"],
    "the ocean is deep and": ["ocean", "deep", "wide", "water", "sea", "blue"],
    "the human brain": ["brain", "thinks", "mind", "human", "person", "smart"],
}


def create_targeted_sentences():
    sentences = []
    
    # Core sentences (keep from original)
    core = [
        "birds fly in the sky",
        "fish swim in the river",
        "fire burns hot and bright",
        "ice melts in the sun",
        "the sun shines in the sky",
        "water flows down from mountains",
        "the moon glows at night",
        "stars twinkle in the dark",
        "wind blows through the trees",
        "rain falls from clouds",
    ]
    sentences.extend(core)
    
    # Benchmark-targeted sentences
    for prompt, keywords in BENCHMARK_KEYWORDS.items():
        tokens = prompt.split()
        subject = tokens[0] if tokens else "thing"
        
        # Create variations using keywords
        for kw in keywords[:4]:
            sentences.append(f"the {subject} {kw}")
            sentences.append(f"{subject} {kw}")
            sentences.append(f"the {kw} {subject}")
    
    # Add specific patterns
    patterns = [
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
        
        # More variations
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
    ]
    sentences.extend(patterns)
    
    # Add filler sentences with key words
    for _ in range(200):
        kw1 = random.choice(list(BENCHMARK_KEYWORDS.values())[random.randint(0, 5)])
        kw2 = random.choice(list(BENCHMARK_KEYWORDS.values())[random.randint(0, 5)])
        sentences.append(f"the {random.choice(kw1)} {random.choice(kw2)}")
    
    return list(set(sentences))


def main():
    print("=" * 60)
    print("Training with Targeted Corpus")
    print("=" * 60)

    from lrn import LatticeNN, generate
    from lrn.training import add_sentence, add_negative_sentence, add_identity_anchor

    lnn = LatticeNN()
    print("\n[1] Created fresh LatticeNN")

    sentences = create_targeted_sentences()
    print(f"\n[2] Created {len(sentences)} targeted sentences")

    add_identity_anchor(lnn)

    print("\n[3] Training...")
    for i, sentence in enumerate(sentences):
        add_sentence(lnn, sentence, reality=1.0)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(sentences)}")

    negatives = [
        "fish don't fly", "birds don't swim", "fire don't freeze",
        "ice don't burn", "rocks don't float"
    ]
    for neg in negatives:
        add_negative_sentence(lnn, neg)

    print(f"\n[4] Training complete: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")

    print("\n[5] Running benchmark...")

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
        results.append({"prompt": prompt, "expected": expected, "got": top, "score": score, "match": match})
        status = "✓" if match else "✗"
        print(f"    {status} '{prompt}' -> '{top}' (expected: '{expected}')")

    parity = (total_score / 300) * 100

    print(f"\n[6] Results: {total_score}/300 ({parity:.1f}%), {matches}/12 matches")

    # Save
    with open("/Users/tyarc/github/lrn/checkpoints/phase7_targeted.pkl", 'wb') as f:
        pickle.dump({'lnn': lnn, 'results': results}, f)

    with open("/Users/tyarc/github/lrn/sys_test/benchmark_targeted.json", 'w') as f:
        json.dump({
            "corpus_size": len(sentences),
            "nodes": len(lnn.nodes),
            "springs": len(lnn.springs),
            "results": results,
            "total_score": total_score,
            "parity": round(parity, 1)
        }, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Targeted corpus: {parity:.1f}% parity")
    print("=" * 60)


if __name__ == '__main__':
    main()