#!/usr/bin/env python3
"""
Training with REPETITION - the key to Hebbian strengthening
Run through the corpus multiple times to build strong springs
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import pickle
import json


def main():
    print("=" * 60)
    print("Training with Repetition (Hebbian Strengthening)")
    print("=" * 60)

    from lrn import LatticeNN, generate
    from lrn.corpus import CorpusExpander
    from lrn.training import add_sentence, add_negative_sentence, add_identity_anchor

    lnn = LatticeNN()
    print("\n[1] Created LatticeNN")

    # Get corpus
    expander = CorpusExpander()
    sentences = expander.expand(target_count=500)
    negatives = expander.get_all_negatives()

    print(f"\n[2] Corpus: {len(sentences)} sentences, {len(negatives)} negatives")

    # Add identity anchor
    add_identity_anchor(lnn)

    # TRAINING WITH REPETITION
    REPETITIONS = 10
    print(f"\n[3] Training with {REPETITIONS} repetitions...")

    for rep in range(REPETITIONS):
        print(f"    Pass {rep + 1}/{REPETITIONS}...", end=" ")
        
        for sentence in sentences:
            add_sentence(lnn, sentence, reality=1.0)
        
        for neg in negatives:
            add_negative_sentence(lnn, neg)
        
        print(f"done. Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")

    print(f"\n[4] Final: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")

    # Count strong springs (exposure > 3)
    strong_springs = sum(1 for sp in lnn.springs.values() if sp.exposure_count > 3)
    print(f"    Strong springs (exposure > 3): {strong_springs}")

    # Run benchmark
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
        results.append({"prompt": prompt, "expected": expected, "got": top, "match": match})
        status = "✓" if match else "✗"
        print(f"    {status} '{prompt}' -> '{top}' (expected: '{expected}')")

    parity = (total_score / 300) * 100
    print(f"\n[6] Results: {total_score}/300 ({parity:.1f}%), {matches}/12 matches")

    # Save
    with open("/Users/tyarc/github/lrn/checkpoints/phase7_repetition.pkl", 'wb') as f:
        pickle.dump({'lnn': lnn, 'results': results, 'repetitions': REPETITIONS}, f)

    with open("/Users/tyarc/github/lrn/sys_test/benchmark_repetition.json", 'w') as f:
        json.dump({
            "repetitions": REPETITIONS,
            "nodes": len(lnn.nodes),
            "springs": len(lnn.springs),
            "strong_springs": strong_springs,
            "results": results,
            "total_score": total_score,
            "parity": round(parity, 1)
        }, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Repetition training: {parity:.1f}% parity")
    print("=" * 60)


if __name__ == '__main__':
    main()