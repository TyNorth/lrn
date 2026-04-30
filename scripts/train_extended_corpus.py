#!/usr/bin/env python3
"""
Train with extended corpus - Improve benchmark parity
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import pickle
import json
import os


def main():
    print("=" * 60)
    print("Training with Extended Corpus")
    print("=" * 60)

    from lrn import LatticeNN, propagate
    from lrn.corpus import CorpusExpander
    from lrn.corpus_expanded import ExtendedCorpusExpander
    from lrn.training import add_sentence, add_negative_sentence, add_identity_anchor

    # Load Phase 4 checkpoint as base
    with open("/Users/tyarc/github/lrn/checkpoints/phase4.pkl", 'rb') as f:
        data = pickle.load(f)
    lnn = data['lnn']

    print(f"\n[1] Loaded base checkpoint, nodes: {len(lnn.nodes)}")

    # Use extended corpus
    expander = ExtendedCorpusExpander()
    sentences = expander.expand(target_count=2000)
    negatives = expander.get_all_negatives()

    print(f"\n[2] Generated {len(sentences)} sentences (extended corpus)")
    print(f"    - Core: {len(expander.core_sentences)}")
    print(f"    - Expanded: {len(sentences) - len(expander.core_sentences)}")
    print(f"    - Negatives: {len(negatives)}")

    print("\n[3] Adding identity:self anchor...")
    add_identity_anchor(lnn)

    print("\n[4] Training on extended corpus...")
    initial_nodes = len(lnn.nodes)
    initial_springs = len(lnn.springs)

    for i, sentence in enumerate(sentences):
        add_sentence(lnn, sentence, reality=1.0)
        if (i + 1) % 500 == 0:
            print(f"    Processed {i+1}/{len(sentences)} sentences...")

    for neg in negatives:
        add_negative_sentence(lnn, neg)

    final_nodes = len(lnn.nodes)
    final_springs = len(lnn.springs)

    print(f"\n[5] Training complete:")
    print(f"    - Nodes added: {final_nodes - initial_nodes}")
    print(f"    - Springs added: {final_springs - initial_springs}")
    print(f"    - Total nodes: {final_nodes}")
    print(f"    - Total springs: {final_springs}")

    # Run benchmark
    print("\n[6] Running benchmark...")

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

    from lrn.generate import generate

    def score_prompt(prompt, expected):
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=5)
        if not candidates:
            return 0, False
        top = candidates[0]["word"]
        match = top.lower() == expected.lower()
        score = 5 if match else 2
        return score, match

    total_score = 0
    max_score = 25 * len(BENCHMARK_PROMPTS)
    matches = 0
    results = []

    for i, (prompt, expected) in enumerate(BENCHMARK_PROMPTS):
        score, match = score_prompt(prompt, expected)
        total_score += score
        if match:
            matches += 1
        results.append({"prompt": prompt, "expected": expected, "score": score, "match": match})
        print(f"    {i+1}. {prompt} -> {score}/25")

    parity = (total_score / max_score) * 100 if max_score > 0 else 0

    print(f"\n[7] Benchmark Results:")
    print(f"    Total: {total_score}/{max_score} ({parity:.1f}%)")
    print(f"    Matches: {matches}/{len(BENCHMARK_PROMPTS)}")

    # Save checkpoint
    with open("/Users/tyarc/github/lrn/checkpoints/phase7_extended.pkl", 'wb') as f:
        pickle.dump({'lnn': lnn, 'results': results, 'total_score': total_score, 'parity': parity}, f)

    # Save sys_test
    sys_test_results = {
        "phase": "extended_corpus",
        "test_type": "benchmark_improved",
        "corpus_size": len(sentences),
        "results": results,
        "total_score": total_score,
        "max_score": max_score,
        "parity_percent": round(parity, 1),
        "exact_matches": matches,
    }

    with open("/Users/tyarc/github/lrn/sys_test/benchmark_improved.json", 'w') as f:
        json.dump(sys_test_results, f, indent=2)

    print(f"\n[8] Saved checkpoint and sys_test results")
    print("\n" + "=" * 60)
    print(f"Extended corpus training complete: {parity:.1f}% parity")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())