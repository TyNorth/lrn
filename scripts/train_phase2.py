#!/usr/bin/env python3
"""
Train Phase 2 - Training Pipeline
Goal: Process corpus, form springs, build n-gram table
"""
import sys
import os
import pickle
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN
from lrn.corpus import CorpusExpander
from lrn.training import add_sentence, add_negative_sentence, add_identity_anchor, train_corpus


def main():
    print("=" * 60)
    print("Phase 2: Training Pipeline - Training")
    print("=" * 60)

    lnn = LatticeNN()
    print("\n[1] Created empty LatticeNN")

    expander = CorpusExpander()
    sentences = expander.expand(target_count=500)
    negatives = expander.get_all_negatives()

    print(f"[2] Generated {len(sentences)} sentences from templates")
    print(f"    - Core sentences: {len(expander.core_sentences)}")
    print(f"    - Expanded sentences: {len(sentences) - len(expander.core_sentences)}")
    print(f"    - Negative sentences: {len(negatives)}")

    print("\n[3] Adding identity:self anchor...")
    add_identity_anchor(lnn)

    print("\n[4] Training on corpus...")
    stats = train_corpus(lnn, sentences, negatives, reality=1.0)

    print(f"\n[5] Training Statistics:")
    print(f"    - Sentences processed: {stats['sentences_processed']}")
    print(f"    - Nodes created: {stats['nodes_created']}")
    print(f"    - Springs created: {stats['springs_created']}")
    print(f"    - Trigrams stored: {stats['trigrams_added']}")
    print(f"    - Total nodes: {len(lnn.nodes)}")
    print(f"    - Total springs: {len(lnn.springs)}")

    print("\n[6] Sample nodes (first 10):")
    sample_nodes = list(lnn.nodes.keys())[:10]
    for node_name in sample_nodes:
        node = lnn.nodes[node_name]
        print(f"    {node_name}: act={node.activation}, roles={node.role_counts}")

    print("\n[7] Sample springs (first 5):")
    sample_springs = list(lnn.springs.items())[:5]
    for (a, b), sp in sample_springs:
        print(f"    {a}-{b}: k={sp.stiffness}, tau={sp.tau}")

    print("\n[8] Sample trigrams (first 5):")
    sample_trigrams = list(lnn.trigrams.items())[:5]
    for gram, count in sample_trigrams:
        print(f"    {gram}: {count}")

    os.makedirs("/Users/tyarc/github/lrn/checkpoints", exist_ok=True)
    checkpoint_path = "/Users/tyarc/github/lrn/checkpoints/phase2.pkl"

    with open(checkpoint_path, 'wb') as f:
        pickle.dump({
            'lnn': lnn,
            'expander': expander,
            'stats': stats,
            'sentences': sentences,
            'negatives': negatives
        }, f)

    print(f"\n[9] Checkpoint saved: {checkpoint_path}")
    print(f"\n{'='*60}")
    print(f"Phase 2 Complete")
    print(f"  Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}, Trigrams: {len(lnn.trigrams)}")
    print(f"{'='*60}")

    return 0


if __name__ == '__main__':
    sys.exit(main())