"""
Full Pipeline: Sensory Grounding → 3rd Grade with Grammar Spiral
Builds complete educational lattice from interoceptive origins to 3rd grade.
Tests grammar integration and generation quality at each level.

Usage:
    python3 full_sensory_to_third_grade.py
"""
import sys
import os
import pickle
import time

sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.sensory_grounding import sensory_grounding
from lrn.sensory_interoceptive import get_sensory_lessons
from lrn.physical_manipulation import physical_manipulation
from lrn.social_interaction import social_interaction
from lrn.trainer import train, optimal_rem, prune_springs
from lrn.fast_trainer import train_fast
from lrn.grammar_lessons import get_grammar_lessons
from lrn.grammar_lessons_2nd import get_grammar_lessons as get_2nd_lessons
from lrn.grammar_lessons_3rd import get_grammar_lessons as get_3rd_lessons
from lrn.corpora import get_corpus, get_corpus_info, AVAILABLE_LEVELS
from lrn.generate import generate_sequence


CHECKPOINT_DIR = '/Users/tyarc/github/lrn/checkpoints/iterative'
REPORT_DIR = '/Users/tyarc/github/lrn/reports/iterative'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def save_checkpoint(lnn, prefix):
    path = f"{CHECKPOINT_DIR}/lattice_{prefix}.pkl"
    with open(path, 'wb') as f:
        pickle.dump({
            'nodes': lnn.nodes,
            'springs': lnn.springs,
            'trigrams': lnn.trigrams,
        }, f)
    print(f"  Saved: {path}")
    return path


def load_checkpoint(prefix):
    path = f"{CHECKPOINT_DIR}/lattice_{prefix}.pkl"
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        lnn = LatticeNN()
        lnn.nodes = data['nodes']
        lnn.springs = data['springs']
        lnn.trigrams = data.get('trigrams', {})
        print(f"  Loaded: {path}")
        return lnn
    return None


def test_generation(lnn, prompt, num_tokens=6):
    result = generate_sequence(lnn, prompt, max_tokens=num_tokens)
    return result


def run_pipeline():
    print("=" * 70)
    print("FULL PIPELINE: SENSORY GROUNDING → 3RD GRADE")
    print("=" * 70)
    
    start_time = time.time()
    results = {}
    
    # ====================================================================
    # PHASE 0: SENSORY GROUNDING (foundation for everything)
    # ====================================================================
    print("\n[PHASE 0] SENSORY GROUNDING")
    print("-" * 70)
    print("  Interoceptive + 12 external modalities")
    
    lnn = sensory_grounding(verbose=False)
    print(f"  After sensory: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    # Physical manipulation simulation
    print("\n  Physical manipulation...")
    physical_manipulation(lnn, verbose=False)
    print(f"  After physical: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    # Social interaction simulation
    print("\n  Social interaction...")
    social_interaction(lnn, verbose=False)
    print(f"  After social: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    save_checkpoint(lnn, "sensory")
    
    # ====================================================================
    # PHASE 0.5: SENSORY/INTEROCEPTIVE CURRICULUM (K-3rd)
    # ====================================================================
    print("\n[PHASE 0.5] SENSORY & INTEROCEPTIVE LESSONS")
    print("-" * 70)
    
    for grade in [0, 1, 2, 3]:
        lessons = get_sensory_lessons(grade)
        if not lessons:
            continue
        sentences = []
        for lesson in lessons:
            sentences.extend(lesson.get("sentences", []))
        if sentences:
            print(f"  Grade {grade}: {len(sentences)} sensory sentences")
            train_fast(lnn, sentences, reps=15, verbose=False)
    
    save_checkpoint(lnn, "interoceptive")
    print(f"  After interoceptive: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    # ====================================================================
    # PHASE 1: PRE-K
    # ====================================================================
    print("\n[PHASE 1] PRE-K")
    print("-" * 70)
    
    info = get_corpus_info("prek")
    print(f"  Corpus: {info['total']} sentences")
    
    t0 = time.time()
    corpus = get_corpus("prek")
    train(lnn, corpus, reps=50, learn_type="sensory", rem_interval="end", verbose=False)
    print(f"  Pre-K training: {time.time()-t0:.1f}s")
    print(f"  Pre-K: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    save_checkpoint(lnn, "prek")
    results["prek"] = {"nodes": len(lnn.nodes), "springs": len(lnn.springs)}
    
    # ====================================================================
    # PHASE 2: KINDERGARTEN
    # ====================================================================
    print("\n[PHASE 2] KINDERGARTEN")
    print("-" * 70)
    
    info = get_corpus_info("kindergarten")
    print(f"  Corpus: {info['total']} sentences")
    
    t0 = time.time()
    corpus = get_corpus("kindergarten")
    train(lnn, corpus, reps=40, learn_type="language", rem_interval="end", verbose=False)
    print(f"  K training: {time.time()-t0:.1f}s")
    print(f"  K: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    save_checkpoint(lnn, "k")
    results["k"] = {"nodes": len(lnn.nodes), "springs": len(lnn.springs)}
    
    # ====================================================================
    # PHASE 3: 1ST GRADE + GRAMMAR
    # ====================================================================
    print("\n[PHASE 3] 1ST GRADE + GRAMMAR")
    print("-" * 70)
    
    info = get_corpus_info("first_grade")
    print(f"  Corpus: {info['total']} sentences")
    
    t0 = time.time()
    corpus = get_corpus("first_grade")
    train(lnn, corpus, reps=30, learn_type="language", rem_interval="end", verbose=False)
    
    # Grammar lessons for 1st grade
    grammar_lessons = get_grammar_lessons(grade=1, full=False)
    grammar_sentences = []
    for lesson in grammar_lessons:
        grammar_sentences.extend(lesson.get("sentences", []))
    print(f"  Grammar: {len(grammar_sentences)} sentences")
    train_fast(lnn, grammar_sentences, reps=15, verbose=False)
    
    print(f"  1st grade training: {time.time()-t0:.1f}s")
    print(f"  1st: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    save_checkpoint(lnn, "1st")
    results["1st"] = {"nodes": len(lnn.nodes), "springs": len(lnn.springs)}
    
    # Test 1st grade grammar generation
    print("\n  1ST GRADE GRAMMAR GENERATION TEST:")
    test_prompts = ["the dog", "a cat", "i see", "she runs", "the sun"]
    for prompt in test_prompts:
        result = test_generation(lnn, prompt, num_tokens=5)
        print(f"    '{prompt}' -> '{result}'")
    
    # ====================================================================
    # PHASE 4: 2ND GRADE + GRAMMAR
    # ====================================================================
    print("\n[PHASE 4] 2ND GRADE + GRAMMAR")
    print("-" * 70)
    
    info = get_corpus_info("second_grade")
    print(f"  Corpus: {info['total']} sentences")
    
    t0 = time.time()
    corpus = get_corpus("second_grade")
    train(lnn, corpus, reps=25, learn_type="language", rem_interval="end", verbose=False)
    
    # Grammar lessons for 2nd grade
    grammar_lessons = get_2nd_lessons(grade=2, full=False)
    grammar_sentences = []
    for lesson in grammar_lessons:
        grammar_sentences.extend(lesson.get("sentences", []))
    print(f"  Grammar: {len(grammar_sentences)} sentences")
    train_fast(lnn, grammar_sentences, reps=15, verbose=False)
    
    print(f"  2nd grade training: {time.time()-t0:.1f}s")
    print(f"  2nd: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    save_checkpoint(lnn, "2nd")
    results["2nd"] = {"nodes": len(lnn.nodes), "springs": len(lnn.springs)}
    
    # Test 2nd grade grammar generation
    print("\n  2ND GRADE GRAMMAR GENERATION TEST:")
    test_prompts = ["the girl runs", "the bird builds", "he walked", "she eats", "they play"]
    for prompt in test_prompts:
        result = test_generation(lnn, prompt, num_tokens=5)
        print(f"    '{prompt}' -> '{result}'")
    
    # ====================================================================
    # PHASE 5: 3RD GRADE + GRAMMAR
    # ====================================================================
    print("\n[PHASE 5] 3RD GRADE + GRAMMAR")
    print("-" * 70)
    
    info = get_corpus_info("third_grade")
    print(f"  Corpus: {info['total']} sentences")
    
    t0 = time.time()
    corpus = get_corpus("third_grade")
    train(lnn, corpus, reps=20, learn_type="language", rem_interval="end", verbose=False)
    
    # Grammar lessons for 3rd grade
    grammar_lessons = get_3rd_lessons(grade=3, full=False)
    grammar_sentences = []
    for lesson in grammar_lessons:
        grammar_sentences.extend(lesson.get("sentences", []))
    print(f"  Grammar: {len(grammar_sentences)} sentences")
    train_fast(lnn, grammar_sentences, reps=15, verbose=False)
    
    print(f"  3rd grade training: {time.time()-t0:.1f}s")
    print(f"  3rd: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    save_checkpoint(lnn, "3rd")
    results["3rd"] = {"nodes": len(lnn.nodes), "springs": len(lnn.springs)}
    
    # Test 3rd grade grammar generation
    print("\n  3RD GRADE GRAMMAR GENERATION TEST:")
    test_prompts = ["he walks", "she is walking", "they were reading", "the dogs run", "it's barking"]
    for prompt in test_prompts:
        result = test_generation(lnn, prompt, num_tokens=5)
        print(f"    '{prompt}' -> '{result}'")
    
    # ====================================================================
    # FINAL SUMMARY
    # ====================================================================
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    
    print(f"\n  Time: {elapsed:.1f}s")
    print(f"\n  {'Level':15s} {'Nodes':>8s} {'Springs':>10s}")
    print(f"  {'-'*40}")
    for level, r in results.items():
        print(f"  {level:15s} {r['nodes']:8d} {r['springs']:10d}")
    
    print(f"\n  Final: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs, {len(lnn.trigrams)} trigrams")
    
    # Generation quality test
    print("\n  GENERATION QUALITY TEST:")
    quality_prompts = [
        "quantum mechanics",
        "photosynthesis",
        "democracy",
        "the dog barks",
        "she walks to",
    ]
    for prompt in quality_prompts:
        result = test_generation(lnn, prompt, num_tokens=6)
        print(f"    '{prompt}' -> '{result}'")
    
    # Save final report
    report_path = os.path.join(REPORT_DIR, f"pipeline_grammar_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, 'w') as f:
        json.dump({
            "elapsed": elapsed,
            "levels": results,
            "final": {
                "nodes": len(lnn.nodes),
                "springs": len(lnn.springs),
                "trigrams": len(lnn.trigrams),
            }
        }, f, indent=2)
    print(f"\n  Report: {report_path}")
    
    return lnn, results


if __name__ == "__main__":
    import json
    lnn, results = run_pipeline()