"""
Full Pipeline: Sensory Grounding → 8th Grade with Grammar and Programming Spiral
Builds complete educational lattice from interoceptive origins through 8th grade.
Tests grammar, programming, and generation quality at each level.

Usage:
    python3 full_k_to_8th.py
"""
import sys
import os
import pickle
import time
import json

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
from lrn.computer_logic import get_programming_lessons, get_all_programming_sentences
from lrn.computer_logic_4th_8th import get_programming_lessons as get_prog_lessons_4th_8th
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


def test_generation(lnn, prompt, num_tokens=6):
    return generate_sequence(lnn, prompt, max_tokens=num_tokens)


def train_programming_lessons(lnn, grade):
    """Train on programming/computer logic lessons for a grade."""
    if grade <= 3:
        lessons = get_programming_lessons(grade)
    else:
        lessons = get_prog_lessons_4th_8th(grade)

    sentences = []
    for lesson in lessons:
        sentences.extend(lesson.get("sentences", []))
    if sentences:
        print(f"  Programming: {len(sentences)} sentences (grade {grade})")
        train_fast(lnn, sentences, reps=12, verbose=False)
    return len(sentences)


def run_pipeline():
    print("=" * 70)
    print("FULL PIPELINE: SENSORY → 8TH GRADE (Grammar + Programming)")
    print("=" * 70)

    start_time = time.time()
    results = {}

    # ====================================================================
    # PHASE 0: SENSORY GROUNDING
    # ====================================================================
    print("\n[PHASE 0] SENSORY GROUNDING")
    print("-" * 70)

    lnn = sensory_grounding(verbose=False)
    print(f"  After sensory: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")

    physical_manipulation(lnn, verbose=False)
    social_interaction(lnn, verbose=False)

    save_checkpoint(lnn, "sensory")

    # ====================================================================
    # PHASE 0.5: SENSORY/INTEROCEPTIVE LESSONS (K-3rd)
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
            print(f"  Grade {grade}: {len(sentences)} sentences")
            train_fast(lnn, sentences, reps=15, verbose=False)

    save_checkpoint(lnn, "interoceptive")

    # ====================================================================
    # PRE-K through 3RD GRADE
    # ====================================================================
    levels_config = [
        ("prek", "prek", 50),
        ("kindergarten", "k", 40),
        ("first_grade", "1st", 30),
        ("second_grade", "2nd", 25),
        ("third_grade", "3rd", 20),
    ]

    grammar_getters = {
        1: (get_grammar_lessons, False),
        2: (get_2nd_lessons, False),
        3: (get_3rd_lessons, False),
    }

    for corpus_name, prefix, reps in levels_config:
        print(f"\n[{prefix.upper()}] {corpus_name.upper()}")
        print("-" * 70)

        info = get_corpus_info(corpus_name)
        print(f"  Corpus: {info['total']} sentences")

        t0 = time.time()
        corpus = get_corpus(corpus_name)
        train(lnn, corpus, reps=reps, learn_type="language", rem_interval="end", verbose=False)

        grade_map = {"prek": 0, "k": 0, "1st": 1, "2nd": 2, "3rd": 3}
        grade = grade_map.get(prefix, int(prefix.replace("th", "")) if prefix.endswith("th") else 0)
        if grade in grammar_getters:
            getter, full = grammar_getters[grade]
            lessons = getter(grade, full)
            sentences = []
            for lesson in lessons:
                sentences.extend(lesson.get("sentences", []))
            print(f"  Grammar: {len(sentences)} sentences")
            train_fast(lnn, sentences, reps=15, verbose=False)

        print(f"  Training: {time.time()-t0:.1f}s")
        print(f"  {prefix}: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")

        save_checkpoint(lnn, prefix)
        results[prefix] = {"nodes": len(lnn.nodes), "springs": len(lnn.springs)}

        print(f"  Generation test:")
        test_prompts = ["the dog", "first i wake up then", "if it rains then", "algorithm step"]
        for prompt in test_prompts[:2]:
            result = test_generation(lnn, prompt, num_tokens=5)
            print(f"    '{prompt}' -> '{result}'")

    # ====================================================================
    # 4TH through 8TH GRADE (with grammar + programming)
    # ====================================================================
    corpus_map = {4: "fourth_grade", 5: "fifth_grade", 6: "sixth_grade", 7: "seventh_grade", 8: "eighth_grade"}
    for grade in [4, 5, 6, 7, 8]:
        corpus_name = corpus_map[grade]
        prefix = f"{grade}th"

        print(f"\n[{prefix.upper()}] {corpus_name.upper()}")
        print("-" * 70)

        info = get_corpus_info(corpus_name)
        print(f"  Corpus: {info['total']} sentences")

        t0 = time.time()
        corpus = get_corpus(corpus_name)
        train(lnn, corpus, reps=15, learn_type="language", rem_interval="end", verbose=False)

        # Grammar for 4th-8th
        if grade == 4:
            from lrn.grammar_lessons_4th import get_grammar_lessons as get_4th_grammar
            lessons = get_4th_grammar(grade, False)
            sentences = []
            for lesson in lessons:
                sentences.extend(lesson.get("sentences", []))
            if sentences:
                print(f"  Grammar: {len(sentences)} sentences")
                train_fast(lnn, sentences, reps=12, verbose=False)
        elif grade == 5:
            from lrn.grammar_lessons_5th import get_grammar_lessons as get_5th_grammar
            lessons = get_5th_grammar(grade, False)
            sentences = []
            for lesson in lessons:
                sentences.extend(lesson.get("sentences", []))
            if sentences:
                print(f"  Grammar: {len(sentences)} sentences")
                train_fast(lnn, sentences, reps=12, verbose=False)
        elif grade in [6, 7, 8]:
            from lrn.grammar_lessons_6th_8th import get_grammar_lessons as get_678_grammar
            lessons = get_678_grammar(grade, False)
            sentences = []
            for lesson in lessons:
                sentences.extend(lesson.get("sentences", []))
            if sentences:
                print(f"  Grammar: {len(sentences)} sentences")
                train_fast(lnn, sentences, reps=12, verbose=False)

        # Programming lessons
        train_programming_lessons(lnn, grade)

        print(f"  Training: {time.time()-t0:.1f}s")
        print(f"  {prefix}: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")

        save_checkpoint(lnn, prefix)
        results[prefix] = {"nodes": len(lnn.nodes), "springs": len(lnn.springs)}

        print(f"  Generation test:")
        test_prompts = [
            "algorithm step by step",
            "if condition then result",
            "function takes input returns",
            "loop through the list",
        ]
        for prompt in test_prompts[:2]:
            result = test_generation(lnn, prompt, num_tokens=5)
            print(f"    '{prompt}' -> '{result}'")

    # ====================================================================
    # FINAL SUMMARY
    # ====================================================================
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE: K → 8TH GRADE")
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
        "photosynthesis in plants",
        "democracy and government",
        "algorithm for sorting",
        "if then else logic",
    ]
    for prompt in quality_prompts:
        result = test_generation(lnn, prompt, num_tokens=6)
        print(f"    '{prompt}' -> '{result}'")

    # Save report
    report_path = os.path.join(REPORT_DIR, f"pipeline_k_to_8th_{time.strftime('%Y%m%d_%H%M%S')}.json")
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
    lnn, results = run_pipeline()