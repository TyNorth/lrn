"""
LRN English Sequential Test Battery
====================================

Tests English learning through developmental stages:
1. Sensory/Letters → 2. Babbling → 3. Phonics → 4. Morphology →
5. Sight Words → 6. Vocabulary → 7. Grammar → 8. Syntax →
9. Sentences → 10. Pragmatics

Each stage trains until its developmental gate passes.
"""
import sys
import time

sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.english_training import SequentialTrainer
from lrn.english_report import generate_markdown_report
from lrn.inference import attention_with_residue
from lrn.grammar_training import infer_pos


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_phonics(lnn):
    """Test phoneme pattern recognition."""
    test_cases = [
        ("cat", ["sat", "mat", "bat", "hat"]),
        ("bike", ["like", "Mike"]),
        ("boat", ["coat", "float", "goat"]),
        ("moon", ["spoon", "June", "noon"]),
    ]
    
    results = []
    for word, family in test_cases:
        query = f"word:{word}"
        
        for n in lnn.nodes.values():
            n.activation = 0
            n.pinned = False
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
        
        propagate(lnn, n_steps=3)
        
        related = []
        for n, node in lnn.nodes.items():
            if n.startswith("word:") and node.activation > 10:
                related.append(n.replace("word:", ""))
        
        found = [w for w in family if w in related]
        passed = len(found) > 0
        results.append({
            "word": word,
            "expected": family,
            "found": found,
            "passed": passed,
        })
    
    return results


def test_grammar(lnn):
    """Test POS tagging via unsupervised context learning."""
    test_words = {
        "cat": "noun",
        "eats": "verb",
        "big": "adjective",
        "the": "determiner",
        "fast": "adjective",
        "runs": "verb",
        "dog": "noun",
        "small": "adjective",
    }
    
    results = []
    for word, expected_pos in test_words.items():
        result = infer_pos(lnn, word)
        actual_pos = result.get("pos", "unknown")
        passed = actual_pos == expected_pos
        
        results.append({
            "word": word,
            "expected": expected_pos,
            "actual": actual_pos,
            "passed": passed,
        })
    
    return results


def test_generation(lnn):
    """Test sentence generation from seed words."""
    seeds = ["the", "i", "you", "she", "he", "they", "we", "a"]
    
    results = []
    for seed in seeds:
        query = f"word:{seed}"
        result = attention_with_residue(lnn, query, propagate_steps=3)
        
        words = [n.replace("word:", "") for n, _ in result["attention"][:3]]
        passed = len(words) > 0
        
        results.append({
            "seed": seed,
            "generated": words,
            "passed": passed,
        })
    
    return results


def test_completion(lnn):
    """Test fill-in-the-blank completion."""
    test_cases = [
        {"sentence": "the cat ___ fish", "query": "word:cat", "expected": "eats"},
        {"sentence": "the big ___ runs", "query": "word:big", "expected": "dog"},
        {"sentence": "the dog ___ fast", "query": "word:dog", "expected": "runs"},
    ]
    
    results = []
    for test in test_cases:
        result = attention_with_residue(lnn, test["query"], propagate_steps=3)
        words = [n.replace("word:", "") for n, _ in result["attention"]]
        
        passed = test["expected"] in words
        
        results.append({
            "sentence": test["sentence"],
            "expected": test["expected"],
            "found": test["expected"] in words,
            "passed": passed,
        })
    
    return results


def test_attention(lnn):
    """Test attention mechanism (residue paths)."""
    test_queries = [
        {"query": "word:fire", "expected_related": ["burns", "hot"]},
        {"query": "word:water", "expected_related": ["cools", "fire"]},
        {"query": "word:cold", "expected_related": ["hot", "ice"]},
    ]
    
    results = []
    for test in test_queries:
        result = attention_with_residue(lnn, test["query"], propagate_steps=3)
        
        has_paths = len(result["attention"]) > 0
        words = [n.replace("word:", "") for n, _ in result["attention"]]
        found = [w for w in test["expected_related"] if w in words]
        
        passed = has_paths and len(found) > 0
        
        results.append({
            "query": test["query"].replace("word:", ""),
            "has_paths": has_paths,
            "expected": test["expected_related"],
            "found": found,
            "passed": passed,
        })
    
    return results


def test_causation(lnn):
    """Test causal chain inference."""
    test_chains = [
        {"name": "Fire → Burns → Wood", "start": "word:fire", "expected_chain": ["burns", "wood"]},
        {"name": "Water → Cools → Fire", "start": "word:water", "expected_chain": ["cools", "fire"]},
        {"name": "Sun → Heats → Ground", "start": "word:sun", "expected_chain": ["heats", "ground"]},
    ]
    
    results = []
    for test in test_chains:
        result = attention_with_residue(lnn, test["start"], propagate_steps=4)
        words = [n.replace("word:", "") for n, _ in result["attention"]]
        
        found = [w for w in test["expected_chain"] if w in words]
        passed = len(found) >= 1
        
        results.append({
            "name": test["name"],
            "start": test["start"].replace("word:", ""),
            "expected": test["expected_chain"],
            "found": found,
            "passed": passed,
        })
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("LRN ENGLISH SEQUENTIAL TEST BATTERY")
    print("=" * 80)
    
    start_time = time.time()
    
    # Train sequentially
    print("\nTraining through developmental stages...")
    trainer = SequentialTrainer(max_reps_per_stage=100, max_stall_rounds=3)
    training_results = trainer.train_all_stages()
    
    lnn = trainer.lnn
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f}s")
    print(f"  Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
    
    # Run tests
    print("\nRunning tests...")
    
    test_results = {}
    test_results["phonics"] = test_phonics(lnn)
    test_results["grammar"] = test_grammar(lnn)
    test_results["generation"] = test_generation(lnn)
    test_results["completion"] = test_completion(lnn)
    test_results["attention"] = test_attention(lnn)
    test_results["causation"] = test_causation(lnn)
    
    total_time = time.time() - start_time
    
    # Generate report
    print("\nGenerating report...\n")
    report = generate_markdown_report(training_results, test_results, total_time)
    
    print(report)
    
    # Save report
    with open("/Users/tyarc/github/lrn/docs/english_sequential_report.md", "w") as f:
        f.write(report)
    
    print(f"\nReport saved to: /Users/tyarc/github/lrn/docs/english_sequential_report.md")
    
    return test_results


if __name__ == "__main__":
    main()