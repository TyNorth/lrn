"""
LRN English Full Training - 10 Minute Session
Sequential training with REM synthesis after each stage.
"""
import sys
import time

sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.natural_tokenize import learn_from_text
from lrn.inference import add_word_nodes
from lrn.english_corpus import ALL_STAGES
from lrn.english_gates import DevelopmentalGate


# Tau assignment per stage
STAGE_TAU = {
    "sensory": 0,
    "babbling": 1,
    "phonics": 1,
    "morphology": 1,
    "sight_words": 2,
    "vocabulary": 2,
    "grammar": 2,
    "syntax": 3,
    "sentences": 3,
    "pragmatics": 4,
}


def flatten_corpus(corpus):
    sentences = []
    if isinstance(corpus, list):
        sentences = corpus
    elif isinstance(corpus, dict):
        for texts in corpus.values():
            if isinstance(texts, list):
                sentences.extend(texts)
            elif isinstance(texts, str):
                sentences.append(texts)
    return sentences


def run_rem_synthesis(lnn, wake_buffer):
    """Lightweight REM - forms τ=3 categorical bridges between co-occurring words."""
    recent_words = set()
    for s in wake_buffer:
        for w in s.lower().split():
            recent_words.add(f"word:{w}")
    
    word_list = list(recent_words)
    for i in range(len(word_list)):
        for j in range(i+1, len(word_list)):
            a, b = word_list[i], word_list[j]
            key = lnn._key(a, b)
            if key in lnn.springs:
                sp = lnn.springs[key]
                if sp.tau > 2:
                    sp.tau = 3
                    sp.stiffness = max(sp.stiffness, 10)
            else:
                lnn.add_or_update_spring(a, b, stiffness=10, tau=3, mode="add")


def train_stage(lnn, stage, reps=50):
    """Train a single stage with REM synthesis."""
    corpus = ALL_STAGES[stage]
    sentences = flatten_corpus(corpus)
    
    learn_type = "sensory" if stage in ("sensory", "babbling", "phonics", "morphology") else "language"
    
    wake_buffer = []
    
    for rep in range(reps):
        for sentence in sentences:
            learn_from_text(lnn, sentence, repetitions=1, learn_type=learn_type)
            add_word_nodes(lnn, [sentence])
            wake_buffer.append(sentence)
            if len(wake_buffer) > 10:
                wake_buffer = wake_buffer[-10:]
        
        # REM synthesis after each corpus pass
        run_rem_synthesis(lnn, wake_buffer)
        propagate(lnn, n_steps=3)
    
    # Final REM
    run_rem_synthesis(lnn, wake_buffer)
    propagate(lnn, n_steps=3)
    
    return len(lnn.nodes), len(lnn.springs)


def test_phonics(lnn):
    """Test phonics word family recognition."""
    test_cases = [
        ("cat", ["hat", "mat", "sat", "bat"]),
        ("bike", ["hike", "like"]),
        ("boat", ["coat", "float"]),
        ("moon", ["spoon", "noon"]),
    ]
    
    passed = 0
    for word, family in test_cases:
        query = f"word:{word}"
        
        for n in lnn.nodes.values():
            n.activation = 0
            n.pinned = False
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            propagate(lnn, n_steps=5)
            
            activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            found = [w for w in family if w in activated]
            if found:
                passed += 1
                print(f"  {word}: PASS (found: {found})")
            else:
                print(f"  {word}: FAIL")
    
    return passed, len(test_cases)


def test_grammar(lnn):
    """Test POS tagging."""
    from lrn.grammar_training import infer_pos
    
    test_words = {
        "cat": "noun", "eats": "verb", "big": "adjective",
        "the": "determiner", "fast": "adjective", "runs": "verb",
        "dog": "noun", "small": "adjective",
    }
    
    passed = 0
    for word, expected_pos in test_words.items():
        result = infer_pos(lnn, word)
        actual_pos = result.get("pos", "unknown")
        if actual_pos == expected_pos:
            passed += 1
            print(f"  {word}: PASS ({actual_pos})")
        else:
            print(f"  {word}: FAIL (expected {expected_pos}, got {actual_pos})")
    
    return passed, len(test_words)


def test_generation(lnn):
    """Test generation."""
    from lrn.inference import attention_with_residue
    
    seeds = ["the", "i", "you", "she", "he", "they", "we", "a"]
    passed = 0
    
    for seed in seeds:
        query = f"word:{seed}"
        result = attention_with_residue(lnn, query, propagate_steps=3)
        
        if result["attention"]:
            passed += 1
            words = [n.replace("word:", "") for n, _ in result["attention"][:3]]
            print(f"  '{seed}' → {words}")
    
    return passed, len(seeds)


def test_completion(lnn):
    """Test completion."""
    from lrn.inference import attention_with_residue
    
    tests = [
        ("word:cat", "eats"),
        ("word:big", "dog"),
        ("word:dog", "runs"),
    ]
    
    passed = 0
    for query, expected in tests:
        result = attention_with_residue(lnn, query, propagate_steps=3)
        words = [n.replace("word:", "") for n, _ in result["attention"]]
        
        if expected in words:
            passed += 1
            print(f"  '{query.replace('word:', '')}': PASS")
        else:
            print(f"  '{query.replace('word:', '')}': FAIL")
    
    return passed, len(tests)


def test_attention(lnn):
    """Test attention."""
    from lrn.inference import attention_with_residue
    
    test_queries = [
        ("word:fire", ["burns", "hot"]),
        ("word:water", ["cools", "fire"]),
        ("word:cold", ["hot", "ice"]),
    ]
    
    passed = 0
    for query, expected in test_queries:
        result = attention_with_residue(lnn, query, propagate_steps=3)
        words = [n.replace("word:", "") for n, _ in result["attention"]]
        found = [w for w in expected if w in words]
        
        if found:
            passed += 1
            print(f"  '{query.replace('word:', '')}': PASS (found: {found})")
        else:
            print(f"  '{query.replace('word:', '')}': FAIL")
    
    return passed, len(test_queries)


def main():
    print("=" * 60)
    print("LRN ENGLISH FULL TRAINING - 10 MINUTE SESSION")
    print("=" * 60)
    
    start_time = time.time()
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    # Train all stages
    stages = list(ALL_STAGES.keys())
    stage_results = {}
    
    for stage in stages:
        elapsed = time.time() - start_time
        print(f"\n[{elapsed:.0f}s] STAGE: {stage.upper()}")
        
        nodes, springs = train_stage(lnn, stage, reps=50)
        print(f"  Nodes: {nodes}, Springs: {springs}")
        
        stage_results[stage] = {"nodes": nodes, "springs": springs}
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f}s")
    print(f"Final: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    # Run tests
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    test_results = {}
    test_results["phonics"] = test_phonics(lnn)
    test_results["grammar"] = test_grammar(lnn)
    test_results["generation"] = test_generation(lnn)
    test_results["completion"] = test_completion(lnn)
    test_results["attention"] = test_attention(lnn)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_passed = 0
    total_tests = 0
    
    for category, (passed, total) in test_results.items():
        pct = (passed * 100) // max(1, total)
        status = "PASS" if pct >= 75 else "FAIL"
        print(f"{category.upper()}: {passed}/{total} ({pct}%) - {status}")
        total_passed += passed
        total_tests += total
    
    overall_pct = (total_passed * 100) // max(1, total_tests)
    print(f"\nTOTAL: {total_passed}/{total_tests} ({overall_pct}%)")
    
    return test_results


if __name__ == "__main__":
    main()
