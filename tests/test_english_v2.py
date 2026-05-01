"""
LRN English Comprehensive Training & Test - IMPROVED
Tests: Phonics, Grammar, Sentence Structure, Generation, Completion
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.natural_tokenize import learn_from_text, TAU_BY_TYPE
from lrn.inference import add_word_nodes, attention_with_residue


# ============================================================================
# COMPREHENSIVE ENGLISH CORPUS
# ============================================================================

PHONICS_CORPUS = [
    "a e i o u", "cat hat mat sat bat", "bed red leg set net",
    "bit sit fit hit kit", "hot pot lot dot rot", "cut nut but gut",
    "cake make take lake fake", "bike hike like Mike side",
    "boat coat float goat note", "moon June spoon noon",
    "bl cl fl gl pl sl", "br cr dr fr gr pr tr", "st sp sk",
    "black flat glad plan", "brush crash dream frame",
    "swim swing sweep sweet", "sheep ship shell",
    "boy toy joy", "cow how now", "car far bar jar",
]


GRAMMAR_CORPUS = {
    "nouns": [
        "the cat dog bird fish horse cow sheep chicken", 
        "my mother father sister brother",
        "the house car tree book water",
        "a cat a dog a bird a fish",
    ],
    "pronouns": [
        "i am you are he is she is it is",
        "we are they are me and you",
        "his her its our their",
    ],
    "verbs": [
        "i eat you drink he runs she sleeps it works",
        "i see you hear she feels we want they need",
        "i like you love he wants she needs",
    ],
    "adjectives": [
        "big small fast slow hot cold old new",
        "happy sad good bad tall short fat thin",
        "bright dark clean dirty loud quiet",
    ],
    "prepositions": [
        "in on under behind between near far",
        "at to from with without",
    ],
    "conjunctions": [
        "i am tired but i am happy",
        "she is smart and kind",
        "he runs or she walks",
    ],
}


SENTENCE_STRUCTURE_CORPUS = {
    "simple": [
        "the cat eats fish", "the dog sees cat", "the bird flies high",
        "i love music", "she wants food", "he needs water",
    ],
    "compound": [
        "the cat eats and dog sleeps", "i study and you play",
        "he runs but she walks", "we work they rest",
    ],
    "complex": [
        "the cat that sleeps eats", "the dog that runs sees",
        "i know that you are here", "she says that he comes",
    ],
    "question": [
        "what is that", "where are you", "when does it start",
        "why are here", "how does work", "who is there",
    ],
    "imperative": [
        "sit down come here open door close window",
        "eat food drink water watch show",
    ],
    "conditional": [
        "if it rains i stay home", "if you study you pass",
        "if he runs he wins", "if she works she succeeds",
    ],
}


def train_full_english(reps=40):
    """Train complete English with more repetitions."""
    print("Training English...")
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    # More repetitions for stronger patterns
    for _ in range(reps):
        for text in PHONICS_CORPUS:
            learn_from_text(lnn, text, repetitions=1, learn_type="sensory")
    
    for _ in range(reps):
        for category, sentences in GRAMMAR_CORPUS.items():
            for sentence in sentences:
                learn_from_text(lnn, sentence, repetitions=1, learn_type="language")
    
    for _ in range(reps):
        for category, sentences in SENTENCE_STRUCTURE_CORPUS.items():
            for sentence in sentences:
                learn_from_text(lnn, sentence, repetitions=1, learn_type="language")
    
    # Build sentence list for word nodes
    all_sentences = []
    for category, sentences in GRAMMAR_CORPUS.items():
        all_sentences.extend(sentences)
    for category, sentences in SENTENCE_STRUCTURE_CORPUS.items():
        all_sentences.extend(sentences)
    
    # Filter to strings only
    all_sentences = [s for s in all_sentences if isinstance(s, str)]
    
    add_word_nodes(lnn, all_sentences)
    
    # Add identity
    lnn.add_node("identity:self")
    for w in ["i", "me", "my", "you", "your"]:
        lnn.add_spring(f"word:{w}", "identity:self", stiffness=30, tau=1)
    
    print(f"  Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
    return lnn


# ============================================================================
# TESTS
# ============================================================================

def test_phonics(lnn):
    """Test phoneme recognition."""
    print("\n=== PHONICS TEST ===")
    
    test_cases = [
        ("cat", ["sat", "mat", "bat"]),  # -at family
        ("bike", ["like", "Mike"]),  # -ike family
        ("boat", ["coat", "float"]),  # -oat family
        ("moon", ["spoon", "noon"]),  # -oon family
    ]
    
    passed = 0
    for word, family in test_cases:
        query = f"word:{word}"
        
        # Simple propagation
        for n in lnn.nodes.values():
            n.activation = 0
            n.pinned = False
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
        
        propagate(lnn, n_steps=3)
        
        # Check related
        related = []
        for n, node in lnn.nodes.items():
            if n.startswith("word:") and node.activation > 10:
                related.append(n.replace("word:", ""))
        
        # Check if any family member in related
        found = [w for w in family if w in related]
        if found:
            passed += 1
            print(f"  {word}: PASS (found: {found})")
        else:
            print(f"  {word}: FAIL")
    
    print(f"Phonics: {passed}/{len(test_cases)}")
    return passed, len(test_cases)


def test_grammar(lnn):
    """Test grammar - find words that co-occur."""
    print("\n=== GRAMMAR TEST ===")
    
    # Grammar = which words appear in similar contexts
    test_pairs = [
        ("word:cat", "word:dog"),  # nouns together
        ("word:eats", "word:runs"),  # verbs
        ("word:big", "word:small"),  # adjectives
        ("word:in", "word:on"),  # prepositions
    ]
    
    passed = 0
    for word_a, word_b in test_pairs:
        result = attention_with_residue(lnn, word_a, propagate_steps=2)
        
        # Check if word_b in attention
        found = any(word_b.replace("word:", "") in n for n, _ in result["attention"])
        
        if found:
            passed += 1
            print(f"  {word_a.replace('word:', '')}/{word_b.replace('word:', '')}: PASS")
        else:
            print(f"  {word_a.replace('word:', '')}/{word_b.replace('word:', '')}: FAIL")
    
    print(f"Grammar: {passed}/{len(test_pairs)}")
    return passed, len(test_pairs)


def test_sentence_structure(lnn):
    """Test sentence patterns."""
    print("\n=== SENTENCE STRUCTURE TEST ===")
    
    patterns = [
        ("word:cat", ["cat", "eats", "fish"]),  # Subject-Verb-Object
        ("word:if", ["if", "study", "pass"]),  # Conditional
    ]
    
    passed = 0
    for query, expected in patterns:
        result = attention_with_residue(lnn, query, propagate_steps=3)
        
        words = [n.replace("word:", "") for n, _ in result["attention"]]
        
        found = sum(1 for w in expected if w in words)
        
        if found >= 2:
            passed += 1
            print(f"  {query.replace('word:', '')}: PASS")
        else:
            print(f"  {query.replace('word:', '')}: FAIL")
    
    print(f"Structure: {passed}/{len(patterns)}")
    return passed, len(patterns)


def test_generation(lnn):
    """Test generation - can we extend sentences?"""
    print("\n=== GENERATION TEST ===")
    
    seeds = ["the", "i", "you", "she", "he", "they", "we", "a"]
    passed = 0
    
    for seed in seeds:
        query = f"word:{seed}"
        result = attention_with_residue(lnn, query, propagate_steps=4)
        
        if result["attention"]:
            passed += 1
            # Show first 2 words
            words = [n.replace("word:", "") for n, _ in result["attention"][:2]]
            print(f"  '{seed}' → {words}")
    
    print(f"Generation: {passed}/{len(seeds)}")
    return passed, len(seeds)


def test_completion(lnn):
    """Test fill-in-the-blank."""
    print("\n=== COMPLETION TEST ===")
    
    tests = [
        ("cat", "eats"),  # The cat ___ fish
        ("study", "pass"),  # If you ___ you pass
        ("blue", "sky"),  # The ___ is blue
    ]
    
    passed = 0
    for query_word, expected in tests:
        query = f"word:{query_word}"
        result = attention_with_residue(lnn, query, propagate_steps=3)
        
        words = [n.replace("word:", "") for n, _ in result["attention"]]
        
        if expected in words:
            passed += 1
            print(f"  '{query_word}' → '{expected}': PASS")
        else:
            print(f"  '{query_word}' → '{expected}': FAIL")
    
    print(f"Completion: {passed}/{len(tests)}")
    return passed, len(tests)


def main():
    print("=" * 60)
    print("LRN ENGLISH COMPREHENSIVE TEST")
    print("=" * 60)
    
    # Train with more reps
    lnn = train_full_english(reps=40)
    
    # Run tests
    results = {}
    results["phonics"] = test_phonics(lnn)
    results["grammar"] = test_grammar(lnn)
    results["structure"] = test_sentence_structure(lnn)
    results["generation"] = test_generation(lnn)
    results["completion"] = test_completion(lnn)
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    total_passed = 0
    total_tests = 0
    
    for name, (passed, total) in results.items():
        pct = (passed/total*100) if total > 0 else 0
        print(f"{name.upper()}: {passed}/{total} ({pct:.0f}%)")
        total_passed += passed
        total_tests += total
    
    print(f"\nTOTAL: {total_passed}/{total_tests} ({(total_passed/total_tests*100):.0f}%)")
    
    return results


if __name__ == "__main__":
    main()