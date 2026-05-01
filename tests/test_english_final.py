"""
LRN English Comprehensive Training & Test - FIXED
Tests: Phonics, Grammar, Sentence Structure, Generation, Completion
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.natural_tokenize import learn_from_text
from lrn.inference import add_word_nodes, attention_with_residue


# ============================================================================
# TRAINING CORPUS - All in one for easier training
# ============================================================================

TRAINING_CORPUS = [
    # Phonics - as complete words
    "cat hat mat sat bat", "bed red leg set net", "bike hike like Mike side",
    "boat coat float goat", "moon spoon June noon", "swim swing sweep sweet",
    "black flat glad plan", "brush crash dream", "sheep ship shell",
    # Grammar parts
    "cat dog bird fish", "i you he she we they", "eat drink run sleep work",
    "big small fast slow hot cold", "in on under behind between near",
    # Sentences - simple
    "the cat eats fish", "the dog sees cat", "the bird flies", 
    "i love music", "she wants food", "he needs water",
    # Sentences - compound
    "the cat eats and dog sleeps", "i study and you play",
    "he runs but she walks",
    # Sentences - question
    "what is that", "where are you", "when does it start",
    "why are you here", "who is there",
    # Sentences - conditional
    "if it rains i stay home", "if you study you pass",
    "if he runs he wins",
]


def train_english_full():
    """Train comprehensive English."""
    print("Training English...")
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    # Train with high repetitions
    for _ in range(50):
        for text in TRAINING_CORPUS:
            # Learn at both sensory and language level
            learn_from_text(lnn, text, repetitions=1, learn_type="sensory")
            learn_from_text(lnn, text, repetitions=1, learn_type="language")
    
    # Add word nodes
    add_word_nodes(lnn, TRAINING_CORPUS)
    
    # Add identity
    lnn.add_node("identity:self")
    for w in ["i", "me", "my", "you", "your"]:
        lnn.add_spring(f"word:{w}", "identity:self", stiffness=30, tau=1)
    
    print(f"  Trained: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    return lnn


# ============================================================================
# TESTS
# ============================================================================

def test_phonics(lnn):
    """Test phoneme patterns - check word families."""
    print("\n=== PHONICS TEST ===")
    
    # Test word families via attention
    test_words = ["cat", "bike", "boat", "moon"]
    passed = 0
    
    for word in test_words:
        query = f"word:{word}"
        
        result = attention_with_residue(lnn, query, propagate_steps=2)
        
        # Find other words in same family
        # cat → sat, mat, bat, hat
        # bike → like, Mike
        # boat → coat, float, goat
        # moon → spoon, June, noon
        
        family_map = {
            "cat": ["sat", "mat", "bat", "hat"],
            "bike": ["like", "Mike"],
            "boat": ["coat", "float", "goat"],
            "moon": ["spoon", "June", "noon"],
        }
        
        family = family_map.get(word, [])
        words_found = [n.replace("word:", "") for n, _ in result["attention"]]
        
        found_family = [w for w in family if w in words_found]
        
        if found_family:
            passed += 1
            print(f"  {word}: PASS (found: {found_family})")
        else:
            print(f"  {word}: FAIL (got: {words_found[:3]})")
    
    print(f"Phonics: {passed}/{len(test_words)}")
    return passed, len(test_words)


def test_grammar(lnn):
    """Test grammar - parts of speech."""
    print("\n=== GRAMMAR TEST ===")
    
    test_pairs = [
        ("word:cat", "word:dog"),  # nouns
        ("word:eats", "word:runs"),  # verbs
        ("word:big", "word:small"),  # adjectives
        ("word:in", "word:on"),  # prepositions
    ]
    
    passed = 0
    for word_a, word_b in test_pairs:
        result = attention_with_residue(lnn, word_a, propagate_steps=2)
        
        # Check if related word appears
        found = word_b.replace("word:", "") in [n for n, _ in result["attention"]]
        
        if found:
            passed += 1
            print(f"  {word_a.replace('word:', '')}: PASS")
        else:
            print(f"  {word_a.replace('word:', '')}: FAIL")
    
    print(f"Grammar: {passed}/{len(test_pairs)}")
    return passed, len(test_pairs)


def test_sentence_structure(lnn):
    """Test sentence patterns."""
    print("\n=== SENTENCE STRUCTURE TEST ===")
    
    # SVO: cat eats fish → cat → eats → fish
    # Conditional: if it rains → if → it → rains
    
    test_start = [
        ("word:cat", ["eats", "fish", "sees"]),  # Subject
        ("word:if", ["it", "rains", "study"]),  # Conditional
    ]
    
    passed = 0
    for query, expected in test_start:
        result = attention_with_residue(lnn, query, propagate_steps=3)
        
        words = [n.replace("word:", "") for n, _ in result["attention"]]
        
        # Need at least 2 expected words found
        found = sum(1 for w in expected if w in words)
        
        if found >= 2:
            passed += 1
            print(f"  {query.replace('word:', '')}: PASS")
        else:
            print(f"  {query.replace('word:', '')}: FAIL")
    
    print(f"Structure: {passed}/{len(test_start)}")
    return passed, len(test_start)


def test_generation(lnn):
    """Test generation - extend from seed."""
    print("\n=== GENERATION TEST ===")
    
    seeds = ["the", "i", "you", "she", "he", "they", "we", "a"]
    passed = 0
    
    for seed in seeds:
        query = f"word:{seed}"
        result = attention_with_residue(lnn, query, propagate_steps=3)
        
        if result["attention"]:
            passed += 1
            words = [n.replace("word:", "") for n, _ in result["attention"][:3]]
            print(f"  '{seed}' → {words}")
    
    print(f"Generation: {passed}/{len(seeds)}")
    return passed, len(seeds)


def test_completion(lnn):
    """Test fill-in-the-blank."""
    print("\n=== COMPLETION TEST ===")
    
    tests = [
        ("word:eats", "word:fish"),  # cat eats ___
        ("word:study", "word:pass"),  # if you ___ you pass
        ("word:blue", "word:sky"),  # the ___ is blue
    ]
    
    passed = 0
    for query, expected in tests:
        result = attention_with_residue(lnn, query, propagate_steps=3)
        
        expected_word = expected.replace("word:", "")
        found = expected_word in [n.replace("word:", "") for n, _ in result["attention"]]
        
        if found:
            passed += 1
            print(f"  {query.replace('word:', '')}: PASS")
        else:
            print(f"  {query.replace('word:', '')}: FAIL")
    
    print(f"Completion: {passed}/{len(tests)}")
    return passed, len(tests)


def main():
    print("=" * 60)
    print("LRN ENGLISH COMPREHENSIVE TEST")
    print("=" * 60)
    
    # Train
    lnn = train_english_full()
    
    # Tests
    results = {}
    results["phonics"] = test_phonics(lnn)
    results["grammar"] = test_grammar(lnn)
    results["structure"] = test_sentence_structure(lnn)
    results["generation"] = test_generation(lnn)
    results["completion"] = test_completion(lnn)
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
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