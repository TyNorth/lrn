"""
LRN English Comprehensive Training & Test
Tests: Phonics, Grammar, Sentence Structure, Generation, Completion
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.natural_tokenize import learn_from_text, TAU_BY_TYPE
from lrn.inference import add_word_nodes, attention_with_residue
from lrn.sleep_cycle import full_sleep_cycle
from lrn.rem_synthesis import REMSleep


# ============================================================================
# COMPREHENSIVE ENGLISH CORPUS
# ============================================================================

PHONICS_CORPUS = [
    # Vowel sounds
    "a e i o u", "cat hat mat sat bat", "bed red leg set net",
    "bit sit fit hit kit", "hot pot lot dot rot", "cut nut but gut",
    "cake make take lake fake", "bike hike like Mike side",
    "boat coat float goat note", "moon June spoon noon",
    # Consonant blends
    "bl cl fl gl pl sl br cr dr fr gr pr tr st sp sk",
    "black flat glad plan sleep brush crash dream frame", 
    "brown crown ground proud crown", "swim swing sweep sweet",
    "chair chin chunk much each", "sheep ship shell should",
    # Diphthongs
    "oi oy", "boy toy joy", "ow ou", "cow how now brown",
    # R-controlled
    "ar or er ir ur", "car far bar jar", "corn form storm short",
]


GRAMMAR_CORPUS = {
    "nouns": [
        "the cat", "the dog", "the bird", "the fish", "the horse",
        "my mother", "your father", "his sister", "her brother",
        "the house", "the car", "the tree", "the book", "the water",
    ],
    "pronouns": [
        "i am here", "you are there", "he is tall", "she is pretty",
        "it is cold", "we are happy", "they are busy",
        "me and you", "her and him", "us and them",
    ],
    "verbs": [
        "i eat", "you drink", "he runs", "she sleeps", "it works",
        "we play", "they talk", "i see", "you hear", "she feels",
        "i want", "you need", "he likes", "she loves", "it becomes",
    ],
    "adjectives": [
        "big and small", "fast and slow", "hot and cold", "old and new",
        "happy and sad", "good and bad", "tall and short", "fat and thin",
        "bright and dark", "clean and dirty", "loud and quiet",
    ],
    "prepositions": [
        "in the house", "on the table", "under the bed", "behind the door",
        "in front of", "next to", "between the", "near the", "far from",
        "at the", "to the", "from the", "with the", "without the",
    ],
    "articles": [
        "a cat", "a dog", "an apple", "an orange", "the book", "the car",
    ],
    "conjunctions": [
        "i am tired but i am happy", "she is smart and kind",
        "he runs fast or he walks slow", "because it is raining i stay home",
    ],
}


SENTENCE_STRUCTURE_CORPUS = {
    "simple_svo": [
        "the cat eats fish", "the dog sees a cat", "the bird flies high",
        "i love music", "she wants food", "he needs water",
    ],
    "compound": [
        "the cat eats and the dog sleeps", "i study and you play",
        "he runs but she walks", "we work and they rest",
    ],
    "complex": [
        "the cat that sleeps eats fish", "the dog that runs sees a cat",
        "i know that you are here", "she says that he is coming",
    ],
    "question_word": [
        "what is that", "where are you", "when does it start",
        "why are you here", "how does it work", "who is there",
    ],
    "imperative": [
        "sit down", "come here", "open the door", "close the window",
        "eat your food", "drink your water", "watch the show",
    ],
    "conditional": [
        "if it rains i stay home", "if you study you pass",
        "if he runs he wins", "if she works she succeeds",
    ],
}


GENERATION_SEEDS = [
    "the", "a", "i", "you", "she", "he", "they", "we",
]


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_phonics(lnn, reps=20):
    """Train phoneme patterns."""
    for _ in range(reps):
        for text in PHONICS_CORPUS:
            learn_from_text(lnn, text, repetitions=1, learn_type="sensory")


def train_grammar(lnn, reps=20):
    """Train grammar patterns."""
    for category, sentences in GRAMMAR_CORPUS.items():
        for _ in range(reps):
            for sentence in sentences:
                learn_from_text(lnn, sentence, repetitions=1, learn_type="language")


def train_sentence_structure(lnn, reps=20):
    """Train sentence structures."""
    for category, sentences in SENTENCE_STRUCTURE_CORPUS.items():
        for _ in range(reps):
            for sentence in sentences:
                learn_from_text(lnn, sentence, repetitions=1, learn_type="language")


def train_full_english(reps=20):
    """Train complete English."""
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    train_phonics(lnn, reps)
    train_grammar(lnn, reps)
    train_sentence_structure(lnn, reps)
    
    # Add word nodes
    all_sentences = PHONICS_CORPUS.copy()
    for sentences in GRAMMAR_CORPUS.values():
        all_sentences.extend(sentences)
    for sentences in SENTENCE_STRUCTURE_CORPUS.values():
        all_sentences.extend(sentences)
    
    add_word_nodes(lnn, all_sentences)
    
    # Add identity
    lnn.add_node("identity:self")
    for w in ["i", "me", "my", "you", "your"]:
        lnn.add_spring(f"word:{w}", "identity:self", stiffness=30, tau=1)
    
    return lnn


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_phonics(lnn):
    """Test phoneme recognition."""
    print("\n=== PHONICS TEST ===")
    
    test_phonemes = ["cat", "sat", "mat", "bed", "red", "bike", "boat", "moon"]
    results = []
    
    for phoneme in test_phonemes:
        query = f"word:{phoneme}"
        
        # Activate query
        for node in lnn.nodes.values():
            node.activation = 0
            node.pinned = False
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            lnn.nodes[query].pinned = True
        
        propagate(lnn, n_steps=3)
        
        # Find related phonemes
        related = []
        for name, node in lnn.nodes.items():
            if name.startswith("word:") and node.activation > 10 and name != query:
                word = name.replace("word:", "")
                # Check if similar pattern
                if len(word) >= 3:
                    related.append((word, node.activation))
        
        related.sort(key=lambda x: -x[1])
        
        # Check for rhyme/family
        found_family = False
        for r, _ in related[:5]:
            # Same ending pattern
            if phoneme.endswith(r[-2:]) or r.endswith(phoneme[-2:]):
                found_family = True
                break
        
        results.append((phoneme, found_family))
        status = "PASS" if found_family else "FAIL"
        print(f"  {phoneme}: {status}")
    
    passed = sum(1 for _, r in results if r)
    print(f"Phonics: {passed}/{len(test_phonemes)}")
    return passed, len(test_phonemes)


def test_grammar(lnn):
    """Test grammar recognition."""
    print("\n=== GRAMMAR TEST ===")
    
    # Test parts of speech recognition via attention
    test_cases = [
        ("word:cat", "noun", ["cat", "dog", "bird", "fish"]),
        ("word:eats", "verb", ["eat", "drink", "run", "sleep"]),
        ("word:big", "adjective", ["big", "small", "fast", "slow"]),
        ("word:in", "preposition", ["in", "on", "under", "behind"]),
    ]
    
    results = []
    for query, pos_type, expected in test_cases:
        result = attention_with_residue(lnn, query, propagate_steps=2)
        
        # Check if expected words appear in attention
        found = []
        for node, _ in result["attention"]:
            word = node.replace("word:", "")
            if word in expected:
                found.append(word)
        
        passed = len(found) >= 1
        results.append((pos_type, passed))
        status = "PASS" if passed else "FAIL"
        print(f"  {pos_type}: {status} (found: {found})")
    
    passed = sum(1 for _, r in results if r)
    print(f"Grammar: {passed}/{len(test_cases)}")
    return passed, len(test_cases)


def test_sentence_structure(lnn):
    """Test sentence structure recognition."""
    print("\n=== SENTENCE STRUCTURE TEST ===")
    
    test_structures = [
        ("word:cat", ["cat", "eats", "fish"]),  # SVO
        ("word:because", ["because", "raining", "stay"]),  # Complex
        ("word:if", ["if", "study", "pass"]),  # Conditional
    ]
    
    results = []
    for query, expected in test_structures:
        result = attention_with_residue(lnn, query, propagate_steps=3)
        
        # Check for structural patterns
        found = []
        for node, _ in result["attention"]:
            word = node.replace("word:", "")
            if word in expected:
                found.append(word)
        
        # SVO needs subject-verb-object chain
        passed = len(found) >= 2
        results.append((query.replace("word:", ""), passed))
        status = "PASS" if passed else "FAIL"
        print(f"  {query.replace('word:', '')}: {status}")
    
    passed = sum(1 for _, r in results if r)
    print(f"Structure: {passed}/{len(test_structures)}")
    return passed, len(test_structures)


def test_generation(lnn):
    """Test sentence generation."""
    print("\n=== GENERATION TEST ===")
    
    generated = []
    
    for seed in GENERATION_SEEDS:
        query = f"word:{seed}"
        
        result = attention_with_residue(lnn, query, propagate_steps=4)
        
        # Build possible continuation
        continuation = []
        for node, info in result["attention"][:5]:
            word = node.replace("word:", "")
            if word != seed:
                continuation.append(word)
        
        if continuation:
            generated.append((seed, " ".join(continuation[:3])))
    
    print("Generated sentences:")
    for seed, cont in generated:
        print(f"  '{seed}' → '{cont}'")
    
    passed = len(generated)
    print(f"Generation: {passed}/{len(GENERATION_SEEDS)}")
    return passed, len(GENERATION_SEEDS)


def test_completion(lnn):
    """Test sentence completion (fill in the blank)."""
    print("\n=== COMPLETION TEST ===")
    
    completion_tests = [
        ("the cat ___ fish", "word:eats", "eats"),
        ("i want to ___ food", "word:eat", "eat"),
        ("the sky is ___", "word:blue", "blue"),
        ("because it ___ i stay home", "word:rains", "rains"),
    ]
    
    results = []
    for sentence, correct_node, expected_word in completion_tests:
        # Find first word
        first_word = sentence.split()[1].replace("___", "").strip()
        query = f"word:{first_word}"
        
        result = attention_with_residue(lnn, query, propagate_steps=3)
        
        # Check if expected word appears in attention
        found = False
        for node, _ in result["attention"]:
            word = node.replace("word:", "")
            if word == expected_word:
                found = True
                break
        
        results.append((sentence, found))
        status = "PASS" if found else "FAIL"
        print(f"  '{sentence}': {status}")
    
    passed = sum(1 for _, r in results if r)
    print(f"Completion: {passed}/{len(completion_tests)}")
    return passed, len(completion_tests)


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def run_english_test_battery():
    """Run complete English test battery."""
    print("=" * 60)
    print("LRN ENGLISH COMPREHENSIVE TEST BATTERY")
    print("=" * 60)
    
    # Train
    print("\n--- Training ---")
    lnn = train_full_english(reps=20)
    print(f"Trained: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    # Run tests
    results = {}
    
    results["phonics"] = test_phonics(lnn)
    results["grammar"] = test_grammar(lnn)
    results["structure"] = test_sentence_structure(lnn)
    results["generation"] = test_generation(lnn)
    results["completion"] = test_completion(lnn)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    total_passed = 0
    total_tests = 0
    
    for test_name, (passed, total) in results.items():
        pct = (passed/total*100) if total > 0 else 0
        print(f"{test_name.upper()}: {passed}/{total} ({pct:.0f}%)")
        total_passed += passed
        total_tests += total
    
    print(f"\nOVERALL: {total_passed}/{total_tests} ({(total_passed/total_tests*100):.0f}%)")
    
    return results


if __name__ == "__main__":
    run_english_test_battery()