"""
LRN Language Learning Test - Phonetics, Grammar, Sentences
Tests language learning in layers: phonemes -> grammar -> sentences
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor
from lrn.natural_tokenize import learn_from_text, discover_words


# ============================================================================
# PHONETICS - Sound patterns (letters -> phonemes)
# ============================================================================

PHONETIC_SAMPLES = [
    # Vowels
    "a e i o u",
    "cat hat bat",
    "big pig dig",
    "hot pot lot",
    "run fun sun",
    # Consonants
    "b c d f g h j k l m n p q r s t v w x y z",
    # Diphthongs
    "ai ea ie ou ow",
    "oi ue ui au",
    # Common words by sound
    "the be see get let met net pet set wet yet",
    "bat cat hat mat pat rat sat",
    "bit fit hit kit lit nit pit sit",
    "boy joy toy",
    "cow how now pow row",
]


def test_phonetics(repetitions: int = 50) -> dict:
    """Test phoneme learning."""
    print("\n=== PHONETICS TEST ===")
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    for _ in range(repetitions):
        for text in PHONETIC_SAMPLES:
            learn_from_text(lnn, text, repetitions=1)
    
    words = discover_words(lnn, PHONETIC_SAMPLES, min_frequency=2)
    
    print(f"  Phonetic patterns learned: {len(words)}")
    print(f"  Samples: {list(words.keys())[:20]}")
    
    return {"patterns": len(words), "vocabulary": list(words.keys())[:20]}


# ============================================================================
# GRAMMAR OPS - Word patterns, tenses, structures
# ============================================================================

GRAMMAR_SAMPLES = {
    "present_simple": [
        "i eat",
        "you eat",
        "he eats",
        "she eats",
        "it eats",
        "we eat",
        "they eat",
        "i drink",
        "you drink",
        "he drinks",
        "i run",
        "you run",
        "she runs",
    ],
    "past_simple": [
        "i ate",
        "you ate",
        "he ate",
        "she ate",
        "it ate",
        "we ate",
        "they ate",
        "i drank",
        "you drank",
        "he drank",
    ],
    "future_simple": [
        "i will eat",
        "you will eat",
        "he will eat",
        "she will eat",
        "we will eat",
        "they will eat",
    ],
    "present_continuous": [
        "i am eating",
        "you are eating",
        "he is eating",
        "she is eating",
        "we are eating",
        "they are eating",
    ],
    "questions": [
        "do you eat",
        "does he eat",
        "did you eat",
        "will you eat",
        "can you eat",
        "are you eating",
    ],
    "negation": [
        "i do not eat",
        "you do not eat",
        "he does not eat",
        "she does not eat",
        "i did not eat",
        "i will not eat",
    ],
    "articles": [
        "the cat",
        "the dog",
        "a cat",
        "a dog",
        "an apple",
        "an orange",
    ],
    "prepositions": [
        "in the house",
        "on the table",
        "under the bed",
        "behind the door",
        "between the trees",
        "near the school",
        "at the park",
        "to the store",
    ],
}


def test_grammar_ops(repetitions: int = 50) -> dict:
    """Test grammar operation learning."""
    print("\n=== GRAMMAR OPS TEST ===")
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    all_samples = []
    for category, samples in GRAMMAR_SAMPLES.items():
        all_samples.extend(samples)
    
    for _ in range(repetitions):
        for text in all_samples:
            learn_from_text(lnn, text, repetitions=1)
    
    words = discover_words(lnn, all_samples, min_frequency=2)
    
    # Test each grammar category
    category_results = {}
    for category, samples in GRAMMAR_SAMPLES.items():
        cat_words = discover_words(lnn, samples, min_frequency=2)
        category_results[category] = len(cat_words)
        print(f"  {category}: {len(cat_words)} patterns")
    
    print(f"  Total grammar patterns: {len(words)}")
    
    return {"patterns": len(words), "categories": category_results}


# ============================================================================
# SENTENCES - Full sentence understanding
# ============================================================================

SENTENCE_SAMPLES = [
    # Simple sentences
    "the cat sits on the mat",
    "the dog runs in the park",
    "the bird flies over the tree",
    "i eat breakfast every morning",
    "she drinks coffee at noon",
    "he walks to school every day",
    
    # Complex sentences
    "the quick brown fox jumps over the lazy dog",
    "she sells seashells by the seashore",
    "how much wood would a woodchuck chuck",
    "i think therefore i am",
    "to be or not to be that is the question",
    
    # Question sentences
    "what is your name",
    "where do you live",
    "when does the train leave",
    "why did he leave so early",
    "how are you feeling today",
    
    # Action sentences
    "the child fell down and cried",
    "she wrote a letter to her friend",
    "he drove to the city alone",
    "they built a house last year",
    "we spent a week in the mountains",
]


def test_sentences(repetitions: int = 50) -> dict:
    """Test sentence learning."""
    print("\n=== SENTENCES TEST ===")
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    for _ in range(repetitions):
        for text in SENTENCE_SAMPLES:
            learn_from_text(lnn, text, repetitions=1)
    
    words = discover_words(lnn, SENTENCE_SAMPLES, min_frequency=2)
    
    # Check sentence-level patterns (multi-word)
    sentence_patterns = {}
    for sentence in SENTENCE_SAMPLES:
        tokens = sentence.split()
        for size in [2, 3, 4]:
            for i in range(len(tokens) - size + 1):
                pattern = " ".join(tokens[i:i+size])
                sentence_patterns[pattern] = sentence_patterns.get(pattern, 0) + 1
    
    # Find learned patterns
    learned_patterns = [p for p, count in sentence_patterns.items() 
                       if count >= repetitions // 5]
    
    print(f"  Word vocabulary: {len(words)}")
    print(f"  Learned phrases: {len(learned_patterns)}")
    print(f"  Sample words: {list(words.keys())[:15]}")
    print(f"  Sample phrases: {learned_patterns[:10]}")
    
    return {"words": len(words), "phrases": len(learned_patterns)}


# ============================================================================
# COMPREHENSIVE TEST
# ============================================================================

def main():
    print("=" * 60)
    print("LRN LANGUAGE LAYERS TEST")
    print("Phonetics -> Grammar -> Sentences")
    print("=" * 60)
    
    results = {}
    
    # 1. Phonetics
    phonetics = test_phonetics(repetitions=60)
    results["phonetics"] = phonetics
    
    # 2. Grammar
    grammar = test_grammar_ops(repetitions=60)
    results["grammar"] = grammar
    
    # 3. Sentences
    sentences = test_sentences(repetitions=60)
    results["sentences"] = sentences
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Phonetics: {phonetics['patterns']} patterns")
    print(f"  Grammar: {grammar['patterns']} patterns")
    print(f"  Sentences: {sentences['words']} words, {sentences['phrases']} phrases")
    
    return results


if __name__ == "__main__":
    main()