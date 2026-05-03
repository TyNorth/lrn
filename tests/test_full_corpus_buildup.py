"""
Full English corpus training with iterative buildup.
Uses the complete corpus from english_corpus.py.
"""
import sys
import time

sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.natural_tokenize import learn_from_text
from lrn.inference import add_word_nodes
from lrn.english_corpus import (
    LETTER_CORPUS, BABBLING_CORPUS, PHONICS_CORPUS,
    MORPHOLOGY_CORPUS, SIGHT_WORDS_CORPUS, VOCABULARY_CORPUS,
    GRAMMAR_CORPUS, SYNTAX_CORPUS, SENTENCE_CORPUS, PRAGMATICS_CORPUS,
)


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


def run_rem(lnn, wake_buffer):
    """Lightweight REM - forms τ=3 categorical bridges."""
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


def train(lnn, sentences, reps=20, learn_type="language"):
    wake_buffer = []
    for rep in range(reps):
        for sentence in sentences:
            learn_from_text(lnn, sentence, repetitions=1, learn_type=learn_type)
            add_word_nodes(lnn, [sentence])
            wake_buffer.append(sentence)
            if len(wake_buffer) > 5:
                wake_buffer = wake_buffer[-5:]
        run_rem(lnn, wake_buffer)
        propagate(lnn, n_steps=2)
    run_rem(lnn, wake_buffer)
    propagate(lnn, n_steps=2)


# Full stage order with complete corpora
STAGES = [
    ("sensory", LETTER_CORPUS, "sensory"),
    ("babbling", BABBLING_CORPUS, "sensory"),
    ("phonics", PHONICS_CORPUS, "sensory"),
    ("morphology", MORPHOLOGY_CORPUS, "sensory"),
    ("sight_words", SIGHT_WORDS_CORPUS, "language"),
    ("vocabulary", VOCABULARY_CORPUS, "language"),
    ("grammar", GRAMMAR_CORPUS, "language"),
    ("syntax", SYNTAX_CORPUS, "language"),
    ("sentences", SENTENCE_CORPUS, "language"),
    ("pragmatics", PRAGMATICS_CORPUS, "language"),
]


def main():
    print("=" * 60)
    print("FULL ENGLISH CORPUS - ITERATIVE BUILDUP")
    print("=" * 60)
    
    start = time.time()
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    cumulative_corpus = []
    results = {}
    
    for stage_name, corpus, learn_type in STAGES:
        print(f"\n{'='*40}")
        print(f"STAGE: {stage_name.upper()}")
        print(f"{'='*40}")
        
        # Add this stage's corpus to cumulative
        new_sentences = flatten_corpus(corpus)
        cumulative_corpus.extend(new_sentences)
        
        t0 = time.time()
        train(lnn, cumulative_corpus, reps=20, learn_type=learn_type)
        
        word_nodes = sum(1 for n in lnn.nodes if n.startswith("word:"))
        tau3 = sum(1 for sp in lnn.springs.values() if sp.tau == 3)
        
        elapsed = time.time() - t0
        print(f"  Corpus: {len(cumulative_corpus)} sentences (+{len(new_sentences)} new)")
        print(f"  Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
        print(f"  Words: {word_nodes}, τ=3: {tau3}")
        print(f"  Time: {elapsed:.1f}s")
        
        results[stage_name] = {
            "nodes": len(lnn.nodes),
            "springs": len(lnn.springs),
            "words": word_nodes,
            "tau3": tau3,
            "time": elapsed,
        }
    
    total_time = time.time() - start
    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {total_time:.1f}s")
    print(f"{'='*60}")
    
    # Run tests
    print("\n=== CATEGORY TEST ===")
    test_categories(lnn)
    
    print("\n=== PHONICS TEST ===")
    test_phonics(lnn)
    
    print("\n=== GRAMMAR TEST ===")
    test_grammar(lnn)
    
    print("\n=== GENERATION TEST ===")
    test_generation(lnn)
    
    return results


def test_categories(lnn):
    """Test if category members cluster together."""
    tests = {
        "animals": ("word:cat", ["dog", "bird", "fish", "horse", "cow", "sheep", "pig"]),
        "fruits": ("word:apple", ["pear", "banana", "orange", "grape", "peach"]),
        "colors": ("word:red", ["blue", "green", "yellow", "orange", "purple"]),
        "vehicles": ("word:car", ["truck", "bus", "bike", "train", "plane"]),
        "body_parts": ("word:head", ["face", "eye", "nose", "mouth", "ear"]),
        "emotions": ("word:happy", ["sad", "angry", "scared", "excited"]),
    }
    
    for category, (query, expected) in tests.items():
        for n in lnn.nodes.values():
            n.activation = 0
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            propagate(lnn, n_steps=5)
            
            activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            found = [w for w in expected if w in activated]
            status = "PASS" if len(found) >= 2 else "FAIL"
            print(f"  {category}: {status} ({len(found)}/{len(expected)} found: {found})")
        else:
            print(f"  {category}: FAIL (query node not found)")


def test_phonics(lnn):
    """Test phonics word family recognition."""
    tests = [
        ("cat", ["hat", "mat", "sat", "bat"]),
        ("bike", ["hike", "like"]),
        ("boat", ["coat", "float"]),
        ("moon", ["spoon", "noon"]),
        ("cake", ["make", "take"]),
    ]
    
    for word, family in tests:
        query = f"word:{word}"
        for n in lnn.nodes.values():
            n.activation = 0
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            propagate(lnn, n_steps=5)
            
            activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            found = [w for w in family if w in activated]
            status = "PASS" if found else "FAIL"
            print(f"  {word}: {status} (found: {found})")
        else:
            print(f"  {word}: FAIL (query node not found)")


def test_grammar(lnn):
    """Test POS tagging."""
    from lrn.grammar_training import infer_pos
    
    test_words = {
        "cat": "noun", "eats": "verb", "big": "adjective",
        "the": "determiner", "runs": "verb", "dog": "noun",
        "fast": "adjective", "sleeps": "verb",
    }
    
    for word, expected in test_words.items():
        result = infer_pos(lnn, word)
        actual = result.get("pos", "unknown")
        status = "PASS" if actual == expected else "FAIL"
        print(f"  {word}: {status} (expected {expected}, got {actual})")


def test_generation(lnn):
    """Test generation from seed words."""
    from lrn.inference import attention_with_residue
    
    seeds = ["the", "cat", "i", "you", "if", "she", "he"]
    
    for seed in seeds:
        result = attention_with_residue(lnn, f"word:{seed}", propagate_steps=3)
        if result["attention"]:
            words = [n.replace("word:", "") for n, _ in result["attention"][:3]]
            print(f"  '{seed}' → {words}")
        else:
            print(f"  '{seed}' → (no generation)")


if __name__ == "__main__":
    main()
