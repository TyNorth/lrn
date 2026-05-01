"""
Minimal test: Core stages only (phonics, grammar, sentences)
"""
import sys
import time

sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.natural_tokenize import learn_from_text
from lrn.inference import add_word_nodes


def run_rem(lnn, wake_buffer):
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


def train(lnn, sentences, reps=50, learn_type="language"):
    wake_buffer = []
    for rep in range(reps):
        for sentence in sentences:
            learn_from_text(lnn, sentence, repetitions=1, learn_type=learn_type)
            add_word_nodes(lnn, [sentence])
            wake_buffer.append(sentence)
            if len(wake_buffer) > 10:
                wake_buffer = wake_buffer[-10:]
        run_rem(lnn, wake_buffer)
        propagate(lnn, n_steps=2)
    run_rem(lnn, wake_buffer)
    propagate(lnn, n_steps=2)


# Phonics corpus
PHONICS = [
    "cat hat mat sat bat",
    "bike hike like Mike",
    "boat coat float goat",
    "moon June spoon noon",
]

# Grammar corpus
GRAMMAR = [
    "the cat eats fish",
    "the dog runs fast",
    "the big dog sleeps",
    "the small cat runs",
    "i eat food",
    "you drink water",
    "he runs fast",
    "she sleeps late",
]

# Sentence corpus
SENTENCES = [
    "the cat eats fish",
    "the dog sees bird",
    "if it rains i stay home",
    "if you study you pass",
    "the cat that sleeps eats",
    "i know that you are here",
]


def main():
    print("=" * 60)
    print("LRN ENGLISH MINIMAL TRAINING")
    print("=" * 60)
    
    start = time.time()
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    # Train phonics
    print("\n1. Training phonics...")
    train(lnn, PHONICS, reps=50, learn_type="sensory")
    print(f"  Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
    
    # Train grammar
    print("\n2. Training grammar...")
    train(lnn, GRAMMAR, reps=50, learn_type="language")
    print(f"  Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
    
    # Train sentences
    print("\n3. Training sentences...")
    train(lnn, SENTENCES, reps=50, learn_type="language")
    print(f"  Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
    
    elapsed = time.time() - start
    print(f"\nTraining time: {elapsed:.1f}s")
    
    # Test phonics
    print("\n=== PHONICS TEST ===")
    test_cases = [
        ("cat", ["hat", "mat", "sat", "bat"]),
        ("bike", ["hike", "like"]),
        ("boat", ["coat", "float"]),
        ("moon", ["spoon", "noon"]),
    ]
    
    phonics_passed = 0
    for word, family in test_cases:
        query = f"word:{word}"
        for n in lnn.nodes.values():
            n.activation = 0
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            propagate(lnn, n_steps=5)
            activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            found = [w for w in family if w in activated]
            if found:
                phonics_passed += 1
                print(f"  {word}: PASS ({found})")
            else:
                print(f"  {word}: FAIL")
    
    print(f"Phonics: {phonics_passed}/{len(test_cases)}")
    
    # Test grammar
    print("\n=== GRAMMAR TEST ===")
    from lrn.grammar_training import infer_pos
    
    test_words = {
        "cat": "noun", "eats": "verb", "big": "adjective",
        "the": "determiner", "runs": "verb", "dog": "noun",
    }
    
    grammar_passed = 0
    for word, expected in test_words.items():
        result = infer_pos(lnn, word)
        actual = result.get("pos", "unknown")
        if actual == expected:
            grammar_passed += 1
            print(f"  {word}: PASS ({actual})")
        else:
            print(f"  {word}: FAIL (expected {expected}, got {actual})")
    
    print(f"Grammar: {grammar_passed}/{len(test_words)}")
    
    # Test generation
    print("\n=== GENERATION TEST ===")
    from lrn.inference import attention_with_residue
    
    seeds = ["the", "i", "you", "she", "he", "a"]
    gen_passed = 0
    for seed in seeds:
        result = attention_with_residue(lnn, f"word:{seed}", propagate_steps=3)
        if result["attention"]:
            gen_passed += 1
            words = [n.replace("word:", "") for n, _ in result["attention"][:2]]
            print(f"  '{seed}' → {words}")
    
    print(f"Generation: {gen_passed}/{len(seeds)}")
    
    # Overall
    total = phonics_passed + grammar_passed + gen_passed
    max_total = len(test_cases) + len(test_words) + len(seeds)
    pct = (total * 100) // max_total
    
    print(f"\nTOTAL: {total}/{max_total} ({pct}%)")
    
    return total, max_total


if __name__ == "__main__":
    main()
