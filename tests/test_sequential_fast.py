"""
Fast sequential training - reduced thresholds for speed.
"""
import sys
import time

sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.natural_tokenize import learn_from_text
from lrn.inference import add_word_nodes


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


# Simplified corpus
STAGES = {
    "sensory": ["A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"],
    "phonics": [
        "cat hat mat sat bat", "bike hike like", "boat coat float",
        "moon June spoon", "bed red fed", "bit sit fit",
    ],
    "grammar": [
        "the cat eats fish", "the dog runs fast", "the big dog sleeps",
        "i eat food", "you drink water", "he runs fast", "she sleeps late",
    ],
    "vocabulary": [
        "cat dog bird fish horse cow",
        "apple pear banana orange grape",
        "red blue green yellow orange purple",
        "car truck bus bike train plane",
        "head face eye nose mouth ear hand",
    ],
    "sentences": [
        "the cat sits on the mat", "the dog runs in the park",
        "if it rains i stay home", "if you study you pass",
        "the cat that sleeps eats", "i know that you are here",
    ],
}


def main():
    print("=" * 60)
    print("FAST SEQUENTIAL TRAINING")
    print("=" * 60)
    
    start = time.time()
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    results = {}
    for stage, corpus in STAGES.items():
        t0 = time.time()
        learn_type = "sensory" if stage in ("sensory", "phonics") else "language"
        train(lnn, corpus, reps=20, learn_type=learn_type)
        
        word_nodes = sum(1 for n in lnn.nodes if n.startswith("word:"))
        tau3 = sum(1 for sp in lnn.springs.values() if sp.tau == 3)
        
        elapsed = time.time() - t0
        print(f"{stage}: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs, {word_nodes} words, {tau3} τ=3 ({elapsed:.1f}s)")
        
        results[stage] = {
            "nodes": len(lnn.nodes),
            "springs": len(lnn.springs),
            "words": word_nodes,
            "tau3": tau3,
        }
    
    total_time = time.time() - start
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Final: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    # Test categories
    print("\n=== CATEGORY TEST ===")
    test_categories(lnn)
    
    # Test phonics
    print("\n=== PHONICS TEST ===")
    test_phonics(lnn)
    
    # Test grammar
    print("\n=== GRAMMAR TEST ===")
    test_grammar(lnn)
    
    return results


def test_categories(lnn):
    """Test if category members cluster together."""
    tests = {
        "animals": ("word:cat", ["dog", "bird", "fish", "horse"]),
        "fruits": ("word:apple", ["pear", "banana", "orange"]),
        "colors": ("word:red", ["blue", "green", "yellow"]),
        "vehicles": ("word:car", ["truck", "bus", "bike"]),
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
            print(f"  {category}: {status} (found: {found})")
        else:
            print(f"  {category}: FAIL (query node not found)")


def test_phonics(lnn):
    """Test phonics word family recognition."""
    tests = [
        ("cat", ["hat", "mat", "sat"]),
        ("bike", ["hike", "like"]),
        ("boat", ["coat", "float"]),
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
    }
    
    for word, expected in test_words.items():
        result = infer_pos(lnn, word)
        actual = result.get("pos", "unknown")
        status = "PASS" if actual == expected else "FAIL"
        print(f"  {word}: {status} (expected {expected}, got {actual})")


if __name__ == "__main__":
    main()
