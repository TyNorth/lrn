"""
Iterative Buildup Training

Each stage builds on all previous stages.
Train sensory → phonics (sensory+phonics) → grammar (sensory+phonics+grammar) etc.
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


# Stage corpora
STAGES = {
    "sensory": [
        "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
        "a b c d e f g h i j k l m n o p q r s t u v w x y z",
    ],
    "phonics": [
        "cat hat mat sat bat", "bike hike like", "boat coat float",
        "moon June spoon", "bed red fed", "bit sit fit",
        "hot pot lot", "bug rug mug", "can fan man",
        "cake make take", "name game same",
    ],
    "vocabulary": [
        "cat dog bird fish horse cow sheep pig",
        "apple pear banana orange grape peach",
        "red blue green yellow orange purple",
        "car truck bus bike train plane",
        "head face eye nose mouth ear hand arm leg foot",
        "big large huge small tiny little",
        "fast quick rapid slow sluggish",
        "happy glad joyful sad unhappy sorrowful",
        "hot warm cold freezing cool chilly",
        "good great excellent bad terrible awful",
    ],
    "grammar": [
        "the cat eats fish", "the dog runs fast", "the big dog sleeps",
        "the small cat runs", "i eat food", "you drink water",
        "he runs fast", "she sleeps late", "it works well",
        "the fast car drives", "the hot fire burns", "the cold ice melts",
        "in the house", "on the table", "under the bed",
    ],
    "sentences": [
        "the cat sits on the mat", "the dog runs in the park",
        "if it rains i stay home", "if you study you pass",
        "the cat that sleeps eats", "i know that you are here",
        "the bird that flies high sees far",
        "she says that he comes tomorrow",
    ],
}

STAGE_ORDER = ["sensory", "phonics", "vocabulary", "grammar", "sentences"]


def main():
    print("=" * 60)
    print("ITERATIVE BUILDUP TRAINING")
    print("=" * 60)
    
    start = time.time()
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    cumulative_corpus = []
    results = {}
    
    for stage in STAGE_ORDER:
        print(f"\n{'='*40}")
        print(f"STAGE: {stage.upper()}")
        print(f"{'='*40}")
        
        # Add this stage's corpus to cumulative
        cumulative_corpus.extend(STAGES[stage])
        
        learn_type = "sensory" if stage in ("sensory", "phonics") else "language"
        
        t0 = time.time()
        train(lnn, cumulative_corpus, reps=20, learn_type=learn_type)
        
        word_nodes = sum(1 for n in lnn.nodes if n.startswith("word:"))
        tau3 = sum(1 for sp in lnn.springs.values() if sp.tau == 3)
        
        elapsed = time.time() - t0
        print(f"  Corpus size: {len(cumulative_corpus)} sentences")
        print(f"  Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
        print(f"  Words: {word_nodes}, τ=3: {tau3}")
        print(f"  Time: {elapsed:.1f}s")
        
        results[stage] = {
            "nodes": len(lnn.nodes),
            "springs": len(lnn.springs),
            "words": word_nodes,
            "tau3": tau3,
        }
    
    total_time = time.time() - start
    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {total_time:.1f}s")
    print(f"{'='*60}")
    
    # Run all tests
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
        "animals": ("word:cat", ["dog", "bird", "fish", "horse", "cow"]),
        "fruits": ("word:apple", ["pear", "banana", "orange", "grape"]),
        "colors": ("word:red", ["blue", "green", "yellow", "orange"]),
        "vehicles": ("word:car", ["truck", "bus", "bike", "train"]),
        "body_parts": ("word:head", ["face", "eye", "nose", "mouth"]),
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
        ("moon", ["spoon", "June"]),
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
    
    seeds = ["the", "cat", "i", "you", "if"]
    
    for seed in seeds:
        result = attention_with_residue(lnn, f"word:{seed}", propagate_steps=3)
        if result["attention"]:
            words = [n.replace("word:", "") for n, _ in result["attention"][:3]]
            print(f"  '{seed}' → {words}")
        else:
            print(f"  '{seed}' → (no generation)")


if __name__ == "__main__":
    main()
