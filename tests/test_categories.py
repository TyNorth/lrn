"""
Test: Semantic Category Learning
Can the LRN learn that cat/dog are animals, apple/pear are fruit, etc?
"""
import sys
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


# Category training corpus
CATEGORY_CORPUS = [
    # Animals
    "the cat is an animal",
    "the dog is an animal",
    "the bird is an animal",
    "the fish is an animal",
    "the horse is an animal",
    "the cow is an animal",
    "the pig is an animal",
    "the sheep is an animal",
    
    # Fruits
    "the apple is a fruit",
    "the pear is a fruit",
    "the banana is a fruit",
    "the orange is a fruit",
    "the grape is a fruit",
    "the peach is a fruit",
    "the cherry is a fruit",
    
    # Colors
    "red is a color",
    "blue is a color",
    "green is a color",
    "yellow is a color",
    "orange is a color",
    "purple is a color",
    
    # Vehicles
    "the car is a vehicle",
    "the truck is a vehicle",
    "the bus is a vehicle",
    "the bike is a vehicle",
    "the train is a vehicle",
    "the plane is a vehicle",
    
    # Body parts
    "the hand is a body part",
    "the foot is a body part",
    "the eye is a body part",
    "the ear is a body part",
    "the nose is a body part",
    "the mouth is a body part",
    
    # Cross-category sentences
    "the cat eats the fish",
    "the dog sees the bird",
    "the apple is red",
    "the car is blue",
    "the bird has a wing",
]


def test_categories(lnn):
    """Test if category members cluster together."""
    test_categories = {
        "animals": {
            "query": "word:cat",
            "expected": ["dog", "bird", "fish", "horse", "cow"],
        },
        "fruits": {
            "query": "word:apple",
            "expected": ["pear", "banana", "orange", "grape"],
        },
        "colors": {
            "query": "word:red",
            "expected": ["blue", "green", "yellow", "orange", "purple"],
        },
        "vehicles": {
            "query": "word:car",
            "expected": ["truck", "bus", "bike", "train", "plane"],
        },
        "body_parts": {
            "query": "word:hand",
            "expected": ["foot", "eye", "ear", "nose", "mouth"],
        },
    }
    
    results = []
    for category_name, test in test_categories.items():
        query = test["query"]
        expected = test["expected"]
        
        # Reset activations
        for n in lnn.nodes.values():
            n.activation = 0
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            propagate(lnn, n_steps=5)
            
            activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            found = [w for w in expected if w in activated]
            passed = len(found) >= 2  # At least 2 category members found
            
            results.append({
                "category": category_name,
                "query": query.replace("word:", ""),
                "expected": expected,
                "found": found,
                "passed": passed,
            })
    
    return results


def main():
    print("=" * 60)
    print("SEMANTIC CATEGORY LEARNING TEST")
    print("=" * 60)
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    print("\nTraining category corpus...")
    train(lnn, CATEGORY_CORPUS, reps=50, learn_type="language")
    
    print(f"Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
    
    # Check τ=3 springs
    tau3 = sum(1 for sp in lnn.springs.values() if sp.tau == 3)
    print(f"τ=3 categorical springs: {tau3}")
    
    # Test categories
    print("\n=== CATEGORY TEST ===")
    results = test_categories(lnn)
    
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"\n  {r['category'].upper()} (query: {r['query']}):")
        print(f"    Expected: {r['expected']}")
        print(f"    Found: {r['found']}")
        print(f"    Status: {status}")
    
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    
    print(f"\n=== SUMMARY ===")
    print(f"Categories: {passed}/{total} passed")
    
    for r in results:
        status = "✓" if r["passed"] else "✗"
        print(f"  {status} {r['category']}: {len(r['found'])}/{len(r['expected'])} found")
    
    return passed, total


if __name__ == "__main__":
    main()
