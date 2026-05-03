"""
Enhanced LRN Training - Optimal Learning Conditions
Implements architectural improvements for maximum learning.
"""
import sys
import time

sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.natural_tokenize import learn_from_text
from lrn.inference import add_word_nodes


def enhanced_rem(lnn, wake_buffer):
    """Enhanced REM - forms τ=2 AND τ=3 bridges, strengthens co-activated paths."""
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
                # Strengthen frequently co-occurring words
                sp.stiffness = max(sp.stiffness, 5)
                # Promote to τ=3 if strong enough (categorical bridge)
                if sp.stiffness >= 15:
                    sp.tau = 3
            else:
                # Create new τ=2 causal bridge
                lnn.add_or_update_spring(a, b, stiffness=5, tau=2, mode="add")


def hebbian_update(lnn, activated_nodes):
    """Strengthen springs between co-activated nodes."""
    node_list = list(activated_nodes)
    for i in range(len(node_list)):
        for j in range(i+1, len(node_list)):
            a, b = node_list[i], node_list[j]
            key = lnn._key(a, b)
            if key in lnn.springs:
                sp = lnn.springs[key]
                sp.stiffness += 1


def train_enhanced(lnn, sentences, reps=20, learn_type="language"):
    """Training with enhanced REM and Hebbian learning."""
    wake_buffer = []
    
    for rep in range(reps):
        for sentence in sentences:
            learn_from_text(lnn, sentence, repetitions=1, learn_type=learn_type)
            add_word_nodes(lnn, [sentence])
            wake_buffer.append(sentence)
            if len(wake_buffer) > 15:
                wake_buffer = wake_buffer[-15:]
        
        enhanced_rem(lnn, wake_buffer)
        propagate(lnn, n_steps=3)
        
        activated = [n for n, node in lnn.nodes.items() if node.activation > 10]
        if activated:
            hebbian_update(lnn, activated)
    
    enhanced_rem(lnn, wake_buffer)
    propagate(lnn, n_steps=5)


# Multi-domain corpus
DOMAINS = {
    "animals": [
        "the cat eats fish", "the dog runs fast", "the bird flies high",
        "the fish swims deep", "the horse runs fast", "the cow eats grass",
        "cat dog bird fish horse cow sheep pig chicken",
        "the lion is strong", "the tiger is fierce", "the bear is large",
    ],
    "fruits": [
        "the apple is red", "the banana is yellow", "the orange is orange",
        "the grape is purple", "the pear is green", "the peach is pink",
        "apple banana orange grape pear peach cherry berry",
        "fruits are sweet", "fruits are healthy", "fruits grow on trees",
    ],
    "colors": [
        "red blue green yellow orange purple",
        "black white brown gray pink gold silver",
        "the sky is blue", "the grass is green", "the sun is yellow",
        "red is warm", "blue is cool", "green is natural",
    ],
    "emotions": [
        "happy sad angry scared surprised tired",
        "glad joyful cheerful delighted pleased",
        "unhappy sorrowful miserable gloomy depressed",
        "i feel happy", "she looks sad", "he seems angry",
        "emotions are feelings", "feelings change quickly",
    ],
    "actions": [
        "run walk jump swim fly crawl",
        "eat drink sleep read write think",
        "the cat runs", "the dog walks", "the bird flies",
        "i eat food", "you drink water", "he sleeps late",
        "actions are verbs", "verbs describe doing",
    ],
    "relationships": [
        "the cat is on the mat", "the book is in the bag",
        "the bird is above the tree", "the fish is below the water",
        "the dog is near the house", "the car is far from home",
        "prepositions show location", "location is relative",
    ],
}


def main():
    print("=" * 60)
    print("ENHANCED LRN TRAINING - OPTIMAL CONDITIONS")
    print("=" * 60)
    
    start = time.time()
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    cumulative = []
    results = {}
    
    for domain, corpus in DOMAINS.items():
        print(f"\n{'='*40}")
        print(f"DOMAIN: {domain.upper()}")
        print(f"{'='*40}")
        
        cumulative.extend(corpus)
        
        t0 = time.time()
        train_enhanced(lnn, cumulative, reps=20, learn_type="language")
        
        word_nodes = sum(1 for n in lnn.nodes if n.startswith("word:"))
        tau2 = sum(1 for sp in lnn.springs.values() if sp.tau == 2)
        tau3 = sum(1 for sp in lnn.springs.values() if sp.tau == 3)
        avg_stiffness = sum(sp.stiffness for sp in lnn.springs.values()) / max(1, len(lnn.springs))
        
        elapsed = time.time() - t0
        print(f"  Corpus: {len(cumulative)} sentences")
        print(f"  Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
        print(f"  Words: {word_nodes}, τ=2: {tau2}, τ=3: {tau3}")
        print(f"  Avg stiffness: {avg_stiffness:.1f}")
        print(f"  Time: {elapsed:.1f}s")
        
        results[domain] = {
            "nodes": len(lnn.nodes),
            "springs": len(lnn.springs),
            "words": word_nodes,
            "tau2": tau2,
            "tau3": tau3,
            "avg_stiffness": avg_stiffness,
            "time": elapsed,
        }
    
    total_time = time.time() - start
    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {total_time:.1f}s")
    print(f"{'='*60}")
    
    # Comprehensive tests
    print("\n=== RETENTION TEST (after 100 propagation steps) ===")
    test_retention(lnn)
    
    print("\n=== ACCESS TEST (category activation) ===")
    test_access(lnn)
    
    print("\n=== GENERALIZATION TEST (novel combinations) ===")
    test_generalization(lnn)
    
    print("\n=== TRANSFER TEST (cross-domain) ===")
    test_transfer(lnn)
    
    print("\n=== FLUENCY TEST (activation speed) ===")
    test_fluency(lnn)
    
    return results


def test_retention(lnn):
    """Test if knowledge persists after many propagation steps."""
    tests = [
        ("word:cat", ["dog", "bird", "fish"]),
        ("word:apple", ["banana", "orange", "grape"]),
        ("word:red", ["blue", "green", "yellow"]),
    ]
    
    for query, expected in tests:
        for n in lnn.nodes.values():
            n.activation = 0
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            propagate(lnn, n_steps=100)  # Long propagation
            
            activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            found = [w for w in expected if w in activated]
            status = "PASS" if found else "FAIL"
            print(f"  {query.replace('word:', '')}: {status} (retained: {found})")
        else:
            print(f"  {query.replace('word:', '')}: FAIL (not found)")


def test_access(lnn):
    """Test if any category member activates others."""
    categories = {
        "animals": ["cat", "dog", "bird", "fish", "horse"],
        "fruits": ["apple", "banana", "orange", "grape", "pear"],
        "colors": ["red", "blue", "green", "yellow", "orange"],
        "emotions": ["happy", "sad", "angry", "scared", "tired"],
    }
    
    for category, members in categories.items():
        passed = 0
        for member in members:
            query = f"word:{member}"
            for n in lnn.nodes.values():
                n.activation = 0
            
            if query in lnn.nodes:
                lnn.nodes[query].activation = 100
                propagate(lnn, n_steps=5)
                
                activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                            if node.activation > 0 and n.startswith("word:") and n != query]
                
                others = [m for m in members if m != member and m in activated]
                if others:
                    passed += 1
        
        status = "PASS" if passed >= 3 else "FAIL"
        print(f"  {category}: {status} ({passed}/{len(members)} members activate others)")


def test_generalization(lnn):
    """Test if lattice handles novel combinations."""
    # Test if "the purple cat runs fast" makes sense
    test_sentences = [
        "the purple cat runs fast",
        "the happy dog eats fruit",
        "the angry bird flies high",
    ]
    
    for sentence in test_sentences:
        words = sentence.lower().split()
        query = f"word:{words[1]}"  # The adjective/noun
        
        for n in lnn.nodes.values():
            n.activation = 0
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            propagate(lnn, n_steps=5)
            
            activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            # Check if related words activate
            related = [w for w in words if w in activated]
            status = "PASS" if len(related) >= 2 else "FAIL"
            print(f"  '{sentence}': {status} (related: {related})")
        else:
            print(f"  '{sentence}': FAIL (query not found)")


def test_transfer(lnn):
    """Test if knowledge transfers across domains."""
    # Does "cat" (animal) activate "runs" (action)?
    transfer_tests = [
        ("word:cat", ["runs", "eats", "sleeps"]),
        ("word:apple", ["red", "yellow", "green"]),
        ("word:happy", ["smile", "laugh", "joy"]),
    ]
    
    for query, expected in transfer_tests:
        for n in lnn.nodes.values():
            n.activation = 0
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            propagate(lnn, n_steps=5)
            
            activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            found = [w for w in expected if w in activated]
            status = "PASS" if found else "FAIL"
            print(f"  {query.replace('word:', '')} → {expected}: {status} (found: {found})")
        else:
            print(f"  {query.replace('word:', '')}: FAIL")


def test_fluency(lnn):
    """Test activation speed - how many steps to reach stable state?"""
    queries = ["word:cat", "word:apple", "word:red", "word:happy"]
    
    for query in queries:
        if query not in lnn.nodes:
            print(f"  {query}: FAIL (not found)")
            continue
        
        # Measure steps to activation plateau
        lnn.nodes[query].activation = 100
        prev_activated = 0
        plateau_step = 0
        
        for step in range(1, 21):
            propagate(lnn, n_steps=1)
            activated = sum(1 for n, node in lnn.nodes.items() 
                          if node.activation > 0 and n.startswith("word:"))
            
            if activated == prev_activated:
                plateau_step = step
                break
            prev_activated = activated
        
        status = "PASS" if plateau_step <= 10 else "SLOW"
        print(f"  {query.replace('word:', '')}: {status} (plateau at step {plateau_step})")


if __name__ == "__main__":
    main()
