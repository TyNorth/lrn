"""
ARCHITECTURAL INSIGHT: Retention vs Activation

The LRN stores knowledge in SPRINGS (stiffness, tau), not node activations.
Node activations are transient - they're the "working memory".
Springs are the "long-term memory".

RETENTION means: Can we re-activate the same pattern by querying a related node?
Not: Does activation persist forever?

This matches how brains work:
- Working memory: transient neural firing (activations)
- Long-term memory: synaptic strength (springs)
- Recall: re-activating patterns via strong connections

OPTIMAL CONDITIONS FOR LEARNING:
1. τ=3 categorical bridges form dense clusters (semantic memory)
2. τ=2 causal bridges connect clusters (relational memory)
3. High stiffness springs create fast activation paths (fluency)
4. Negative springs prevent incorrect activations (error correction)
5. REM synthesis after meaningful chunks (consolidation)
6. Hebbian learning strengthens frequently used paths (practice)
"""
import sys
import time

sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.natural_tokenize import learn_from_text
from lrn.inference import add_word_nodes


def enhanced_rem(lnn, wake_buffer):
    """Enhanced REM - forms τ=2 AND τ=3 bridges."""
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
                sp.stiffness = max(sp.stiffness, 5)
                if sp.stiffness >= 10:  # Lower threshold for τ=3
                    sp.tau = 3
            else:
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


# Comprehensive corpus
CORPUS = [
    # Animals - in sentence context
    "the cat eats fish", "the dog runs fast", "the bird flies high",
    "the fish swims deep", "the horse runs fast", "the cow eats grass",
    "the cat and dog are animals", "the bird and fish are animals",
    "the horse and cow are animals", "the sheep and pig are animals",
    "the chicken and bird are animals", "the lion is a strong animal",
    "the tiger is a fierce animal", "the bear is a large animal",
    "cat dog bird fish horse cow sheep pig",
    
    # Fruits - in sentence context
    "the apple is red", "the banana is yellow", "the orange is orange",
    "the grape is purple", "the pear is green", "the peach is pink",
    "the apple and banana are fruits", "the orange and grape are fruits",
    "the pear and peach are fruits", "the cherry and berry are fruits",
    "fruits are sweet and healthy", "fruits grow on trees",
    "apple banana orange grape pear peach cherry",
    
    # Colors - in sentence context
    "red blue green yellow orange purple",
    "black white brown gray pink gold silver",
    "the sky is blue", "the grass is green", "the sun is yellow",
    "red is a warm color", "blue is a cool color", "green is a natural color",
    "the red apple", "the blue sky", "the green grass",
    "the yellow sun", "the orange fruit", "the purple grape",
    
    # Emotions - in sentence context
    "happy sad angry scared surprised tired",
    "glad joyful cheerful delighted pleased",
    "unhappy sorrowful miserable gloomy depressed",
    "i feel happy", "she looks sad", "he seems angry",
    "happy and glad are similar", "sad and unhappy are similar",
    "angry and mad are similar", "scared and afraid are similar",
    
    # Actions - in sentence context
    "run walk jump swim fly crawl",
    "eat drink sleep read write think",
    "the cat runs", "the dog walks", "the bird flies",
    "i eat food", "you drink water", "he sleeps late",
    "run and walk are actions", "jump and swim are actions",
    "eat and drink are actions", "read and write are actions",
    
    # Grammar patterns
    "the big dog runs", "the small cat sleeps",
    "the fast car drives", "the slow turtle walks",
    "the hot fire burns", "the cold ice melts",
]


def main():
    print("=" * 60)
    print("OPTIMAL LEARNING CONDITIONS TEST")
    print("=" * 60)
    
    start = time.time()
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    print("\nTraining...")
    t0 = time.time()
    train_enhanced(lnn, CORPUS, reps=50, learn_type="language")
    
    word_nodes = sum(1 for n in lnn.nodes if n.startswith("word:"))
    tau2 = sum(1 for sp in lnn.springs.values() if sp.tau == 2)
    tau3 = sum(1 for sp in lnn.springs.values() if sp.tau == 3)
    
    print(f"Training time: {time.time() - t0:.1f}s")
    print(f"Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
    print(f"Words: {word_nodes}, τ=2: {tau2}, τ=3: {tau3}")
    
    # Test recall (proper retention test)
    print("\n=== RECALL TEST (re-activation via query) ===")
    test_recall(lnn)
    
    print("\n=== CATEGORY ACCESS TEST ===")
    test_category_access(lnn)
    
    print("\n=== FLUENCY TEST (activation speed) ===")
    test_fluency(lnn)
    
    print("\n=== TRANSFER TEST (cross-domain) ===")
    test_transfer(lnn)
    
    total_time = time.time() - start
    print(f"\nTotal time: {total_time:.1f}s")


def test_recall(lnn):
    """Test if knowledge can be recalled by querying related nodes."""
    tests = [
        ("word:cat", ["dog", "bird", "fish", "horse"]),
        ("word:apple", ["banana", "orange", "grape", "pear"]),
        ("word:red", ["blue", "green", "yellow", "orange"]),
        ("word:happy", ["sad", "angry", "glad", "joyful"]),
        ("word:run", ["walk", "jump", "swim", "fly"]),
    ]
    
    for query, expected in tests:
        for n in lnn.nodes.values():
            n.activation = 0
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            propagate(lnn, n_steps=5)
            
            activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            found = [w for w in expected if w in activated]
            recall_rate = len(found) / len(expected)
            status = "PASS" if recall_rate >= 0.5 else "FAIL"
            print(f"  {query.replace('word:', '')}: {status} ({len(found)}/{len(expected)}: {found})")
        else:
            print(f"  {query.replace('word:', '')}: FAIL (not found)")


def test_category_access(lnn):
    """Test if any category member activates others."""
    categories = {
        "animals": ["cat", "dog", "bird", "fish", "horse"],
        "fruits": ["apple", "banana", "orange", "grape", "pear"],
        "colors": ["red", "blue", "green", "yellow", "orange"],
        "emotions": ["happy", "sad", "angry", "glad", "joyful"],
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


def test_fluency(lnn):
    """Test activation speed."""
    queries = ["word:cat", "word:apple", "word:red", "word:happy", "word:run"]
    
    for query in queries:
        if query not in lnn.nodes:
            print(f"  {query}: FAIL (not found)")
            continue
        
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


def test_transfer(lnn):
    """Test cross-domain transfer."""
    tests = [
        ("word:cat", ["runs", "eats"]),  # animal → action
        ("word:apple", ["red", "yellow"]),  # fruit → color
        ("word:happy", ["smile", "laugh"]),  # emotion → action
    ]
    
    for query, expected in tests:
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


if __name__ == "__main__":
    main()
