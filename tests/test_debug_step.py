"""
Debug: Check activation after each propagation step.
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.natural_tokenize import learn_from_text
from lrn.inference import add_word_nodes


def enhanced_rem(lnn, wake_buffer):
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
                if sp.stiffness >= 10:
                    sp.tau = 3
            else:
                lnn.add_or_update_spring(a, b, stiffness=5, tau=2, mode="add")


def train_enhanced(lnn, sentences, reps=20, learn_type="language"):
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
    
    enhanced_rem(lnn, wake_buffer)
    propagate(lnn, n_steps=5)


CORPUS = [
    "the cat eats fish", "the dog runs fast", "the bird flies high",
    "the cat and dog are animals", "cat dog bird fish horse",
    "the apple is red", "the banana is yellow", "apple banana orange",
    "red blue green yellow", "happy sad angry", "run walk jump",
]


def main():
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    print("Training...")
    train_enhanced(lnn, CORPUS, reps=30, learn_type="language")
    
    print(f"Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
    
    # Check "cat" springs
    query = "word:cat"
    if query in lnn.nodes:
        neighbors = lnn.get_neighbors(query)
        print(f"\nSprings from 'cat':")
        for n, sp in neighbors:
            if n.startswith("word:"):
                print(f"  {n}: τ={sp.tau}, stiffness={sp.stiffness}")
    
    # Track activation step by step
    print(f"\n=== Activation tracking for '{query}' ===")
    
    for n in lnn.nodes.values():
        n.activation = 0
    
    lnn.nodes[query].activation = 100
    
    # Track specific words
    track_words = ["cat", "dog", "bird", "fish", "horse", "eats", "runs"]
    
    for step in range(1, 11):
        propagate(lnn, n_steps=1)
        
        print(f"\nStep {step}:")
        for w in track_words:
            node_name = f"word:{w}"
            if node_name in lnn.nodes:
                act = lnn.nodes[node_name].activation
                if act > 0:
                    print(f"  {w}: {act}")
        
        # Count total active word nodes
        active = sum(1 for n, node in lnn.nodes.items() 
                    if node.activation > 0 and n.startswith("word:"))
        print(f"  Total active word nodes: {active}")


if __name__ == "__main__":
    main()
