"""
Verify: Knowledge is stored in springs, activations are transient.
Test if single propagation step activates category members.
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.natural_tokenize import learn_from_text
from lrn.inference import add_word_nodes


def optimal_rem(lnn, wake_buffer):
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
                sp.stiffness = max(sp.stiffness, 10)
                sp.tau = 3
            else:
                lnn.add_or_update_spring(a, b, stiffness=10, tau=3, mode="add")


def train_optimal(lnn, sentences, reps=20, learn_type="language"):
    wake_buffer = []
    for rep in range(reps):
        for sentence in sentences:
            learn_from_text(lnn, sentence, repetitions=1, learn_type=learn_type)
            add_word_nodes(lnn, [sentence])
            
            wake_buffer.append(sentence)
            if len(wake_buffer) > 20:
                wake_buffer = wake_buffer[-20:]
            
            optimal_rem(lnn, wake_buffer)
        
        propagate(lnn, n_steps=3)
    
    propagate(lnn, n_steps=5)


CORPUS = [
    "the cat eats fish", "the dog runs fast", "the bird flies high",
    "the fish swims deep", "the horse runs fast", "the cow eats grass",
    "the cat and dog are animals", "the bird and fish are animals",
    "cat dog bird fish horse cow sheep pig",
    "the apple is red", "the banana is yellow", "the orange is orange",
    "the grape is purple", "the pear is green", "the peach is pink",
    "apple banana orange grape pear peach cherry",
    "red blue green yellow orange purple",
    "the sky is blue", "the grass is green", "the sun is yellow",
    "happy sad angry scared surprised tired",
    "glad joyful cheerful delighted pleased",
    "i feel happy", "she looks sad", "he seems angry",
]


def main():
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    print("Training...")
    train_optimal(lnn, CORPUS, reps=30, learn_type="language")
    
    print(f"Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
    
    # Test: Single propagation step activation
    print("\n=== SINGLE STEP ACTIVATION ===")
    
    tests = [
        ("word:cat", ["dog", "bird", "fish", "horse"]),
        ("word:apple", ["banana", "orange", "grape", "pear"]),
        ("word:red", ["blue", "green", "yellow", "orange"]),
        ("word:happy", ["sad", "angry", "glad", "joyful"]),
    ]
    
    for query, expected in tests:
        for n in lnn.nodes.values():
            n.activation = 0
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            
            # Check after 1 step
            propagate(lnn, n_steps=1)
            
            activated = [(n.replace("word:", ""), node.activation) 
                        for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            activated.sort(key=lambda x: x[1], reverse=True)
            
            found = [w for w, _ in activated if w in expected]
            status = "PASS" if len(found) >= 2 else "FAIL"
            
            print(f"\n  {query.replace('word:', '')} (after 1 step):")
            print(f"    Top 10: {[w for w, _ in activated[:10]]}")
            print(f"    Expected found: {found}")
            print(f"    Status: {status}")
        else:
            print(f"  {query.replace('word:', '')}: FAIL (not found)")


if __name__ == "__main__":
    main()
