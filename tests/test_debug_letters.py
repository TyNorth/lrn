"""
Debug: How are single letters stored in the lattice?
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


def train(lnn, sentences, reps=30, learn_type="language"):
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
    "A a B b C c D d E e F f G g H h I i J j K k L l M m",
    "N n O o P p Q q R r S s T t U u V v W w X x Y y Z z",
    "A is for apple a is for apple",
    "B is for ball b is for ball",
    "C is for cat c is for cat",
    "cat hat mat sat bat",
    "dog log fog hog jog",
]


def main():
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    print("Training...")
    train(lnn, CORPUS, reps=30, learn_type="sensory")
    
    print(f"Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
    
    # Check how letters are stored
    print("\n=== LETTER NODES ===")
    for letter in "abc":
        lower = f"word:{letter}"
        upper = f"word:{letter.upper()}"
        
        print(f"\n  '{letter}':")
        print(f"    {lower} exists: {lower in lnn.nodes}")
        print(f"    {upper} exists: {upper in lnn.nodes}")
        
        # Check all nodes containing this letter
        matching = [n for n in lnn.nodes if letter in n.lower()]
        if matching:
            print(f"    Matching nodes: {matching[:10]}")
        
        # Check springs from word:a
        if lower in lnn.nodes:
            neighbors = lnn.get_neighbors(lower)
            word_neighbors = [(n, sp.tau, sp.stiffness) for n, sp in neighbors if n.startswith("word:")]
            print(f"    Springs from {lower}: {word_neighbors[:5]}")
    
    # Check what happens when we activate "word:a"
    print("\n=== ACTIVATION TEST ===")
    for letter in "abc":
        query = f"word:{letter}"
        if query in lnn.nodes:
            for n in lnn.nodes.values():
                n.activation = 0
            
            lnn.nodes[query].activation = 100
            propagate(lnn, n_steps=1)
            
            activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            print(f"  {query} activates: {activated[:10]}")
        else:
            print(f"  {query}: NOT FOUND")


if __name__ == "__main__":
    main()
