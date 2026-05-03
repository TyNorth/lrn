"""
Debug: Check what activates when querying specific words.
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
    "run walk jump swim fly crawl",
    "eat drink sleep read write think",
    "the cat runs", "the dog walks", "the bird flies",
]


def main():
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    print("Training...")
    train_enhanced(lnn, CORPUS, reps=30, learn_type="language")
    
    word_nodes = sum(1 for n in lnn.nodes if n.startswith("word:"))
    tau3 = sum(1 for sp in lnn.springs.values() if sp.tau == 3)
    
    print(f"Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
    print(f"Words: {word_nodes}, τ=3: {tau3}")
    
    # Debug: Check what activates for each query
    queries = ["word:cat", "word:apple", "word:red", "word:happy", "word:run"]
    
    for query in queries:
        print(f"\n=== Query: {query} ===")
        
        for n in lnn.nodes.values():
            n.activation = 0
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            propagate(lnn, n_steps=5)
            
            activated = [(n.replace("word:", ""), node.activation) 
                        for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            # Sort by activation
            activated.sort(key=lambda x: x[1], reverse=True)
            
            print(f"  Top 20 activated words:")
            for word, act in activated[:20]:
                print(f"    {word}: {act}")
            
            # Check springs from query
            if query in lnn.nodes:
                neighbors = lnn.get_neighbors(query)
                tau3_neighbors = [(n.replace("word:", ""), sp.tau, sp.stiffness) 
                                 for n, sp in neighbors if n.startswith("word:")]
                tau3_neighbors.sort(key=lambda x: x[2], reverse=True)
                
                print(f"  Top 10 τ=3 neighbors:")
                for word, tau, stiff in tau3_neighbors[:10]:
                    if tau == 3:
                        print(f"    {word}: τ={tau}, stiffness={stiff}")
        else:
            print(f"  Query node not found!")


if __name__ == "__main__":
    main()
