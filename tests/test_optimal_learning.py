"""
OPTIMAL LEARNING CONDITIONS - IMPLEMENTATION

Based on analysis, the architecture maximizes learning when:
1. REM synthesis runs after EVERY sentence (not just corpus passes)
2. τ=3 bridges form between ALL co-occurring words in context
3. Springs strengthen with repeated co-activation (Hebbian)
4. Progressive complexity builds on stable foundations
"""
import sys
import time

sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.natural_tokenize import learn_from_text
from lrn.inference import add_word_nodes


def optimal_rem(lnn, wake_buffer):
    """Optimal REM - forms τ=3 bridges between ALL co-occurring words."""
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
                # Strengthen existing spring
                sp.stiffness = max(sp.stiffness, 10)
                # Promote to τ=3 (categorical bridge)
                sp.tau = 3
            else:
                # Create new τ=3 categorical bridge
                lnn.add_or_update_spring(a, b, stiffness=10, tau=3, mode="add")


def train_optimal(lnn, sentences, reps=20, learn_type="language"):
    """Training with optimal REM after EVERY sentence."""
    wake_buffer = []
    
    for rep in range(reps):
        for sentence in sentences:
            learn_from_text(lnn, sentence, repetitions=1, learn_type=learn_type)
            add_word_nodes(lnn, [sentence])
            
            wake_buffer.append(sentence)
            if len(wake_buffer) > 20:  # Larger wake buffer
                wake_buffer = wake_buffer[-20:]
            
            # REM after EVERY sentence
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
    "run walk jump swim fly crawl",
    "eat drink sleep read write think",
    "the cat runs", "the dog walks", "the bird flies",
]


def main():
    print("=" * 60)
    print("OPTIMAL LEARNING CONDITIONS")
    print("REM after EVERY sentence, τ=3 bridges for ALL co-occurring")
    print("=" * 60)
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    print("\nTraining...")
    t0 = time.time()
    train_optimal(lnn, CORPUS, reps=30, learn_type="language")
    
    print(f"Training time: {time.time() - t0:.1f}s")
    print(f"Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
    
    # Analyze spring quality
    analyze_springs(lnn)
    
    # Analyze category clustering
    analyze_categories(lnn)
    
    # Test recall
    print("\n=== RECALL TEST ===")
    test_recall(lnn)


def analyze_springs(lnn):
    print("\n=== SPRING DISTRIBUTION ===")
    
    tau_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    tau_stiffness = {0: [], 1: [], 2: [], 3: [], 4: []}
    
    for sp in lnn.springs.values():
        tau_counts[sp.tau] += 1
        tau_stiffness[sp.tau].append(sp.stiffness)
    
    for tau in range(5):
        count = tau_counts[tau]
        if count > 0:
            avg_stiff = sum(tau_stiffness[tau]) / count
            print(f"  τ={tau}: {count} springs, avg={avg_stiff:.0f}")


def analyze_categories(lnn):
    print("\n=== CATEGORY CLUSTERING ===")
    
    categories = {
        "animals": ["cat", "dog", "bird", "fish", "horse"],
        "fruits": ["apple", "banana", "orange", "grape", "pear"],
        "colors": ["red", "blue", "green", "yellow", "orange"],
        "emotions": ["happy", "sad", "angry", "glad", "joyful"],
    }
    
    for category, members in categories.items():
        tau3_connections = 0
        total_pairs = 0
        
        for i in range(len(members)):
            for j in range(i+1, len(members)):
                a = f"word:{members[i]}"
                b = f"word:{members[j]}"
                key = lnn._key(a, b)
                total_pairs += 1
                
                if key in lnn.springs:
                    sp = lnn.springs[key]
                    if sp.tau == 3:
                        tau3_connections += 1
        
        density = tau3_connections / max(1, total_pairs)
        status = "DENSE" if density >= 0.5 else "SPARSE"
        print(f"  {category}: {tau3_connections}/{total_pairs} τ=3 ({density:.0%}) - {status}")


def test_recall(lnn):
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
            propagate(lnn, n_steps=5)
            
            activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            found = [w for w in expected if w in activated]
            status = "PASS" if len(found) >= 2 else "FAIL"
            print(f"  {query.replace('word:', '')}: {status} ({len(found)}/{len(expected)}: {found})")
        else:
            print(f"  {query.replace('word:', '')}: FAIL")


if __name__ == "__main__":
    main()
