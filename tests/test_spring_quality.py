"""
ARCHITECTURAL INSIGHT: The LRN stores knowledge in SPRINGS, not activations.

Activations are transient working memory (like neural firing).
Springs are long-term memory (like synaptic strength).

RETENTION = Can we re-activate the pattern by querying a related node?
Not: Does activation persist forever?

METRICS THAT MATTER:
1. Spring density: How many τ=3 categorical bridges exist?
2. Spring strength: What's the average stiffness of τ=3 bridges?
3. Category clustering: Do category members have τ=3 bridges to each other?
4. Cross-domain bridges: Do related concepts across domains have springs?
5. Negative springs: Are wrong patterns inhibited?

OPTIMAL LEARNING CONDITIONS:
1. REM synthesis after every meaningful chunk
2. τ=3 bridges form between co-occurring words
3. Hebbian learning strengthens frequently used paths
4. Progressive complexity (scaffolding)
5. Interleaved practice (mix old and new)
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
    print("=" * 60)
    print("SPRING QUALITY ANALYSIS")
    print("=" * 60)
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    print("\nTraining...")
    t0 = time.time()
    train_enhanced(lnn, CORPUS, reps=30, learn_type="language")
    
    print(f"Training time: {time.time() - t0:.1f}s")
    print(f"Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
    
    # Analyze spring quality
    analyze_springs(lnn)
    
    # Analyze category clustering
    analyze_categories(lnn)
    
    # Analyze cross-domain bridges
    analyze_cross_domain(lnn)


def analyze_springs(lnn):
    """Analyze spring distribution by tau level."""
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
            max_stiff = max(tau_stiffness[tau])
            min_stiff = min(tau_stiffness[tau])
            print(f"  τ={tau}: {count} springs, avg={avg_stiff:.0f}, min={min_stiff}, max={max_stiff}")
        else:
            print(f"  τ={tau}: 0 springs")


def analyze_categories(lnn):
    """Analyze if category members have τ=3 bridges to each other."""
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
        print(f"  {category}: {tau3_connections}/{total_pairs} τ=3 bridges ({density:.0%}) - {status}")


def analyze_cross_domain(lnn):
    """Analyze cross-domain bridges."""
    print("\n=== CROSS-DOMAIN BRIDGES ===")
    
    cross_tests = [
        ("cat", "runs"),  # animal → action
        ("cat", "eats"),  # animal → action
        ("apple", "red"),  # fruit → color
        ("happy", "smile"),  # emotion → action
        ("dog", "fast"),  # animal → adjective
    ]
    
    for a, b in cross_tests:
        key = lnn._key(f"word:{a}", f"word:{b}")
        if key in lnn.springs:
            sp = lnn.springs[key]
            print(f"  {a} ↔ {b}: τ={sp.tau}, stiffness={sp.stiffness}")
        else:
            print(f"  {a} ↔ {b}: NO SPRING")


if __name__ == "__main__":
    main()
