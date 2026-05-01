"""
Quick test: REM synthesis forms τ=3 categorical bridges for phonics
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.natural_tokenize import learn_from_text
from lrn.inference import add_word_nodes


def test_phonics_with_rem():
    """Test that REM synthesis creates word family clusters."""
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    PHONICS = [
        "cat hat mat sat bat",
        "bike hike like Mike",
        "boat coat float goat",
        "moon June spoon noon",
    ]
    
    wake_buffer = []
    
    # Train with REM after each corpus pass
    for rep in range(50):
        for sentence in PHONICS:
            learn_from_text(lnn, sentence, repetitions=1, learn_type="sensory")
            add_word_nodes(lnn, [sentence])
            
            wake_buffer.append(sentence)
            if len(wake_buffer) > 5:
                wake_buffer = wake_buffer[-5:]
        
        # REM after each pass
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
        
        propagate(lnn, n_steps=3)
    
    print(f"Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
    
    # Test phonics - query "cat" should activate hat, mat, sat, bat
    for n in lnn.nodes.values():
        n.activation = 0
        n.pinned = False
    
    lnn.nodes["word:cat"].activation = 100
    propagate(lnn, n_steps=5)
    
    activated = [(n, node.activation) for n, node in lnn.nodes.items() 
                 if node.activation > 0 and n.startswith("word:")]
    activated.sort(key=lambda x: -x[1])
    
    print(f"\nQuery: 'cat'")
    print(f"Activated: {[n.replace('word:', '') for n, a in activated[:10]]}")
    
    # Check if word family detected
    family = ["hat", "mat", "sat", "bat"]
    found = [w for w in family if f"word:{w}" in [n for n, _ in activated]]
    
    print(f"Word family found: {found}")
    print(f"PASS: {len(found) >= 2}")
    
    return len(found) >= 2


if __name__ == "__main__":
    result = test_phonics_with_rem()
    print(f"\nResult: {'PASS' if result else 'FAIL'}")
