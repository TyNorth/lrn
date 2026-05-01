"""
Test: Sensory + Causation with Tau Hierarchy
Sensory: τ=0 after 5 reps (rigid)
Causation: τ=0 after 10 reps (more rigid)
Language: τ=0 after 30 reps (flexible)
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor
from lrn.natural_tokenize import learn_from_text, discover_words


SENSORY = [
    "ball box cup book lamp sun moon",
    "cat dog bird fish horse sheep",
    "tree flower grass leaf root",
    "rock sand water fire air",
    "hand foot eye ear nose mouth",
    "red blue green yellow orange",
    "big small hot cold fast slow",
    "round square hard soft wet dry",
    "loud quiet bright dark clean dirty",
    "heavy light new old tall short",
]

CAUSATION = [
    "the ball hits the window and the window breaks",
    "the fire burns the wood and the wood turns black",
    "the water spills and the floor gets wet",
    "the wind blows and the tree shakes",
    "the sun heats the ground and the ground gets warm",
    "the cold freezes the water and the water becomes ice",
    "the rain soaks the ground and the ground becomes muddy",
    "the noise wakes the baby and the baby cries",
    "the light attracts the moth and the moth flies to it",
    "because the sun shines the ground gets warm",
    "because it rains the grass grows",
    "if you drop the glass it will break",
    "if you touch the fire it will burn",
    "the child breaks the toy and the toy cannot be played with",
    "the doctor heals the patient and the patient becomes healthy",
    "the farmer waters the plant and the plant grows",
    "the person pushes the door and the door opens",
    "the person throws the ball and the ball flies",
    "the bird eats the seed and the seed disappears",
    "the cat chases the mouse and the mouse runs away",
]


def test_tau_hierarchy():
    print("=" * 60)
    print("TAU HIERARCHY TEST")
    print("Sensory: τ=0 after 5 | Causation: τ=0 after 10 | Language: τ=0 after 30")
    print("=" * 60)
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    # Phase 1: Sensory (rigid - 5 reps)
    print("\n--- Phase 1: Sensory (rigid) ---")
    for _ in range(10):
        for text in SENSORY:
            learn_from_text(lnn, text, repetitions=1, learn_type="sensory")
    
    sens_words = discover_words(lnn, SENSORY, min_frequency=1)
    print(f"  Sensory words: {len(sens_words)}")
    
    # Count rigid springs for sensory
    rigid_sens = sum(1 for sp in lnn.springs.values() if sp.tau == 0 and sp.stiffness >= 3)
    print(f"  Rigid springs: {rigid_sens}")
    
    # Phase 2: Causation (more rigid - 10 reps)
    print("\n--- Phase 2: Causation (rigid) ---")
    for _ in range(20):
        for text in CAUSATION:
            learn_from_text(lnn, text, repetitions=1, learn_type="causation")
    
    cauz_words = discover_words(lnn, CAUSATION, min_frequency=1)
    print(f"  Causation words: {len(cauz_words)}")
    
    rigid_cauz = sum(1 for sp in lnn.springs.values() if sp.tau == 0 and sp.stiffness >= 3)
    print(f"  Rigid springs: {rigid_cauz}")
    
    # Check total
    all_words = discover_words(lnn, SENSORY + CAUSATION, min_frequency=1)
    rigid_total = sum(1 for sp in lnn.springs.values() if sp.tau == 0)
    
    print(f"\n--- Summary ---")
    print(f"  Total vocabulary: {len(all_words)}")
    print(f"  Total rigid springs: {rigid_total}")
    
    # Check different spring types
    print(f"\n--- Spring Analysis ---")
    sens_springs = [(k, v) for k, v in lnn.springs.items() if k.startswith("sens:")]
    cauz_springs = [(k, v) for k, v in lnn.springs.items() if k.startswith("cauz:")]
    print(f"  Sensory springs: {len(sens_springs)}")
    print(f"  Causation springs: {len(cauz_springs)}")
    
    # Tau values
    print(f"\n--- Tau Distribution ---")
    tau_counts = {}
    for sp in lnn.springs.values():
        tau_counts[sp.tau] = tau_counts.get(sp.tau, 0) + 1
    print(f"  τ=0 (rigid): {tau_counts.get(0, 0)}")
    print(f"  τ=4 (flexible): {tau_counts.get(4, 0)}")
    
    return {"total": len(all_words), "rigid": rigid_total, "tau": tau_counts}


def scale_test():
    """Test vocabulary scaling with tau hierarchy."""
    print("\n" + "=" * 60)
    print("SCALE TEST")
    print("=" * 60)
    
    all_corpus = SENSORY + CAUSATION
    
    for total_reps in [20, 40, 60, 80]:
        lnn = LatticeNN()
        add_identity_anchor(lnn)
        
        for rep in range(total_reps):
            for text in all_corpus:
                if rep < total_reps // 2:
                    learn_from_text(lnn, text, repetitions=1, learn_type="sensory")
                else:
                    learn_from_text(lnn, text, repetitions=1, learn_type="causation")
        
        words = discover_words(lnn, all_corpus, min_frequency=1)
        rigid = sum(1 for sp in lnn.springs.values() if sp.tau == 0)
        
        print(f"  {total_reps} reps: {len(words)} words, {rigid} rigid springs")


if __name__ == "__main__":
    result = test_tau_hierarchy()
    scale_test()
    print(f"\nResult: {result['total']} words, {result['rigid']} rigid springs")