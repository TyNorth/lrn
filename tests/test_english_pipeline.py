"""
LRN English Training Pipeline
Full sequence: Sensory → Causation → Language → REM Sleep → Inference
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor
from lrn.natural_tokenize import learn_from_text, TAU_BY_TYPE
from lrn.inference import add_word_nodes, attention_with_residue
from lrn.rem_synthesis import REMSleep, TAU_PURPOSE
from lrn.sleep_cycle import full_sleep_cycle


# ============================================================================
# ENGLISH TRAINING CORPUS
# ============================================================================

SENSORY_CORPUS = [
    # Objects - visual
    "ball box cup book lamp sun moon star", 
    "cat dog bird fish horse cow sheep chicken",
    "tree flower grass leaf root branch bark",
    "rock sand water fire air earth",
    "hand foot eye ear nose mouth head",
    "chair table bed door window floor wall",
    # Colors
    "red blue green yellow orange purple white black brown pink",
    # Sizes
    "big small tiny huge giant massive minute microscopic",
    # Temperature
    "hot cold warm cool freezing boiling tepid",
    # Texture
    "hard soft smooth rough bumpy fuzzy silky",
    # States
    "wet dry clean dirty wet wet heavy light fast slow",
    # Sound
    "loud quiet noisy silent musical harsh",
    # Light
    "bright dark dim glowing shining glowing",
    # Movement
    "moving still fast slow quick rapid gradual",
]

CAUSATION_CORPUS = [
    # Direct causation
    "the ball hits the window and the window breaks",
    "the fire burns the wood and the wood turns black",
    "the water spills on the floor and the floor gets wet",
    "the wind blows the leaf and the leaf falls",
    "the sun heats the water and the water gets warm",
    "the cold freezes the water and the water becomes ice",
    "the rain soaks the ground and the ground becomes muddy",
    "the noise wakes the baby and the baby cries",
    "the light attracts the moth and the moth flies to it",
    "the food feeds the dog and the dog grows strong",
    # Actor-action-consequence
    "the child breaks the toy and the toy cannot be played with",
    "the mother feeds the baby and the baby becomes full",
    "the doctor heals the patient and the patient becomes healthy",
    "the farmer waters the plant and the plant grows",
    "the teacher teaches the student and the student learns",
    # Conditional causation
    "if you drop the glass it will break",
    "if you touch the fire it will burn",
    "if you add water it will get wet",
    "if you push the swing it will move",
    "if you turn the key the car will start",
    # Because patterns
    "because the sun shines the ground gets warm",
    "because it rains the grass grows",
    "because the dog barks the cat runs",
    "because the light is bright the room is warm",
    "because the road is wet driving is dangerous",
]

LANGUAGE_CORPUS = [
    # Basic sentences
    "the cat sits on the mat", "the dog runs in the park",
    "the bird flies over the tree", "i eat breakfast every morning",
    "she drinks coffee at noon", "he walks to school every day",
    "we play games after lunch", "they watch movies at night",
    # Questions
    "what is your name", "where do you live", "when does the train leave",
    "why did he leave so early", "how are you feeling today",
    "which book should i read", "who is coming to the party",
    # Tenses
    "i walked to the store yesterday", "she bought a new dress last week",
    "he ate breakfast at seven am", "they played soccer in the park",
    "i am reading a book right now", "she is cooking dinner in the kitchen",
    "i will go to the market tomorrow", "she will call you later today",
    # Common phrases
    "actions speak louder than words", "the early bird catches the worm",
    "a penny saved is a penny earned", "all that glitters is not gold",
    "better late than never", "every cloud has a silver lining",
    "no pain no gain", "time is of the essence", "to err is human",
]


def train_english_pipeline(repetitions: int = 30) -> dict:
    """Full English training pipeline."""
    print("=" * 60)
    print("LRN ENGLISH TRAINING PIPELINE")
    print("=" * 60)
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    stats = {"phase": [], "total_nodes": 0, "total_springs": 0}
    
    # Phase 1: Sensory (rigid - τ=0 after 5 reps)
    print("\n=== Phase 1: Sensory Grounding ===")
    for _ in range(repetitions):
        for text in SENSORY_CORPUS:
            learn_from_text(lnn, text, repetitions=1, learn_type="sensory")
    
    # Count sensory results
    sens_words = set()
    for n in lnn.nodes:
        if n.startswith("sens:"):
            sens_words.add(n)
    print(f"Sensory vocabulary: {len(sens_words)}")
    stats["phase"].append({"phase": "sensory", "vocab": len(sens_words)})
    
    # Phase 2: Causation (rigid - τ=0 after 10 reps)
    print("\n=== Phase 2: Causation Learning ===")
    for _ in range(repetitions):
        for text in CAUSATION_CORPUS:
            learn_from_text(lnn, text, repetitions=1, learn_type="causation")
    
    cauz_words = set()
    for n in lnn.nodes:
        if n.startswith("cauz:"):
            cauz_words.add(n)
    print(f"Causation vocabulary: {len(cauz_words)}")
    stats["phase"].append({"phase": "causation", "vocab": len(cauz_words)})
    
    # Phase 3: Language (flexible - τ=0 after 30 reps)
    print("\n=== Phase 3: Language Mapping ===")
    for _ in range(repetitions):
        for text in LANGUAGE_CORPUS:
            learn_from_text(lnn, text, repetitions=1, learn_type="language")
    
    # Add word-level nodes for inference
    print("\nAdding word-level nodes...")
    all_corpus = SENSORY_CORPUS + CAUSATION_CORPUS + LANGUAGE_CORPUS
    add_word_nodes(lnn, all_corpus)
    add_word_nodes(lnn, ["identity:self"])  # Add self for REM
    
    # Add identity binding
    for text in ["i am me", "i am here"]:
        for word in text.split():
            wn = f"word:{word}"
            if wn in lnn.nodes:
                lnn.add_spring(wn, "identity:self", stiffness=30, tau=1)
    
    lang_words = set()
    for n in lnn.nodes:
        if n.startswith("lang:"):
            lang_words.add(n)
    word_words = set()
    for n in lnn.nodes:
        if n.startswith("word:"):
            word_words.add(n)
    
    print(f"Language vocabulary: {len(lang_words)}")
    print(f"Word-level vocabulary: {len(word_words)}")
    stats["phase"].append({"phase": "language", "vocab": len(lang_words)})
    stats["phase"].append({"phase": "word_nodes", "vocab": len(word_words)})
    
    stats["total_nodes"] = len(lnn.nodes)
    stats["total_springs"] = len(lnn.springs)
    
    # Phase 4: Sleep (REM synthesis)
    print("\n=== Phase 4: Sleep & Consolidation ===")
    
    # Initialize REM
    rem = REMSleep(lnn)
    rem.tag_event("fire", surprise=100)
    rem.tag_event("water", surprise=80)
    rem.tag_event("learn", surprise=60)
    
    # Wake context
    wake_context = {f"word:{w}": 200 for w in ["fire", "water", "hot", "cold", "learn"]}
    
    # Run REM cycle
    rem_result = rem.run_rem_cycle(wake_context)
    print(f"REM: {rem_result['inference_count']} novel bridges, τ3={rem_result['tau3_count']}")
    
    # Full sleep cycle
    sleep_result = full_sleep_cycle(lnn)
    print(f"Sleep: {sleep_result['crystallization']}")
    
    # Final stats
    tau_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for sp in lnn.springs.values():
        tau_counts[sp.tau] = tau_counts.get(sp.tau, 0) + 1
    
    print("\n=== Final Tau Distribution ===")
    for tau, count in sorted(tau_counts.items()):
        print(f"  τ={tau} ({TAU_PURPOSE.get(tau, '')}): {count}")
    
    stats["tau_distribution"] = tau_counts
    
    return stats


def test_attention_queries(lnn: LatticeNN):
    """Test attention queries on trained network."""
    print("\n=== ATTENTION QUERY TESTS ===")
    
    test_queries = ["fire", "water", "learn", "cold", "break"]
    
    for query in test_queries:
        result = attention_with_residue(lnn, f"word:{query}", propagate_steps=3)
        
        if result["attention"]:
            print(f"\nQuery: '{query}'")
            print(f"  Activated: {result['activated_count']} nodes")
            print(f"  Top attention:")
            for node, info in result["attention"][:3]:
                print(f"    {node.replace('word:', '')}: {info['path']}")


def run_english_test_battery():
    """Run full English test battery."""
    print("=" * 60)
    print("LRN ENGLISH TEST BATTERY")
    print("=" * 60)
    
    # Train
    stats = train_english_pipeline(repetitions=30)
    
    # Get LNN for testing
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    # Retrain (simplified for test)
    for text in SENSORY_CORPUS[:20]:
        learn_from_text(lnn, text, repetitions=20, learn_type="sensory")
    for text in LANGUAGE_CORPUS[:20]:
        learn_from_text(lnn, text, repetitions=20, learn_type="language")
    add_word_nodes(lnn, SENSORY_CORPUS[:20] + LANGUAGE_CORPUS[:20])
    
    # Test attention
    test_attention_queries(lnn)
    
    # Test results
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total springs: {stats['total_springs']}")
    print(f"Phases: {len(stats['phase'])}")
    
    for phase in stats['phase']:
        print(f"  {phase['phase']}: {phase['vocab']} vocab")
    
    tau = stats['tau_distribution']
    print(f"\nTau layers:")
    print(f"  τ=0 Geometric: {tau.get(0, 0)}")
    print(f"  τ=1 Definitive: {tau.get(1, 0)}")
    print(f"  τ=2 Causal: {tau.get(2, 0)}")
    print(f"  τ=3 Categorical: {tau.get(3, 0)}")
    print(f"  τ=4 Contextual: {tau.get(4, 0)}")
    
    return stats


if __name__ == "__main__":
    run_english_test_battery()