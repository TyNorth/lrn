"""
Sensory Grounding + Causation + Language
Phase 1: Physical concepts (objects, actions, senses)
Phase 2: Actor → Action → Consequence (causation)
Phase 3: Language mapping
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor
from lrn.natural_tokenize import learn_from_text, discover_words


# ============================================================================
# PHASE 1: SENSORY GROUNDING - Physical world
# ============================================================================

SENSORY_CORPUS = [
    # Objects
    "ball box cup book pen lamp sun moon star",
    "cat dog bird fish horse cow sheep chicken",
    "tree flower grass leaf root branch bark",
    "rock sand water fire air earth",
    "hand foot eye ear nose mouth head",
    "chair table bed door window floor wall",
    "red blue green yellow orange purple white black",
    "big small hot cold fast slow heavy light",
    "round square hard soft smooth rough wet dry",
    "loud quiet bright dark clean dirty new old",
    
    # Actions
    "move stop start push pull drop throw catch",
    "walk run jump climb slide roll spin rise fall",
    "look see watch find search hear listen smell taste",
    "open close turn push pull lift carry hold",
    "eat drink sleep wake sit stand lie bend",
    "break fix build destroy make create change",
    "hit kick punch bite scratch push shove",
    "write read draw paint cut glue attach separate",
]

# ============================================================================
# PHASE 2: CAUSATION - Actor → Action → Consequence
# ============================================================================

CAUSATION_PATTERNS = [
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
    
    # Actor causes effect
    "the child breaks the toy and the toy cannot be played with",
    "the mother feeds the baby and the baby becomes full",
    "the teacher teaches the student and the student learns",
    "the doctor heals the patient and the patient becomes healthy",
    "the farmer waters the plant and the plant grows",
    "the rain stops and the sun comes out",
    "the clock rings and the people wake up",
    "the car crashes and the car gets damaged",
    "the wind blows and the tree shakes",
    "the fire spreads and the house burns",
    
    # Action sequences
    "the person pushes the door and the door opens",
    "the person pulls the rope and the box moves",
    "the person throws the ball and the ball flies",
    "the person catches the ball and the ball stops",
    "the person mixes the colors and new colors appear",
    "the person breaks the stick and the stick splits",
    "the person connects the wires and the light turns on",
    "the person locks the door and the door stays closed",
    "the person fills the cup and the cup becomes full",
    "the person empties the bag and the bag becomes empty",
    
    # Consequence chains
    "the bird eats the seed and the seed disappears",
    "the cat chases the mouse and the mouse runs away",
    "the dog hears the sound and the dog barks",
    "the baby feels hungry and the baby cries",
    "the person feels tired and the person sleeps",
    "the person feels happy and the person smiles",
    "the person feels sad and the person cries",
    "the sky gets dark and the stars appear",
    "the ice gets warm and the ice melts",
    "the wind gets strong and the tree falls",
    
    # Cause and effect patterns
    "because the sun shines the ground gets warm",
    "because it rains the grass grows",
    "because the dog barks the cat runs",
    "because the light is bright the room is warm",
    "because the road is wet driving is dangerous",
    "because the food is hot it burns the tongue",
    "because the glass is fragile it breaks easily",
    "because the bird sings the morning is beautiful",
    "because the water flows the river gets deeper",
    "because the snow falls the ground becomes white",
    
    # If-then causation
    "if you drop the glass it will break",
    "if you touch the fire it will burn",
    "if you add water it will get wet",
    "if you push the swing it will move",
    "if you turn the key the car will start",
    "if you press the button the light will turn on",
    "if you open the door the room will get air",
    "if you close the window the sound will stop",
    "if you feed the animal it will be happy",
    "if you practice every day you will improve",
    
    # Result patterns
    "the rain made the ground wet",
    "the sun made the day warm",
    "the noise made everyone jump",
    "the fall made the bone break",
    "the heat made the ice melt",
    "the wind made the tree shake",
    "the sound made the dog alert",
    "the food made the person strong",
    "the lesson made the student smart",
    "the exercise made the body fit",
]

# ============================================================================
# PHASE 3: LANGUAGE (for comparison)
# ============================================================================

LANGUAGE_CORPUS = [
    "the ball is round", "the sky is blue", "the fire is hot",
    "the cat sits on the mat", "the dog runs in the park",
    "i eat breakfast", "she drinks water", "he walks to school",
    "what is your name", "where do you live", "when does it start",
    "the big dog barks", "the small cat sleeps", "the red apple is sweet",
]


def phase1_sensory(reps: int) -> int:
    """Phase 1: Physical concepts."""
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    for _ in range(reps):
        for text in SENSORY_CORPUS:
            learn_from_text(lnn, text, repetitions=1)
    
    words = discover_words(lnn, SENSORY_CORPUS, min_frequency=1)
    print(f"Phase 1 (Sensory): {len(words)} concepts")
    return len(words)


def phase2_causation(sensory_count: int, reps: int) -> dict:
    """Phase 2: Causation patterns."""
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    # First train sensory
    for _ in range(reps):
        for text in SENSORY_CORPUS:
            learn_from_text(lnn, text, repetitions=1)
    
    # Then train causation
    for _ in range(reps):
        for text in CAUSATION_PATTERNS:
            learn_from_text(lnn, text, repetitions=1)
    
    all_samples = SENSORY_CORPUS + CAUSATION_PATTERNS
    words = discover_words(lnn, all_samples, min_frequency=1)
    
    # Check causation-specific words
    causation_words = discover_words(lnn, CAUSATION_PATTERNS, min_frequency=1)
    
    # Check 3-word patterns (actor-action-consequence)
    patterns = {}
    for text in CAUSATION_PATTERNS:
        tokens = text.split()
        for i in range(len(tokens) - 2):
            pattern = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
            patterns[pattern] = patterns.get(pattern, 0) + 1
    
    # Filter for patterns that appear multiple times (learned causation)
    learned_patterns = {p: c for p, c in patterns.items() if c >= 3}
    
    return {
        "total": len(words),
        "causation_specific": len(causation_words),
        "learned_patterns": len(learned_patterns),
    }


def phase3_full(sensory_reps: int, causation_reps: int, language_reps: int) -> dict:
    """All phases combined."""
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    # Phase 1: Sensory
    for _ in range(sensory_reps):
        for text in SENSORY_CORPUS:
            learn_from_text(lnn, text, repetitions=1)
    
    # Phase 2: Causation
    for _ in range(causation_reps):
        for text in CAUSATION_PATTERNS:
            learn_from_text(lnn, text, repetitions=1)
    
    # Phase 3: Language
    for _ in range(language_reps):
        for text in LANGUAGE_CORPUS:
            learn_from_text(lnn, text, repetitions=1)
    
    all_samples = SENSORY_CORPUS + CAUSATION_PATTERNS + LANGUAGE_CORPUS
    words = discover_words(lnn, all_samples, min_frequency=1)
    
    return {"total": len(words)}


def main():
    print("=" * 60)
    print("SENSORY + CAUSATION + LANGUAGE")
    print("=" * 60)
    
    print("\n--- Testing different configurations ---")
    
    for reps in [30, 60]:
        print(f"\n=== {reps} repetitions ===")
        
        result = phase2_causation(sensory_count=0, reps=reps)
        print(f"  Total: {result['total']} words")
        print(f"  Causation-specific: {result['causation_specific']}")
        print(f"  Learned patterns: {result['learned_patterns']}")
    
    print("\n--- Full pipeline test ---")
    full = phase3_full(sensory_reps=60, causation_reps=60, language_reps=30)
    print(f"Total vocabulary: {full['total']}")
    
    print("\n--- Scale to grade targets ---")
    # Try with more repetitions
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    all_corpus = SENSORY_CORPUS + CAUSATION_PATTERNS + LANGUAGE_CORPUS
    
    for reps in [50, 100, 150, 200]:
        for _ in range(reps):
            for text in all_corpus[:50]:
                learn_from_text(lnn, text, repetitions=1)
        
        words = discover_words(lnn, all_corpus[:50], min_frequency=1)
        print(f"  {reps*50} exposures: {len(words)} words")


if __name__ == "__main__":
    main()