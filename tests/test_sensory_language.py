"""
Sensory Grounding First - Then Language
Like a child: learns physical world first, then maps words to concepts
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor
from lrn.natural_tokenize import learn_from_text, discover_words


# ============================================================================
# PHASE 1: SENSORY GROUNDING - Physical world concepts (no language yet)
# ============================================================================

PHYSICAL_CONCEPTS = {
    "objects": [
        # Visual - shapes
        "round square triangle circle oval rectangle",
        "big small tiny huge giant massive",
        "red blue green yellow orange purple pink black white brown",
        "shiny dull smooth rough hard soft",
        # Household
        "chair table bed door window floor ceiling wall roof",
        "cup plate bowl spoon fork knife pot pan",
        "book pen paper pencil computer phone clock",
        "light lamp candle sun moon star",
        # Body parts
        "head face eye nose mouth ear hand arm leg foot",
        "hair skin bone blood heart brain",
    ],
    "actions": [
        # Physical movements
        "move stop start push pull lift drop throw catch",
        "walk run jump hop skip slide roll",
        "sit stand lie bend twist stretch",
        "open close turn spin rise fall",
        # Sensory actions
        "look see watch observe notice find search",
        "listen hear sound listen to",
        "touch feel grab hold squeeze pinch",
        "taste smell eat drink lick sniff",
        "breathe cough sneeze yawn blink",
    ],
    "senses": [
        # Visual
        "bright dark light shadow color shape size",
        "near far close distant high low",
        "straight curved wavy zig zag",
        # Sound
        "loud quiet soft noisy silent",
        "high low fast slow节奏",
        "musical harsh smooth rough",
        # Touch
        "hot cold warm cool",
        "wet dry damp moist",
        "heavy light weight",
        "rough smooth bumpy flat",
        # Taste
        "sweet sour bitter salty spicy",
        "delicious tasty yummy bland",
        # Smell
        "sweet fragrant aromatic stinky",
        "fresh stale pleasant unpleasant",
    ],
    "states": [
        # Object states
        "full empty open closed on off",
        "clean dirty wet dry new old",
        "broken fixed whole divided",
        "hot cold warm cool",
        "moving still fast slow",
        # Location states
        "here there inside outside above below",
        "in out up down forward back",
        "near far middle center edge",
        "left right front back",
    ],
    "relationships": [
        # Spatial
        "big bigger biggest small smaller smallest",
        "same different similar like unlike",
        "before after next to between among",
        "above below under over through",
        # Temporal
        "now then soon later before after",
        "fast slow quick rapid gradual",
        "early late on time",
        "always never sometimes often rarely",
        # Quantity
        "one two three four five six seven eight nine ten",
        "many few some all none",
        "more less most least",
        "half whole part piece",
    ],
}


def phase1_sensory_grounding(repetitions: int = 50) -> dict:
    """Phase 1: Learn physical world - no language mapping yet."""
    print("=" * 60)
    print("PHASE 1: SENSORY GROUNDING")
    print("Learning physical world first - objects, actions, senses")
    print("=" * 60)
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    all_physical = []
    for category, texts in PHYSICAL_CONCEPTS.items():
        all_physical.extend(texts)
    
    for _ in range(repetitions):
        for text in all_physical:
            learn_from_text(lnn, text, repetitions=1)
    
    concepts = discover_words(lnn, all_physical, min_frequency=1)
    
    print(f"Physical concepts learned: {len(concepts)}")
    print(f"Categories: {list(PHYSICAL_CONCEPTS.keys())}")
    
    # Category breakdown
    for category, texts in PHYSICAL_CONCEPTS.items():
        cat_words = discover_words(lnn, texts, min_frequency=1)
        print(f"  {category}: {len(cat_words)} concepts")
    
    return {"concepts": len(concepts), "vocabulary": list(concepts.keys())[:30]}


# ============================================================================
# PHASE 2: LANGUAGE MAPPING - Connect words to physical concepts
# ============================================================================

LANGUAGE_MAPPINGS = {
    "nouns": [
        # Map word to physical concept
        "the ball is round", "the box is square", "the sky is blue",
        "the fire is hot", "the ice is cold", "the rock is hard",
        "the pillow is soft", "the water is wet", "the sand is dry",
        "the cat is small", "the elephant is big", "the bug is tiny",
    ],
    "verbs": [
        # Map action words
        "i walk to the door", "i run to the car", "i jump over the fence",
        "i look at the bird", "i listen to the music", "i touch the surface",
        "i taste the food", "i smell the flower", "i breathe the air",
        "i open the door", "i close the window", "i turn on the light",
    ],
    "adjectives": [
        # Map descriptive words
        "the apple is red", "the sky is blue", "the grass is green",
        "the coffee is hot", "the ice is cold", "the bed is soft",
        "the rock is hard", "the sound is loud", "the room is quiet",
        "the book is big", "the ant is small", "the sun is bright",
    ],
    "prepositions": [
        # Map spatial words to physical locations
        "the cat is on the mat", "the book is under the bed",
        "the bird is above the tree", "the fish is below the surface",
        "the car is in front of the house", "the tree is behind the shed",
        "the park is near the school", "the store is far from here",
        "the ball is between the chairs", "the house is next to the road",
    ],
    "time_words": [
        # Map temporal words
        "the sun rises in the morning", "the sun sets at night",
        "i eat breakfast in the morning", "i sleep at night",
        "yesterday was monday", "today is tuesday", "tomorrow is wednesday",
        "spring comes after winter", "summer is the hottest season",
    ],
    "questions_grounded": [
        # Questions about physical world
        "what color is the apple", "where is the cat",
        "how big is the house", "why is the fire hot",
        "when does the sun rise", "who is at the door",
    ],
}


def phase2_language_mapping(sensory_vocabulary: list, repetitions: int = 50) -> dict:
    """Phase 2: Connect language to physical concepts."""
    print("\n" + "=" * 60)
    print("PHASE 2: LANGUAGE MAPPING")
    print("Connecting words to physical concepts")
    print("=" * 60)
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    # First, train on physical concepts (sensory grounding)
    all_physical = []
    for texts in PHYSICAL_CONCEPTS.values():
        all_physical.extend(texts)
    
    for _ in range(repetitions):
        for text in all_physical:
            learn_from_text(lnn, text, repetitions=1)
    
    # Then, map language to physical concepts
    all_language = []
    for texts in LANGUAGE_MAPPINGS.values():
        all_language.extend(texts)
    
    for _ in range(repetitions):
        for text in all_language:
            learn_from_text(lnn, text, repetitions=1)
    
    # Discover words from both phases
    all_samples = all_physical + all_language
    words = discover_words(lnn, all_samples, min_frequency=1)
    
    print(f"Total vocabulary (sensory + language): {len(words)}")
    
    # Check language-specific words
    lang_words = discover_words(lnn, all_language, min_frequency=1)
    print(f"Language words: {len(lang_words)}")
    
    # Physical words after mapping
    phys_words = discover_words(lnn, all_physical, min_frequency=1)
    print(f"Physical concepts: {len(phys_words)}")
    
    return {"total": len(words), "language": len(lang_words), "physical": len(phys_words)}


# ============================================================================
# COMPREHENSIVE TEST
# ============================================================================

def main():
    print("=" * 60)
    print("SENSORY GROUNDING → LANGUAGE LEARNING")
    print("Like a child: world first, words second")
    print("=" * 60)
    
    # Phase 1: Sensory grounding
    phase1 = phase1_sensory_grounding(repetitions=60)
    
    # Phase 2: Language mapping
    phase2 = phase2_language_mapping(phase1["vocabulary"], repetitions=60)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Phase 1 (Sensory): {phase1['concepts']} physical concepts")
    print(f"Phase 2 (Language): {phase2['total']} total words")
    print(f"  - Language-specific: {phase2['language']}")
    print(f"  - Physical concepts: {phase2['physical']}")
    
    return {"phase1": phase1, "phase2": phase2}


if __name__ == "__main__":
    main()