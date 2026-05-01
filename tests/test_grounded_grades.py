"""
Scale: Sensory Grounding → Language → Grade Targets
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor
from lrn.natural_tokenize import learn_from_text, discover_words


# Expanded sensory + language sources
PHYSICAL_SOURCES = {
    "objects": "round square triangle circle oval rectangle big small tiny huge massive red blue green yellow orange purple pink black white brown shiny dull smooth rough hard soft chair table bed door window floor ceiling wall roof cup plate bowl spoon fork knife pot pan book pen paper pencil computer phone clock light lamp candle sun moon star head face eye nose mouth ear hand arm leg foot hair skin bone blood heart brain".split(),
    "actions": "move stop start push pull lift drop throw catch walk run jump hop skip slide roll sit stand lie bend twist stretch look see watch observe notice find search listen hear sound touch feel grab hold squeeze pinch taste smell eat drink lick sniff breathe cough sneeze yawn blink open close turn spin rise fall".split(),
    "senses": "bright dark light shadow color shape size near far close distant high low straight curved wavy loud quiet soft noisy silent hot cold warm cool wet dry damp heavy light rough smooth bumpy sweet sour bitter salty spicy fresh stinky".split(),
    "states": "full empty open closed on off clean dirty wet dry new old broken fixed whole divided hot cold warm cool moving still fast slow here there inside outside above below in out up down forward back near far middle center edge left right".split(),
    "relationships": "big bigger biggest small smaller smallest same different similar like unlike before after next to between among fast slow quick rapid early late always never sometimes often half whole part piece one two three four five six seven eight nine ten many few some all none more less most least".split(),
}

LANGUAGE_SOURCES = {
    "nouns": [
        "the ball is round", "the box is square", "the sky is blue",
        "the fire is hot", "the ice is cold", "the rock is hard",
        "the pillow is soft", "the water is wet", "the sand is dry",
        "the cat is small", "the elephant is big", "the bug is tiny",
        "the tree is tall", "the flower is pretty", "the bird can fly",
        "the fish swims", "the dog runs", "the baby cries",
    ],
    "verbs": [
        "i walk to the door", "i run to the car", "i jump over the fence",
        "i look at the bird", "i listen to the music", "i touch the surface",
        "i taste the food", "i smell the flower", "i breathe the air",
        "i open the door", "i close the window", "i turn on the light",
        "i eat breakfast", "i drink water", "i sleep at night",
        "i read a book", "i write a letter", "i draw a picture",
    ],
    "adjectives": [
        "the apple is red", "the sky is blue", "the grass is green",
        "the coffee is hot", "the ice is cold", "the bed is soft",
        "the rock is hard", "the sound is loud", "the room is quiet",
        "the book is big", "the ant is small", "the sun is bright",
        "the water is clean", "the food is fresh", "the day is long",
    ],
    "prepositions": [
        "the cat is on the mat", "the book is under the bed",
        "the bird is above the tree", "the fish is below the surface",
        "the car is in front of the house", "the tree is behind the shed",
        "the park is near the school", "the store is far from here",
        "the ball is between the chairs", "the house is next to the road",
    ],
    "questions": [
        "what color is the apple", "where is the cat", "how big is the house",
        "why is the fire hot", "when does the sun rise", "who is at the door",
        "what is that thing", "where does it come from", "how does it work",
    ],
    "time": [
        "the sun rises in the morning", "the sun sets at night",
        "i eat breakfast in the morning", "i sleep at night",
        "yesterday was monday", "today is tuesday", "tomorrow is wednesday",
    ],
    "family": [
        "my mother is kind", "my father is strong", "my sister is smart",
        "my brother is tall", "my grandma is sweet", "my grandpa is wise",
    ],
    "nature": [
        "the sun rises", "the moon shines", "the stars twinkle",
        "the river flows", "the ocean is big", "the mountain is high",
        "the tree grows", "the flower blooms", "the wind blows",
    ],
}


def train_grounded_language(target_words: int, max_exposures: int = 20000) -> dict:
    """Train sensory-grounded language to target."""
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    # Build flat lists
    physical_samples = []
    for category, words in PHYSICAL_SOURCES.items():
        for word in words:
            physical_samples.append(word)
    
    language_samples = []
    for category, texts in LANGUAGE_SOURCES.items():
        language_samples.extend(texts)
    
    all_samples = physical_samples + language_samples
    
    words = 0
    exposures = 0
    batch_size = 100
    
    while words < target_words and exposures < max_exposures:
        for _ in range(10):
            for text in all_samples[:batch_size]:
                learn_from_text(lnn, text, repetitions=1)
        
        exposures += 10 * min(batch_size, len(all_samples))
        
        # Check both physical and language words
        phys = len(discover_words(lnn, physical_samples[:50], min_frequency=1))
        lang = len(discover_words(lnn, language_samples[:50], min_frequency=1))
        
        words = len(discover_words(lnn, all_samples[:batch_size], min_frequency=1))
        
        if exposures % 2000 == 0:
            print(f"  {exposures}: {words} words (physical: {phys}, language: {lang})")
    
    return {"words": words, "exposures": exposures, "physical": phys, "language": lang}


def main():
    print("=" * 60)
    print("SENSORY-GROUNDED LANGUAGE → GRADE TARGETS")
    print("=" * 60)
    
    targets = [(250, "grade1"), (500, "grade2"), (1000, "grade3")]
    results = {}
    
    for target, grade in targets:
        print(f"\n--- Target: {target} ({grade}) ---")
        result = train_grounded_language(target)
        results[grade] = result
        
        status = "PASS" if result["words"] >= target else "FAIL"
        print(f"Result: {result['words']}/{target} - {status}")
        
        if result["words"] < target:
            break
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for grade, result in results.items():
        print(f"  {grade}: {result['words']} words (phys: {result['physical']}, lang: {result['language']})")


if __name__ == "__main__":
    main()