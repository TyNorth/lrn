"""
Extended CorpusExpander - Larger, more diverse training corpus
"""
import random

CORE_SENTENCES = [
    # Basic physics
    "birds fly in the sky",
    "fish swim in the river",
    "fire burns hot and bright",
    "ice melts in the sun",
    "the sun shines in the sky",
    "water flows down from mountains",
    "the moon glows at night",
    "stars twinkle in the dark",
    "wind blows through the trees",
    "rain falls from clouds",
    "snow covers the ground in winter",
    "clouds float across the sky",
    "the ocean waves crash on shore",
    "flowers bloom in the garden",
    "dogs bark at strangers",
    "cats sleep on soft beds",
    "birds build nests in trees",
    "fish swim in schools",
    "fire creates smoke",
    "ice is cold to touch",
    "the sun gives us light",
    "water quenches thirst",
    "the moon changes shape",
    "wind can be strong",
    "rain makes puddles",
    "snow is white",
    "clouds block the sun",
    "the ocean is vast",
    "flowers need sunlight",
    "dogs are loyal pets",
    "cats like to play",
    "trees have roots",
    "rivers flow to the sea",
    "mountains are tall",
    "the sky is blue",
    "grass is green",
    "rocks are hard",
    "sand is at the beach",
    
    # Causality (τ=2 springs)
    "fire causes burning",
    "ice causes melting",
    "rain causes wetness",
    "sun causes warmth",
    "wind causes movement",
    "food causes hunger",
    "water causes hydration",
    "sleep causes rest",
    "medicine causes healing",
    "exercise causes strength",
    
    # Categorical (τ=3 springs)
    "fish and birds are animals",
    "hot and cold are temperatures",
    "day and night are times",
    "sky and ground are places",
    "sun and moon are celestial",
    "mountain and valley are terrain",
    "river and ocean are water",
    "tree and flower are plants",
    "dog and cat are pets",
    "rain and snow are precipitation",
]

NEGATIVES = [
    "fish don't fly",
    "birds don't swim",
    "fire don't freeze",
    "ice don't burn",
    "sun don't go dark",
    "water don't flow up",
    "moon don't shine day",
    "stars don't fall",
    "wind don't stay still",
    "rain don't fall up",
    "fire don't cool",
    "ice don't heat",
    "rocks don't float",
    "fish don't walk",
    "birds don't swim",
]


EXPANDED_TEMPLATES = [
    # Simple continuation
    ("the {noun} {verb}", ["sky", "ground", "water", "air", "sun"]),
    # Causality
    ("{noun1} causes {noun2}", ["fire/burns", "ice/melts", "rain/wetness", "sun/light", "wind/motion"]),
    # Categorical
    ("{noun1} and {noun2} are both {adj}", ["hot/cold/temperatures", "fast/slow/speeds", "big/small/sizes"]),
    # Temporal
    ("when {noun} {verb}, it {verb2}", ["sun/shines/it/warms", "rain/falls/it/is/wet", "wind/blows/it/moves"]),
    # Physical properties
    ("the {noun} is {adj}", ["fire/hot", "ice/cold", "water/wet", "rock/hard", "cloud/white"]),
    # Actions
    ("{noun} can {verb}", ["bird/fly", "fish/swim", "dog/run", "cat/sleep", "sun/shine"]),
    # Comparison
    ("{noun1} is like {noun2}", ["sun/star", "moon/star", "river/ocean", "cloud/sky"]),
    # Location
    ("{noun} lives in {place}", ["fish/water", "bird/sky", "dog/house", "tree/forest"]),
    # Purpose
    ("the {noun} is for {noun2}", ["food/eating", "water/drinking", "bed/sleeping", "car/driving"]),
    # Composition
    ("{noun} is made of {noun2}", "water/ice,rivers/water"),
]

NOUNS = [
    "bird", "fish", "dog", "cat", "tree", "flower", "sun", "moon", "star", "cloud",
    "water", "fire", "ice", "wind", "rain", "snow", "rock", "sand", "grass", "leaf",
    "mountain", "river", "ocean", "sky", "ground", "house", "car", "book", "food", "chair"
]

VERBS = [
    "fly", "swim", "run", "walk", "jump", "sleep", "eat", "drink", "play", "work",
    "shine", "glow", "burn", "melt", "freeze", "flow", "fall", "rise", "blow", "float",
    "grow", "change", "move", "stop", "start", "break", "fix", "build", "destroy", "create"
]

ADJECTIVES = [
    "hot", "cold", "warm", "cool", "big", "small", "fast", "slow", "bright", "dark",
    "hard", "soft", "wet", "dry", "heavy", "light", "loud", "quiet", "high", "low",
    "good", "bad", "new", "old", "young", "strong", "weak", "happy", "sad", "angry"
]

PLACES = [
    "sky", "water", "ground", "forest", "mountain", "ocean", "river", "desert", "jungle", "city",
    "house", "room", "garden", "park", "beach", "cave", "tree", "cloud", "space", "sea"
]


class ExtendedCorpusExpander:
    def __init__(self):
        self.core_sentences = CORE_SENTENCES
        self.negatives = NEGATIVES
        
    def expand(self, target_count: int = 2000) -> list:
        sentences = list(self.core_sentences)
        
        # Generate variations
        for _ in range(target_count - len(sentences)):
            noun = random.choice(NOUNS)
            verb = random.choice(VERBS)
            adj = random.choice(ADJECTIVES)
            place = random.choice(PLACES)
            
            template_type = random.randint(0, 9)
            
            if template_type == 0:
                s = f"the {noun} {verb}"
            elif template_type == 1:
                s = f"{noun} {verb} in the {place}"
            elif template_type == 2:
                s = f"the {noun} is {adj}"
            elif template_type == 3:
                s = f"{noun} can {verb}"
            elif template_type == 4:
                s = f"when {noun} {verb}, it {random.choice(VERBS)}s"
            elif template_type == 5:
                noun2 = random.choice(NOUNS)
                s = f"{noun} and {noun2} are both {adj}"
            elif template_type == 6:
                s = f"the {noun} {verb}s {adj}ly"
            elif template_type == 7:
                s = f"{noun} {verb}s to the {place}"
            elif template_type == 8:
                s = f"every {noun} {verb}s"
            else:
                s = f"the {noun} always {verb}s"
            
            if s.lower() not in [x.lower() for x in sentences]:
                sentences.append(s.lower())
        
        return sentences[:target_count]

    def get_all_negatives(self) -> list:
        return self.negatives