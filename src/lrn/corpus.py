"""
CorpusExpander - Template-based sentence generator
Generates ~500 sentences from templates without external data
"""

CORE_SENTENCES = [
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
]


TEMPLATES = [
    "{subject} {verb} in the {place}",
    "the {subject} {verb} {location}",
    "{subject} {verb} {modifier}",
    "{subject} can {verb}",
    "the {subject} is {modifier}",
    "{subject} and {subject2} are both {modifier}",
    "when {subject} {verb}, it {verb2}",
    "{subject} {verb} because it is {modifier}",
    "every {subject} {verb}",
    "many {subject}s {verb} together",
    "the {subject} always {verb}",
    "some {subject}s {verb} sometimes",
    "no {subject} {verb}",
    "all {subject}s {verb}",
    "{subject} often {verb}",
]

SUBJECTS = ["bird", "fish", "fire", "ice", "sun", "water", "moon", "star",
            "wind", "rain", "snow", "cloud", "ocean", "flower", "dog", "cat",
            "tree", "river", "mountain", "grass", "rock", "sand"]

VERBS = ["fly", "swim", "burn", "melt", "shine", "flow", "glow", "twinkle",
         "blow", "fall", "cover", "float", "crash", "bloom", "bark", "sleep",
         "build", "create", "change", "need", "like", "have", "are"]

PLACES = ["sky", "river", "ocean", "forest", "mountain", "garden"]

LOCATIONS = ["in the sky", "in the river", "at night", "in the dark",
             "through trees", "from clouds", "on shore", "in the garden"]

MODIFIERS = ["bright", "cold", "fast", "slow", "hot", "warm", "cool",
             "soft", "hard", "light", "dark", "loud", "big", "small"]


class CorpusExpander:
    def __init__(self):
        self.core_sentences = CORE_SENTENCES
        self.negatives = NEGATIVES

    def expand(self, target_count: int = 500) -> list:
        sentences = list(self.core_sentences)
        
        idx = 0
        while len(sentences) < target_count:
            template = TEMPLATES[idx % len(TEMPLATES)]
            subj = SUBJECTS[idx % len(SUBJECTS)]
            verb = VERBS[idx % len(VERBS)]
            place = PLACES[idx % len(PLACES)]
            loc = LOCATIONS[idx % len(LOCATIONS)]
            mod = MODIFIERS[idx % len(MODIFIERS)]
            subj2 = SUBJECTS[(idx + 7) % len(SUBJECTS)]
            verb2 = VERBS[(idx + 1) % len(VERBS)]
            
            s = template.format(
                subject=subj,
                subject2=subj2,
                verb=verb,
                verb2=verb2,
                place=place,
                location=loc,
                modifier=mod
            )
            
            sentences.append(s)
            idx += 1
        
        return sentences[:target_count]

    def get_all_negatives(self) -> list:
        return self.negatives