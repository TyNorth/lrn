"""
CorpusExpander - Template-based sentence generator
Generates ~500 sentences from 15 templates without external data
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

SUBJECT_TEMPLATES = [
    "birds", "fish", "fire", "ice", "sun", "water", "moon", "stars",
    "wind", "rain", "snow", "clouds", "ocean", "flowers", "dogs", "cats",
    "trees", "rivers", "mountains", "grass", "rocks", "sand"
]

VERB_TEMPLATES = [
    "fly", "swim", "burn", "melt", "shine", "flow", "glow", "twinkle",
    "blow", "fall", "cover", "float", "crash", "bloom", "bark", "sleep",
    "build", "create", "change", "need", "like", "have", "are", "is"
]

LOCATION_TEMPLATES = [
    "in the sky", "in the river", "in the sun", "in the water",
    "at night", "in the dark", "through trees", "from clouds",
    "on shore", "in the garden", "in winter", "across the sky",
    "to the sea", "tall", "green", "hard", "at the beach"
]

MODIFIER_TEMPLATES = [
    "bright", "cold", "fast", "slow", "high", "low", "big", "small",
    "hot", "warm", "cool", "soft", "hard", "light", "dark", "loud"
]


class CorpusExpander:
    def __init__(self):
        self.core_sentences = CORE_SENTENCES
        self.negatives = NEGATIVES

    def expand(self, target_count: int = 500) -> list:
        sentences = list(self.core_sentences)
        template_id = 0

        while len(sentences) < target_count:
            template_id = (template_id + 1) % 15

            if template_id == 0:
                s = f"{SUBJECT_TEMPLATES[template_id % len(SUBJECT_TEMPLATES)]} {VERB_TEMPLATES[template_id % len(VERB_TEMPLATES)]} in the {['sky', 'river', 'ocean'][template_id % 3]}"
            elif template_id == 1:
                s = f"the {SUBJECT_TEMPLATES[template_id % len(SUBJECT_TEMPLATES)]} {VERB_TEMPLATES[template_id % len(VERB_TEMPLATES)]} {LOCATION_TEMPLATES[template_id % len(LOCATION_TEMPLATES)]}"
            elif template_id == 2:
                s = f"{SUBJECT_TEMPLATES[template_id % len(SUBJECT_TEMPLATES)]} {VERB_TEMPLATES[template_id % len(VERB_TEMPLATES)]} {MODIFIER_TEMPLATES[template_id % len(MODIFIER_TEMPLATES)]}"
            elif template_id == 3:
                s = f"{SUBJECT_TEMPLATES[template_id % len(SUBJECT_TEMPLATES)]} can {VERB_TEMPLATES[template_id % len(VERB_TEMPLATES)]}"
            elif template_id == 4:
                s = f"the {SUBJECT_TEMPLATES[template_id % len(SUBJECT_TEMPLATES)]} is {MODIFIER_TEMPLATES[template_id % len(MODIFIER_TEMPLATES)]}"
            elif template_id == 5:
                s = f"{SUBJECT_TEMPLATES[template_id % len(SUBJECT_TEMPLATES)]} and {SUBJECT_TEMPLATES[(template_id+3) % len(SUBJECT_TEMPLATES)]} are both {MODIFIER_TEMPLATES[template_id % len(MODIFIER_TEMPLATES)]}"
            elif template_id == 6:
                s = f"when {SUBJECT_TEMPLATES[template_id % len(SUBJECT_TEMPLATES)]} {VERB_TEMPLATES[template_id % len(VERB_TEMPLATES)]}, it {VERB_TEMPLATES[(template_id+1) % len(VERB_TEMPLATES)]}"
            elif template_id == 7:
                s = f"{SUBJECT_TEMPLATES[template_id % len(SUBJECT_TEMPLATES)]} {VERB_TEMPLATES[template_id % len(VERB_TEMPLATES)]} because it is {MODIFIER_TEMPLATES[template_id % len(MODIFIER_TEMPLATES)]}"
            elif template_id == 8:
                s = f"every {SUBJECT_TEMPLATES[template_id % len(SUBJECT_TEMPLATES)]} {VERB_TEMPLATES[template_id % len(VERB_TEMPLATES)]}"
            elif template_id == 9:
                s = f"many {SUBJECT_TEMPLATES[template_id % len(SUBJECT_TEMPLATES)]} {VERB_TEMPLATES[template_id % len(VERB_TEMPLATES)]} together"
            elif template_id == 10:
                s = f"the {SUBJECT_TEMPLATES[template_id % len(SUBJECT_TEMPLATES)]} always {VERB_TEMPLATES[template_id % len(VERB_TEMPLATES)]}"
            elif template_id == 11:
                s = f"some {SUBJECT_TEMPLATES[template_id % len(SUBJECT_TEMPLATES)]} {VERB_TEMPLATES[template_id % len(VERB_TEMPLATES)]} sometimes"
            elif template_id == 12:
                s = f"no {SUBJECT_TEMPLATES[template_id % len(SUBJECT_TEMPLATES)]} {VERB_TEMPLATES[template_id % len(VERB_TEMPLATES)]}"
            elif template_id == 13:
                s = f"all {SUBJECT_TEMPLATES[template_id % len(SUBJECT_TEMPLATES)]} {VERB_TEMPLATES[template_id % len(VERB_TEMPLATES)]}"
            else:
                s = f"{SUBJECT_TEMPLATES[template_id % len(SUBJECT_TEMPLATES)]} often {VERB_TEMPLATES[template_id % len(VERB_TEMPLATES)]}"

            if s not in sentences:
                sentences.append(s.lower())

        return sentences[:target_count]

    def get_all_negatives(self) -> list:
        return self.negatives