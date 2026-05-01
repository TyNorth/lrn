"""
Expanded English Corpus - 5000+ unique words for grade targets
"""
import random

# Base sentences
BASE = [
    "the cat sits on the mat",
    "the dog runs in the park",
    "the bird flies over the tree",
    "i eat breakfast every morning",
    "she drinks coffee at noon",
    "he walks to school every day",
    "we play games after lunch",
    "they watch movies at night",
]

# Expand to thousands of unique sentences
def generate_sentences():
    subjects = ["i", "you", "he", "she", "it", "we", "they", "the cat", "the dog", "the bird", 
               "the child", "the man", "the woman", "my mother", "my father", "your friend",
               "our team", "their boss", "a student", "the teacher", "the doctor"]
    
    verbs = ["eat", "drink", "run", "walk", "sit", "stand", "sleep", "wake", "think", "know",
             "see", "hear", "feel", "smell", "touch", "look", "watch", "listen", "speak", "talk",
             "read", "write", "draw", "paint", "sing", "dance", "play", "work", "study", "rest",
             "move", "stop", "start", "finish", "begin", "continue", "love", "hate", "want", "need",
             "like", "dislike", "fear", "hope", "wish", "believe", "trust", "doubt", "understand"]
    
    objects = ["breakfast", "lunch", "dinner", "water", "coffee", "tea", "juice", "milk",
               "the book", "the paper", "the letter", "the email", "the news", "the story",
               "the park", "the garden", "the house", "the office", "the school", "the store",
               "the car", "the bike", "the train", "the bus", "the plane", "the ship",
               "music", "art", "sports", "games", "movies", "news", "the weather", "the time"]
    
    adverbs = ["quickly", "slowly", "carefully", "easily", "hard", "well", "badly", "early",
               "late", "often", "always", "never", "sometimes", "usually", "today", "yesterday",
               "tomorrow", "now", "then", "here", "there", "inside", "outside", "up", "down"]
    
    prepositions = ["in", "on", "at", "to", "from", "with", "by", "for", "about", "into",
                    "over", "under", "behind", "between", "among", "near", "far", "through"]
    
    # Generate thousands of combinations
    sentences = []
    
    # Pattern 1: Subject + Verb
    for s in subjects:
        for v in verbs:
            sentences.append(f"{s} {v}")
    
    # Pattern 2: Subject + Verb + Object
    for s in subjects:
        for v in verbs[:20]:  # Top 20 verbs
            for o in objects:
                sentences.append(f"{s} {v} {o}")
    
    # Pattern 3: Subject + Verb + Adverb
    for s in subjects:
        for v in verbs[:15]:
            for a in adverbs:
                sentences.append(f"{s} {v} {a}")
    
    # Pattern 4: Subject + Verb + Preposition + Object
    for s in subjects:
        for v in verbs[:15]:
            for p in prepositions:
                for o in objects[:20]:
                    sentences.append(f"{s} {v} {p} {o}")
    
    # Add question patterns
    questions = ["what", "where", "when", "why", "how", "who"]
    for q in questions:
        for s in subjects:
            for v in verbs[:10]:
                sentences.append(f"{q} does {s} {v}")
    
    # Add past tense patterns
    past_verbs = ["ate", "drank", "ran", "walked", "sat", "stood", "slept", "woke", "thought", "knew",
                  "saw", "heard", "felt", "smelled", "touched", "looked", "watched", "listened", "spoke", "talked"]
    for s in subjects:
        for pv in past_verbs:
            for o in objects:
                sentences.append(f"{s} {pv} {o}")
    
    # Add future patterns
    for s in subjects:
        for v in verbs[:15]:
            sentences.append(f"{s} will {v}")
            sentences.append(f"{s} will {v} tomorrow")
    
    # Add continuous patterns
    for s in subjects:
        for v in verbs[:10]:
            sentences.append(f"{s} is {v}ing")
            sentences.append(f"{s} is {v}ing now")
    
    # Add conditional patterns
    for s in subjects[:10]:
        for v in verbs[:10]:
            sentences.append(f"if {s} {v} i will help")
    
    # Add negation patterns
    for s in subjects[:10]:
        for v in verbs[:10]:
            sentences.append(f"{s} does not {v}")
            sentences.append(f"{s} did not {v}")
    
    return sentences


# Generate 5000 unique sentences
ENGLISH_CORPUS = generate_sentences()

# Deduplicate and limit
ENGLISH_CORPUS = list(set(ENGLISH_CORPUS))[:5000]

print(f"Generated {len(ENGLISH_CORPUS)} unique sentences")