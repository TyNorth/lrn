"""
Natural Language Absorption Test - Like a baby learning
Massive diverse input from many sources - no explicit rules
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor
from lrn.natural_tokenize import learn_from_text, discover_words


def simulate_natural_absorption():
    """
    Simulate how a baby learns language naturally:
    - Many different sources (books, conversations, TV, radio)
    - Many different contexts (home, school, work, play)
    - Over a long period (thousands of exposures)
    """
    print("=" * 60)
    print("NATURAL LANGUAGE ABSORPTION")
    print("Like a baby learning from birth - no rules, just exposure")
    print("=" * 60)
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    # Diverse sources - simulate real language exposure
    sources = {
        "baby_talk": [
            "mama", "dada", "baba", "tata", "nana", "dada", "mama", "bye bye",
            "more", "no", "yes", "up", "down", "ball", "dog", "cat", "milk",
            "eat", "drink", "sleep", "play", "hug", "kiss", "wave", "clap",
        ],
        "conversations": [
            "hello how are you", "i am fine thank you", "what is your name",
            "where do you live", "i live in the city", "nice to meet you",
            "good morning good afternoon good evening", "see you later",
            "what time is it", "i do not know", "can you help me",
            "please thank you sorry excuse me", "i understand now",
        ],
        "everyday_actions": [
            "i am eating breakfast", "she is drinking coffee", "he is walking",
            "they are playing games", "we are going to school", "the cat sits",
            "the dog runs fast", "the bird flies high", "the sun shines bright",
            "water flows down", "fire burns hot", "air moves around",
        ],
        "stories": [
            "once upon a time there was a king", "the princess lived in a castle",
            "the dragon guarded the treasure", "the hero saved the day",
            "they lived happily ever after", "the end", "in the beginning",
            "a long time ago", "suddenly something happened",
        ],
        "questions": [
            "what is that", "who is there", "where are we going", "why not",
            "how does it work", "when will it happen", "which one do you want",
            "can i have some", "may i come in", "would you like to play",
        ],
        "emotions": [
            "i feel happy today", "she feels sad sometimes", "he feels angry",
            "they feel excited", "we feel tired", "i love you very much",
            "you make me smile", "that makes me laugh", "i am scared",
            "do not worry be happy",
        ],
        "descriptions": [
            "the big red dog", "a small blue bird", "the tall green tree",
            "a beautiful sunny day", "a dark cloudy night", "the old wise man",
            "the young curious child", "a fast quick rabbit", "a slow fat turtle",
        ],
        "instructions": [
            "sit down please stand up", "open the door close the window",
            "look at this listen to that", "read the book write your name",
            "come here go there", "do this do that", "try again try harder",
        ],
        "school": [
            "the teacher explains the lesson", "the student takes notes",
            "the class starts at nine", "we learn math and reading",
            "the test is on friday", "homework is due monday",
            "the library has many books", "the playground is fun",
        ],
        "family": [
            "my mother cooks dinner", "my father works hard", "my sister plays piano",
            "my brother watches tv", "grandma tells stories", "grandpa fixes things",
            "the family eats together", "we love each other", "happy family",
        ],
        "nature": [
            "the sun rises in the east", "the moon shines at night", "the stars twinkle",
            "the river flows to the sea", "the mountains are very high",
            "the forest has many trees", "the flowers smell sweet",
            "the seasons change always", "weather is different each day",
        ],
        "food": [
            "i eat bread and butter", "she drinks orange juice", "he likes pizza",
            "they want some ice cream", "breakfast is in the morning",
            "lunch is at noon dinner at night", "fruits are healthy",
            "vegetables are good for you", "the restaurant serves food",
        ],
        "time": [
            "today is monday", "tomorrow is tuesday", "yesterday was sunday",
            "the week has seven days", "the month has thirty days",
            "the year has twelve months", "spring summer autumn winter",
            "morning noon afternoon evening night", "what time is it now",
        ],
        "places": [
            "i am at home", "she is in the park", "he goes to school",
            "they work at the office", "we shop at the store",
            "the hospital is nearby", "the library is far", "the beach is warm",
            "the city is busy", "the village is quiet",
        ],
        "technology": [
            "the computer works well", "the phone rings often", "the internet connects",
            "the screen shows pictures", "the keyboard types words",
            "the mouse clicks buttons", "the speaker plays sound",
            "the camera takes photos", "the data is stored", "the code runs",
        ],
    }
    
    # Train with many exposures - simulate years of absorption
    total_samples = 0
    
    print("\nPhase 1: Baby talk (early exposure)...")
    for _ in range(50):
        for text in sources["baby_talk"]:
            learn_from_text(lnn, text, repetitions=1)
            total_samples += 1
    
    print("Phase 2: Conversations...")
    for _ in range(50):
        for text in sources["conversations"]:
            learn_from_text(lnn, text, repetitions=1)
            total_samples += 1
    
    print("Phase 3: Everyday actions...")
    for _ in range(50):
        for text in sources["everyday_actions"]:
            learn_from_text(lnn, text, repetitions=1)
            total_samples += 1
    
    print("Phase 4: Stories...")
    for _ in range(40):
        for text in sources["stories"]:
            learn_from_text(lnn, text, repetitions=1)
            total_samples += 1
    
    print("Phase 5: Questions, emotions, descriptions...")
    for category in ["questions", "emotions", "descriptions", "instructions"]:
        for _ in range(30):
            for text in sources[category]:
                learn_from_text(lnn, text, repetitions=1)
                total_samples += 1
    
    print("Phase 6: School, family, nature...")
    for category in ["school", "family", "nature", "food", "time", "places"]:
        for _ in range(30):
            for text in sources[category]:
                learn_from_text(lnn, text, repetitions=1)
                total_samples += 1
    
    print("Phase 7: Modern (technology)...")
    for _ in range(20):
        for text in sources["technology"]:
            learn_from_text(lnn, text, repetitions=1)
            total_samples += 1
    
    print(f"\nTotal exposures: {total_samples}")
    
    # Collect all samples for word discovery
    all_samples = []
    for samples in sources.values():
        all_samples.extend(samples)
    
    # Discover vocabulary with low threshold (like a baby - learns from few exposures)
    words = discover_words(lnn, all_samples, min_frequency=1)
    
    print(f"\n=== RESULTS ===")
    print(f"Total vocabulary: {len(words)} words")
    print(f"Top 30 words: {list(words.keys())[:30]}")
    
    # Check grammar patterns (multi-word)
    print(f"\n=== GRAMMAR CHECK ===")
    
    # Simple 2-word patterns
    patterns = {}
    for text in all_samples:
        tokens = text.split()
        for i in range(len(tokens) - 1):
            pattern = f"{tokens[i]} {tokens[i+1]}"
            patterns[pattern] = patterns.get(pattern, 0) + 1
    
    # Learned patterns (appear multiple times)
    learned_patterns = {p: c for p, c in patterns.items() if c >= 5}
    print(f"Grammar patterns (2-word): {len(learned_patterns)}")
    print(f"Sample: {list(learned_patterns.keys())[:15]}")
    
    return {"words": len(words), "patterns": len(learned_patterns), "vocabulary": list(words.keys())[:50]}


if __name__ == "__main__":
    result = simulate_natural_absorption()
    print(f"\nFinal: {result['words']} words, {result['patterns']} grammar patterns")