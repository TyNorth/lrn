"""
Natural Language Absorption - Scale to Grade Targets
Like a child learning over years - more sources = more vocabulary
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor
from lrn.natural_tokenize import learn_from_text, discover_words


def get_expanded_sources():
    """More diverse sources - simulating real language exposure."""
    sources = {
        "basic": [
            "mama", "dada", "baba", "ball", "dog", "cat", "milk", "eat", "drink",
            "sleep", "play", "hug", "kiss", "wave", "clap", "up", "down", "yes", "no",
            "hello", "bye", "hi", "oh", "wow", "ouch", "yum", "yay", "boo",
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
            "my", "your", "his", "her", "its", "our", "their", "mine", "yours",
            "this", "that", "these", "those", "here", "there", "where", "when", "what",
            "who", "why", "how", "which", "can", "could", "will", "would", "should",
            "do", "does", "did", "have", "has", "had", "go", "goes", "went", "gone",
            "come", "came", "come", "see", "saw", "seen", "know", "knew", "known",
            "think", "thought", "say", "said", "tell", "told", "give", "gave", "given",
            "make", "made", "take", "took", "taken", "get", "got", "got", "keep", "kept",
        ],
        "verbs": [
            "i eat", "you eat", "he eats", "she eats", "it eats", "we eat", "they eat",
            "i drink", "you drink", "he drinks", "she drinks", "it drinks", "we drink",
            "i run", "you run", "he runs", "she runs", "it runs", "we run", "they run",
            "i walk", "you walk", "he walks", "she walks", "it walks", "we walk",
            "i sit", "you sit", "he sits", "she sits", "it sits", "we sit", "they sit",
            "i sleep", "you sleep", "he sleeps", "she sleeps", "it sleeps", "we sleep",
            "i wake", "you wake", "he wakes", "she wakes", "it wakes", "we wake",
            "i think", "you think", "he thinks", "she thinks", "it thinks", "we think",
            "i know", "you know", "he knows", "she knows", "it knows", "we know",
            "i see", "you see", "he sees", "she sees", "it sees", "we see", "they see",
            "i hear", "you hear", "he hears", "she hears", "it hears", "we hear",
            "i feel", "you feel", "he feels", "she feels", "it feels", "we feel",
            "i want", "you want", "he wants", "she wants", "it wants", "we want",
            "i need", "you need", "he needs", "she needs", "it needs", "we need",
            "i love", "you love", "he loves", "she loves", "it loves", "we love",
            "i like", "you like", "he likes", "she likes", "it likes", "we like",
            "i hate", "you hate", "he hates", "she hates", "it hates", "we hate",
        ],
        "actions": [
            "i am eating", "you are eating", "he is eating", "she is eating",
            "i am drinking", "you are drinking", "he is drinking", "she is drinking",
            "i am running", "you are running", "he is running", "she is running",
            "i am walking", "you are walking", "he is walking", "she is walking",
            "i am sleeping", "you are sleeping", "he is sleeping", "she is sleeping",
            "i am thinking", "you are thinking", "he is thinking", "she is thinking",
            "i am working", "you are working", "he is working", "she is working",
            "i am playing", "you are playing", "he is playing", "she is playing",
            "i am reading", "you are reading", "he is reading", "she is reading",
            "i am writing", "you are writing", "he is writing", "she is writing",
        ],
        "past": [
            "i ate yesterday", "you ate yesterday", "he ate yesterday", "she ate yesterday",
            "i drank water", "you drank water", "he drank water", "she drank water",
            "i ran fast", "you ran fast", "he ran fast", "she ran fast",
            "i walked home", "you walked home", "he walked home", "she walked home",
            "i slept well", "you slept well", "he slept well", "she slept well",
            "i saw a movie", "you saw a movie", "he saw a movie", "she saw a movie",
            "i heard a sound", "you heard a sound", "he heard a sound", "she heard a sound",
            "i learned english", "you learned english", "he learned english",
            "i bought a book", "you bought a book", "he bought a book", "she bought a book",
            "i made dinner", "you made dinner", "he made dinner", "she made dinner",
        ],
        "future": [
            "i will eat later", "you will eat later", "he will eat later", "she will eat",
            "i will drink later", "you will drink later", "he will drink", "she will drink",
            "i will go tomorrow", "you will go tomorrow", "he will go", "she will go",
            "i will come back", "you will come back", "he will come back", "she will come",
            "i will see you soon", "you will see me soon", "he will see us", "she will see",
            "i will call you", "you will call me", "he will call her", "she will call him",
            "i will work tomorrow", "you will work tomorrow", "he will work", "she will work",
            "i will play later", "you will play later", "he will play", "she will play",
        ],
        "questions": [
            "what is your name", "where do you live", "when does it start",
            "why are you sad", "how are you today", "which one do you want",
            "who is that man", "can i help you", "may i sit here", "would you like tea",
            "what time is it", "where are we going", "why not try again",
            "how does this work", "which is better", "who can help me",
            "what do you think", "where have you been", "when will it end",
        ],
        "descriptions": [
            "the big dog", "the small cat", "the red ball", "the blue sky",
            "a tall tree", "a short man", "a pretty flower", "a cute baby",
            "the happy child", "the sad person", "the angry teacher", "the tired worker",
            "a beautiful day", "a terrible storm", "a quiet room", "a noisy street",
            "the old house", "the new car", "the fast train", "the slow bus",
            "hot coffee", "cold water", "warm soup", "cool breeze",
        ],
        "places": [
            "i am at home", "she is in school", "he is at work", "they are in the park",
            "we are at the beach", "the cat is on the bed", "the dog is under the table",
            "the bird is in the tree", "the fish is in the water", "the book is on the shelf",
            "the store is near", "the hospital is far", "the library is open",
            "the city is big", "the village is small", "the mountain is high",
            "in the house", "on the wall", "under the bed", "behind the door",
            "between the trees", "near the school", "at the station", "by the river",
        ],
        "time": [
            "today is monday", "tomorrow is tuesday", "yesterday was sunday",
            "the morning is early", "the afternoon is warm", "the evening is cool",
            "the night is dark", "the week has seven days", "the month has thirty days",
            "spring is warm", "summer is hot", "autumn is cool", "winter is cold",
            "in the morning", "at noon", "in the afternoon", "at night",
            "every day", "every week", "every month", "every year",
        ],
        "family": [
            "my mother is kind", "my father is strong", "my sister is smart",
            "my brother is tall", "my grandma is sweet", "my grandpa is wise",
            "your mother is caring", "your father is busy", "his mother loves him",
            "her father supports her", "our family is happy", "their home is nice",
            "i love my family", "we help each other", "they support us",
            "mother cooks", "father works", "sister studies", "brother plays",
        ],
        "school": [
            "the teacher explains", "the student listens", "the class learns",
            "the book is interesting", "the lesson is easy", "the test is hard",
            "we study math", "we read english", "we write essays", "we do science",
            "the school starts at eight", "the break is at ten", "the lunch is at twelve",
            "homework is important", "the exam is tomorrow", "the grade is good",
            "i learn new things", "you understand well", "he works hard",
        ],
        "nature": [
            "the sun rises", "the moon shines", "the stars twinkle",
            "the river flows", "the ocean is big", "the mountain is high",
            "the tree grows", "the flower blooms", "the grass is green",
            "the wind blows", "the rain falls", "the snow melts",
            "birds fly", "fish swim", "dogs run", "cats jump",
            "the sky is blue", "the clouds are white", "the air is fresh",
        ],
        "food": [
            "i eat bread", "you eat rice", "he eats pasta", "she eats vegetables",
            "i drink water", "you drink juice", "he drinks milk", "she drinks tea",
            "breakfast is important", "lunch is at noon", "dinner is at night",
            "fruits are healthy", "vegetables are good", "meat has protein",
            "the restaurant serves food", "the chef cooks well", "the menu has choices",
        ],
        "technology": [
            "the computer works", "the phone rings", "the internet connects",
            "the screen shows data", "the keyboard types", "the mouse clicks",
            "the camera takes photos", "the speaker plays sound", "the printer prints",
            "i use the laptop", "you use the tablet", "he uses the phone",
            "the app loads fast", "the wifi is strong", "the battery lasts",
        ],
        "common_phrases": [
            "good morning", "good afternoon", "good evening", "good night",
            "thank you very much", "you are welcome", "excuse me please",
            "i am sorry", "do not worry", "be happy", "stay safe",
            "have a nice day", "see you later", "take care", "good luck",
            "congratulations", "well done", "keep trying", "never give up",
        ],
    }
    return sources


def train_to_target(target_words: int, max_exposures: int = 20000) -> dict:
    """Train until reaching target vocabulary."""
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    sources = get_expanded_sources()
    all_samples = []
    for s in sources.values():
        all_samples.extend(s)
    
    words = 0
    exposures = 0
    reps_per_batch = 20
    
    while words < target_words and exposures < max_exposures:
        batch = all_samples[:200]  # Use subset for speed
        
        for _ in range(reps_per_batch):
            for text in batch:
                learn_from_text(lnn, text, repetitions=1)
        
        exposures += reps_per_batch * len(batch)
        words = len(discover_words(lnn, all_samples[:500], min_frequency=1))
        
        if exposures % 2000 == 0:
            print(f"  {exposures} exposures: {words} words")
    
    return {"words": words, "exposures": exposures}


def main():
    print("=" * 60)
    print("NATURAL LANGUAGE - GRADE TARGETS")
    print("Massive diverse absorption from many sources")
    print("=" * 60)
    
    targets = [
        (250, "grade1"),
        (500, "grade2"),
        (1000, "grade3"),
        (2000, "grade4"),
        (3500, "grade5"),
        (5000, "grade6"),
    ]
    
    results = {}
    
    for target, grade in targets:
        print(f"\n--- Target: {target} ({grade}) ---")
        result = train_to_target(target)
        results[grade] = result
        
        status = "PASS" if result["words"] >= target else "FAIL"
        print(f"Result: {result['words']}/{target} words - {status}")
        
        if result["words"] < target:
            print(f"Stopped at {result['exposures']} exposures")
            break
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for grade, result in results.items():
        print(f"  {grade}: {result['words']} words ({result['exposures']} exposures)")


if __name__ == "__main__":
    main()