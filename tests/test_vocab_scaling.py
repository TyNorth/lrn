"""
LRN Language Learning - Test with increasing repetitions
See how vocabulary scales with training
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor
from lrn.natural_tokenize import learn_from_text, discover_words


SENTENCES = [
    "the cat sits on the mat",
    "the dog runs in the park",
    "the bird flies over the tree",
    "i eat breakfast every morning",
    "she drinks coffee at noon",
    "he walks to school every day",
    "we play games after lunch",
    "they watch movies at night",
    "the sun rises in the east",
    "the moon shines at night",
    "water flows downstream",
    "fire burns hot and bright",
    "air moves across the sky",
    "earth rotates around the sun",
    "time passes without stopping",
]


def test_scaling():
    """Test vocabulary growth with more repetitions."""
    print("=" * 60)
    print("VOCABULARY SCALING TEST")
    print("=" * 60)
    
    for reps in [20, 50, 100, 200, 300]:
        lnn = LatticeNN()
        add_identity_anchor(lnn)
        
        for _ in range(reps):
            for text in SENTENCES:
                learn_from_text(lnn, text, repetitions=1)
        
        words = discover_words(lnn, SENTENCES, min_frequency=2)
        
        print(f"  {reps} reps: {len(words)} words")
        
        if len(words) >= 15:
            print(f"    Words: {list(words.keys())[:15]}")


def test_more_sentences():
    """Test with more sentence variety."""
    print("\n" + "=" * 60)
    print("MORE SENTENCES TEST")
    print("=" * 60)
    
    more_sentences = SENTENCES + [
        "what is your name",
        "where do you live",
        "when does the train leave",
        "why did he leave so early",
        "how are you feeling today",
        "which book should i read",
        "who is coming to the party",
        "can you help me please",
        "would you like some tea",
        "may i ask a question",
        "i walked to the store yesterday",
        "she bought a new dress last week",
        "he ate breakfast at seven am",
        "they played soccer in the park",
        "we saw a beautiful sunset",
        "the child fell down and cried",
        "she wrote a letter to her friend",
        "he drove to the city alone",
        "they built a house last year",
        "i learned english when i was young",
        "i am reading a book right now",
        "she is cooking dinner in the kitchen",
        "he is driving to work this morning",
        "they are playing tennis at the club",
        "we are studying for the exam",
        "i will go to the market tomorrow",
        "she will call you later today",
        "he will finish the work next week",
        "they will visit us next month",
    ]
    
    print(f"Total sentences: {len(more_sentences)}")
    
    for reps in [50, 100, 200]:
        lnn = LatticeNN()
        add_identity_anchor(lnn)
        
        for _ in range(reps):
            for text in more_sentences:
                learn_from_text(lnn, text, repetitions=1)
        
        words = discover_words(lnn, more_sentences, min_frequency=2)
        
        print(f"  {reps} reps: {len(words)} words")
        
        if reps == 200:
            print(f"    Words: {list(words.keys())[:20]}")


def main():
    test_scaling()
    test_more_sentences()


if __name__ == "__main__":
    main()