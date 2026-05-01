"""
English Language - Test with correct threshold to reach grade targets
Targets: 250, 500, 1000, 2000, 3500, 5000
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, get_language_samples
from lrn.natural_tokenize import learn_from_text, discover_words


# Better corpus - more variety = more words
ENGLISH_CORPUS = [
    # Basic (100)
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
    "the cat is on the bed",
    "the dog barks at the mailman",
    "the bird sings in the tree",
    "i drink water at lunch",
    "she reads books in the evening",
    "he works at the office",
    "we dance at the party",
    "they sing songs at church",
    "the star twinkles at night",
    "the cloud floats in the sky",
    "rain falls from the clouds",
    "snow covers the ground",
    "wind blows across the fields",
    "leaves fall from the trees",
    "flowers bloom in spring",
    
    # Questions (100)
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
    "does she understand the problem",
    "was he there yesterday",
    "have they finished the work",
    "will you be there tomorrow",
    "should i call the doctor",
    "what time is it now",
    "where are you going",
    "why are you laughing",
    "how does this work",
    "which one is yours",
    
    # Past tense (150)
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
    "she became a doctor after many years",
    "he broke his leg in the accident",
    "they caught a big fish yesterday",
    "we spent a week in the mountains",
    "i thought about the problem all night",
    "she closed the door quietly",
    "he opened the window for air",
    "they painted the wall blue",
    "we cleaned the house today",
    "i finished my homework late",
    
    # Present continuous (100)
    "i am reading a book right now",
    "she is cooking dinner in the kitchen",
    "he is driving to work this morning",
    "they are playing tennis at the club",
    "we are studying for the exam",
    "the baby is sleeping in the crib",
    "she is talking on the phone",
    "he is working on the project",
    "they are building a new house",
    "we are waiting for the bus",
    "i am thinking about the future",
    "she is writing a letter to her mother",
    "he is running in the park",
    "they are singing at the concert",
    "we are dancing at the party",
    
    # Future (100)
    "i will go to the market tomorrow",
    "she will call you later today",
    "he will finish the work next week",
    "they will visit us next month",
    "we will learn a new language soon",
    "the sun will rise at six tomorrow",
    "she will be here at nine pm",
    "he will become a lawyer soon",
    "they will change the world",
    "we will celebrate our success",
    "i will remember this day forever",
    "she will marry him next spring",
    "he will travel around the world",
    "they will find the solution",
    "we will achieve our goals together",
    
    # Conditionals (80)
    "if it rains tomorrow i will stay home",
    "if you study hard you will pass the exam",
    "if she comes i will be very happy",
    "if they help us we can finish faster",
    "if he is late we will start without him",
    "if you need help just ask me",
    "if it is too hot we can go swimming",
    "if you are tired we can rest here",
    "if the food is good i will come again",
    "if you want more just tell me",
    
    # Common phrases (150)
    "actions speak louder than words",
    "the early bird catches the worm",
    "a penny saved is a penny earned",
    "all that glitters is not gold",
    "better late than never",
    "birds of a feather flock together",
    "every cloud has a silver lining",
    "no pain no gain",
    "time is of the essence",
    "to err is human",
    "the ball is in your court",
    "burn the midnight oil",
    "catch your breath",
    "change your mind",
    "cost an arm and a leg",
    "cut to the chase",
    "hit the nail on the head",
    "kill two birds with one stone",
    "let the cat out of the bag",
    "make ends meet",
    
    # More variety (200)
    "the quick brown fox jumps over the lazy dog",
    "she sells seashells by the seashore",
    "how much wood would a woodchuck chuck",
    "i think therefore i am",
    "to be or not to be that is the question",
    "all that glitters is not gold",
    "the apple doesn't fall far from the tree",
    "a bird in the hand is worth two in the bush",
    "better safe than sorry",
    "charity begins at home",
    "don't count your chickens before they hatch",
    "every rose has its thorn",
    "good things come to those who wait",
    "hope for the best but prepare for the worst",
    "if you can't beat them join them",
    "it takes two to tango",
    "knowledge is power",
    "look before you leap",
    "mistakes are proof that you are trying",
    "never say never",
]


def test_to_grade(target: int, min_freq: int = 1) -> dict:
    """Train until reaching target word count."""
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    words = 0
    reps = 0
    max_reps = 500
    
    while words < target and reps < max_reps:
        batch_size = min(50, len(ENGLISH_CORPUS))
        batch = ENGLISH_CORPUS[:batch_size]
        
        for _ in range(10):
            for text in batch:
                learn_from_text(lnn, text, repetitions=1)
        
        reps += 10
        words = len(discover_words(lnn, batch, min_frequency=min_freq))
        
        if reps % 50 == 0:
            print(f"  {reps} reps: {words} words")
    
    return {"words": words, "reps": reps}


def main():
    print("=" * 60)
    print("ENGLISH LANGUAGE - GRADE TARGETS")
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
        result = test_to_grade(target)
        results[grade] = result
        
        status = "PASS" if result["words"] >= target else "FAIL"
        print(f"Result: {result['words']}/{target} - {status}")
        
        if result["words"] < target:
            print(f"Stopped at {result['reps']} reps")
            break
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for grade, result in results.items():
        print(f"  {grade}: {result['words']} words ({result['reps']} reps)")


if __name__ == "__main__":
    main()