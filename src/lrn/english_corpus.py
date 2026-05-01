"""
English Training Corpus - Organized by Developmental Stage

Stages:
1. Sensory/Letters - Visual letter recognition
2. Babbling - Sound exploration, motor memory
3. Phonics - Letter→sound mapping (CVC words)
4. Morphology - Prefixes/suffixes (language quarks)
5. Sight Words - High-frequency instant recognition
6. Vocabulary - Word meanings, synonyms, antonyms
7. Grammar - POS, word order, tenses
8. Syntax Trees - Hierarchical phrase structure
9. Sentences - Full comprehension/production
10. Pragmatics - Social context, tone
"""

# ============================================================================
# STAGE 1: SENSORY/LETTER RECOGNITION (τ=0)
# ============================================================================

LETTER_CORPUS = [
    # Uppercase letters as visual shapes
    "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
    # Lowercase letters
    "a b c d e f g h i j k l m n o p q r s t u v w x y z",
    # Letter groups by shape similarity
    "A H K M N V W X Y Z",  # Straight lines
    "C G O Q S U",  # Curved shapes
    "B D P Q",  # Loop shapes
    "E F L T",  # Right angles
    # Letter pairs that look similar
    "b d", "p q", "m n", "u n", "i l",
]

# ============================================================================
# STAGE 2: BABBLING (τ=1)
# ============================================================================

BABBLING_CORPUS = [
    # Repetitive syllables (motor memory)
    "ba ba ba", "da da da", "ma ma ma", "pa pa pa",
    "na na na", "ta ta ta", "ga ga ga", "ka ka ka",
    # Consonant-vowel combinations
    "ba be bi bo bu", "da de di do du", "ma me mi mo mu",
    "pa pe pi po pu", "na ne ni no nu", "ta te ti to tu",
    # Vowel-consonant combinations
    "ab eb ib ob ub", "ad ed id od ud", "am em im om um",
    "ap ep ip op up", "an en in on un", "at et it ot ut",
    # Consonant clusters
    "bl bl bl", "cl cl cl", "fl fl fl", "gl gl gl",
    "br br br", "cr cr cr", "dr dr dr", "fr fr fr",
    "st st st", "sp sp sp", "sk sk sk",
    # Sound exploration
    "ah eh ih oh uh", "ay ee eye oh oo",
    "sh sh sh", "ch ch ch", "th th th", "wh wh wh",
]

# ============================================================================
# STAGE 3: PHONICS (τ=1)
# ============================================================================

PHONICS_CORPUS = {
    "short_vowels": [
        # -at family
        "cat hat mat sat bat fat rat",
        # -ed family
        "bed red fed led shed",
        # -it family
        "bit sit fit hit kit",
        # -ot family
        "hot pot lot dot rot got",
        # -ug family
        "bug rug mug hug tug",
        # -an family
        "can fan man pan ran van",
        # -ap family
        "cap map lap nap tap",
        # -en family
        "pen hen ten men",
        # -ig family
        "pig big dig wig fig",
        # -og family
        "dog fog log hog jog",
    ],
    "long_vowels": [
        # -ake family
        "cake make take lake fake bake",
        # -ike family
        "bike hike like Mike side ride",
        # -oat family
        "boat coat float goat note",
        # -oon family
        "moon June spoon noon soon",
        # -ame family
        "name game same fame came",
        # -ine family
        "line mine fine nine pine",
        # -ope family
        "hope rope scope",
        # -ute family
        "cute mute flute",
    ],
    "consonant_blends": [
        # Initial blends
        "black flat glad plan sleep",
        "brush crash dream frame",
        "swim swing sweep sweet",
        "clap flip grip snap trap",
        # Final blends
        "fast last mast cast",
        "desk mask risk",
        "help melt belt",
    ],
    "digraphs": [
        # sh
        "ship chin thin when phone",
        "sheep shell shop fish",
        # ch
        "chair chin chunk much",
        "chick chat chop",
        # th
        "thin thick this that",
        "the them they",
        # wh
        "when what where why",
        "white wheel whale",
        # ph
        "phone photo graph",
    ],
    "diphthongs": [
        "boy toy joy",
        "cow how now brown",
        "coin join point",
        "loud mouth house",
    ],
    "r_controlled": [
        "car far bar jar star",
        "corn form storm short",
        "her term verb",
        "bird shirt skirt",
        "nurse purse turn",
    ],
}

# ============================================================================
# STAGE 4: MORPHOLOGY (τ=1-2)
# ============================================================================

MORPHOLOGY_CORPUS = {
    "prefixes": [
        # un-
        "unhappy undo unfair unlock unknown",
        # re-
        "redo rewrite return rebuild replay",
        # pre-
        "preview preheat prepay pretest",
        # dis-
        "disagree disappear dislike disconnect",
        # mis-
        "misread mislead misspell misplace",
        # in-
        "incorrect inactive invisible incomplete",
        # im-
        "impossible impatient impolite",
        # non-
        "nonstop nonsense nonfiction",
    ],
    "suffixes": [
        # -ing
        "running walking eating sleeping playing",
        # -ed
        "walked talked played jumped",
        # -tion
        "action creation education information",
        # -ly
        "quickly slowly happily sadly",
        # -ness
        "happiness kindness darkness",
        # -ment
        "movement agreement development",
        # -er
        "runner walker teacher player",
        # -est
        "biggest smallest fastest tallest",
        # -ful
        "helpful careful beautiful",
        # -less
        "helpless careless hopeless",
    ],
    "compound_words": [
        "sunflower butterfly basketball",
        "rainbow notebook backpack",
        "bedroom bathroom classroom",
        "playground homework toothbrush",
    ],
}

# ============================================================================
# STAGE 5: SIGHT WORDS (τ=2)
# ============================================================================

SIGHT_WORDS_CORPUS = [
    # Dolch Pre-Primer (20)
    "a and away big blue can come down find for funny",
    "go help here I in is it jump little look make me",
    "my not one play red run said see the three to two",
    "up we where yellow you",
    # Dolch Primer (20)
    "all am are at ate be black brown but came did do",
    "eat four get good have he into like must new no",
    "now on our out please pretty ran ride saw say she",
    "so soon that there they this too under want was",
    "well went what white who will with yes",
    # Dolch First Grade (20)
    "about better bring carry clean cut done draw drink",
    "eight fall far full got grow hold hot if keep kind",
    "laugh light long much myself never only own pick",
    "seven shall show six small start ten today together",
    "try warm",
    # Dolch Second Grade (20)
    "always around because been before begin call can't",
    "could every found give going grow just know let",
    "off once open over put round some their think walk",
    "were when would write",
]

# ============================================================================
# STAGE 6: VOCABULARY (τ=2)
# ============================================================================

VOCABULARY_CORPUS = {
    "synonyms": [
        "big large huge giant massive",
        "small tiny little minute",
        "fast quick rapid swift",
        "slow sluggish gradual",
        "happy glad joyful cheerful",
        "sad unhappy sorrowful",
        "angry mad furious",
        "beautiful pretty gorgeous",
        "ugly hideous",
        "smart intelligent clever",
        "stupid foolish dumb",
        "strong powerful mighty",
        "weak feeble frail",
        "hot warm burning",
        "cold freezing chilly",
        "good great excellent",
        "bad terrible awful",
        "old ancient elderly",
        "new fresh modern",
        "rich wealthy affluent",
        "poor needy destitute",
    ],
    "antonyms": [
        "big small", "hot cold", "fast slow", "happy sad",
        "good bad", "old new", "tall short", "rich poor",
        "hard soft", "heavy light", "dark bright", "loud quiet",
        "clean dirty", "full empty", "open closed", "high low",
        "early late", "near far", "rough smooth", "wet dry",
    ],
    "categories": [
        # Animals
        "cat dog bird fish horse cow sheep chicken pig",
        # Colors
        "red blue green yellow orange purple pink black white brown",
        # Body parts
        "head face eye nose mouth ear hand arm leg foot",
        # Food
        "apple banana bread cheese milk egg rice meat",
        # Weather
        "sun rain snow wind cloud storm fog",
        # Emotions
        "happy sad angry scared surprised tired excited",
    ],
}

# ============================================================================
# STAGE 7: GRAMMAR (τ=2-3)
# ============================================================================

GRAMMAR_CORPUS = [
    # SVO sentences
    "the cat eats fish", "the dog sees bird", "the bird flies high",
    "a cat eats food", "a dog runs fast", "the fish swims deep",
    "the horse runs fast", "the cow eats grass", "the sheep eats hay",
    # Pronoun sentences
    "i eat food", "you drink water", "he runs fast",
    "she sleeps late", "it works well", "we play games", "they talk loud",
    # Adjective + Noun
    "the big dog runs", "the small cat sleeps", "the fast car drives",
    "the slow turtle walks", "the hot fire burns", "the cold ice melts",
    "the tall tree grows", "the short grass grows",
    # Preposition phrases
    "in the house", "on the table", "under the bed",
    "behind the door", "near the park", "at the store",
    # Wrong patterns (negative reinforcement)
    "cat dog", "dog bird", "fish horse",
    "eats runs", "sees flies", "drinks sleeps",
    "big small", "fast slow", "hot cold",
    "eats the cat", "runs the dog", "big the dog",
]

# ============================================================================
# STAGE 8: SYNTAX TREES (τ=3)
# ============================================================================

SYNTAX_CORPUS = {
    "simple": [
        "the cat eats fish", "the dog sees cat", "the bird flies",
        "i love music", "she wants food", "he needs water",
    ],
    "compound": [
        "the cat eats and dog sleeps", "i study and you play",
        "he runs but she walks", "we work they rest",
    ],
    "complex": [
        "the cat that sleeps eats", "the dog that runs sees",
        "i know that you are here", "she says that he comes",
    ],
    "question": [
        "what is that", "where are you", "when does it start",
        "why are you here", "who is there",
    ],
    "conditional": [
        "if it rains i stay home", "if you study you pass",
        "if he runs he wins", "if she works she succeeds",
    ],
    "imperative": [
        "sit down", "come here", "open door", "close window",
        "eat food", "drink water", "watch show",
    ],
}

# ============================================================================
# STAGE 9: SENTENCES (τ=3-4)
# ============================================================================

SENTENCE_CORPUS = [
    # Simple sentences
    "the cat sits on the mat", "the dog runs in the park",
    "the bird flies over the tree", "i eat breakfast every morning",
    "she drinks coffee at noon", "he walks to school every day",
    # Questions
    "what is your name", "where do you live", "when does the train leave",
    "why did he leave so early", "how are you feeling today",
    # Tenses
    "i walked to the store yesterday", "she bought a new dress last week",
    "he ate breakfast at seven am", "they played soccer in the park",
    "i am reading a book right now", "she is cooking dinner in the kitchen",
    "i will go to the market tomorrow", "she will call you later today",
    # Common phrases
    "actions speak louder than words", "the early bird catches the worm",
    "a penny saved is a penny earned", "all that glitters is not gold",
    "better late than never", "every cloud has a silver lining",
]

# ============================================================================
# STAGE 10: PRAGMATICS (τ=4)
# ============================================================================

PRAGMATICS_CORPUS = {
    "requests": [
        "can you help me", "could you please open the door",
        "would you mind waiting", "may i ask a question",
        "please pass the salt", "would you like some tea",
    ],
    "politeness": [
        "thank you very much", "you are welcome", "excuse me please",
        "i am sorry", "please and thank you", "good morning",
    ],
    "tone": [
        "i am so happy to see you", "i am really disappointed",
        "that is absolutely wonderful", "i am quite tired",
        "this is somewhat interesting", "i am extremely angry",
    ],
    "implication": [
        "it is cold in here",  # means: close the window
        "i am hungry",  # means: let's eat
        "do you have the time",  # means: tell me the time
        "can you hear me",  # means: speak louder
    ],
}

# ============================================================================
# ALL STAGES COMBINED
# ============================================================================

ALL_STAGES = {
    "sensory": LETTER_CORPUS,
    "babbling": BABBLING_CORPUS,
    "phonics": PHONICS_CORPUS,
    "morphology": MORPHOLOGY_CORPUS,
    "sight_words": SIGHT_WORDS_CORPUS,
    "vocabulary": VOCABULARY_CORPUS,
    "grammar": GRAMMAR_CORPUS,
    "syntax": SYNTAX_CORPUS,
    "sentences": SENTENCE_CORPUS,
    "pragmatics": PRAGMATICS_CORPUS,
}
