"""
Masterclass English Training - Pre-K to College Level
Iterative buildup with progressively complex English content.
"""
import sys
import time

sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.natural_tokenize import learn_from_text
from lrn.inference import add_word_nodes


def run_rem(lnn, wake_buffer):
    recent_words = set()
    for s in wake_buffer:
        for w in s.lower().split():
            recent_words.add(f"word:{w}")
    
    word_list = list(recent_words)
    for i in range(len(word_list)):
        for j in range(i+1, len(word_list)):
            a, b = word_list[i], word_list[j]
            key = lnn._key(a, b)
            if key in lnn.springs:
                sp = lnn.springs[key]
                if sp.tau > 2:
                    sp.tau = 3
                    sp.stiffness = max(sp.stiffness, 10)
            else:
                lnn.add_or_update_spring(a, b, stiffness=10, tau=3, mode="add")


def train(lnn, sentences, reps=20, learn_type="language"):
    wake_buffer = []
    for rep in range(reps):
        for sentence in sentences:
            learn_from_text(lnn, sentence, repetitions=1, learn_type=learn_type)
            add_word_nodes(lnn, [sentence])
            wake_buffer.append(sentence)
            if len(wake_buffer) > 5:
                wake_buffer = wake_buffer[-5:]
        run_rem(lnn, wake_buffer)
        propagate(lnn, n_steps=2)
    run_rem(lnn, wake_buffer)
    propagate(lnn, n_steps=2)


# Pre-K: Letters, sounds, basic words
PREK = [
    "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
    "a b c d e f g h i j k l m n o p q r s t u v w x y z",
    "cat hat mat sat bat", "dog log fog hog",
    "big red sun run fun", "the cat is big",
    "i see the cat", "i see the dog",
    "the sun is hot", "the moon is bright",
]

# Kindergarten: Simple sentences, sight words
KINDERGARTEN = [
    "i can see the cat", "the dog can run",
    "the bird can fly", "the fish can swim",
    "i like to play", "we go to school",
    "the book is good", "i have a friend",
    "the tree is tall", "the flower is pretty",
    "i love my mom", "i love my dad",
    "the sky is blue", "the grass is green",
    "one two three four five six seven eight nine ten",
]

# Grade 1-2: Phonics families, simple grammar
GRADE_1_2 = [
    "cat hat mat sat bat fat rat", "bike hike like Mike",
    "boat coat float goat", "moon June spoon noon",
    "cake make take lake", "name game same fame",
    "the cat eats fish", "the dog runs fast",
    "the big dog sleeps", "the small cat runs",
    "i eat food", "you drink water",
    "he runs fast", "she sleeps late",
    "the red ball is big", "the blue sky is clear",
    "i go to the park", "she goes to school",
    "we play in the yard", "they read a book",
]

# Grade 3-4: Complex sentences, vocabulary building
GRADE_3_4 = [
    "the cat that sleeps on the mat is very lazy",
    "the dog that runs in the park is very fast",
    "i know that you are my best friend",
    "she says that the book is very good",
    "if it rains we will stay inside",
    "if you study hard you will pass the test",
    "the apple is a fruit", "the carrot is a vegetable",
    "the lion is an animal", "the rose is a flower",
    "happy glad joyful cheerful delighted",
    "sad unhappy sorrowful miserable gloomy",
    "big large huge massive enormous gigantic",
    "small tiny little miniature microscopic",
    "fast quick rapid swift speedy",
    "slow sluggish gradual leisurely",
]

# Grade 5-6: Advanced grammar, literary devices
GRADE_5_6 = [
    "the boy who won the race was very proud",
    "the girl whom i met yesterday is very kind",
    "the book which i read was fascinating",
    "although it was raining we went outside",
    "because she studied hard she passed the exam",
    "the metaphor was like a bridge between ideas",
    "the simile made the description vivid and clear",
    "the personification gave life to the inanimate object",
    "the alliteration added rhythm to the poem",
    "the onomatopoeia created sound effects in the text",
    "the irony was that he failed despite studying",
    "the foreshadowing hinted at the tragic ending",
    "the symbolism represented freedom and hope",
    "the theme of the story was courage and perseverance",
]

# Grade 7-8: Analytical writing, complex structures
GRADE_7_8 = [
    "the author uses imagery to create a vivid scene",
    "the narrative structure follows a chronological order",
    "the protagonist faces a moral dilemma",
    "the antagonist represents the forces of oppression",
    "the rising action builds tension throughout the chapter",
    "the climax reveals the true nature of the conflict",
    "the falling action resolves the main plot threads",
    "the denouement provides closure to the story",
    "the thesis statement presents the main argument",
    "the evidence supports the claim effectively",
    "the counterargument acknowledges opposing views",
    "the conclusion synthesizes the key findings",
    "the rhetorical question engages the reader",
    "the persuasive essay appeals to logic and emotion",
]

# High School: Literary analysis, academic writing
HIGH_SCHOOL = [
    "shakespeare uses iambic pentameter to create rhythm",
    "the tragic flaw leads to the hero downfall",
    "the dramatic irony creates tension for the audience",
    "the soliloquy reveals the character inner thoughts",
    "the sonnet follows a strict fourteen line structure",
    "the essay employs deductive reasoning effectively",
    "the research methodology ensures valid results",
    "the literature review synthesizes existing scholarship",
    "the hypothesis predicts a measurable outcome",
    "the statistical analysis confirms the correlation",
    "the peer review process validates the findings",
    "the abstract summarizes the entire study",
    "the bibliography cites all referenced sources",
    "the appendix contains supplementary data",
]

# College: Advanced academic, specialized vocabulary
COLLEGE = [
    "the epistemological framework underpins the theoretical model",
    "the phenomenological approach examines lived experience",
    "the hermeneutic circle interprets textual meaning",
    "the dialectical method synthesizes opposing viewpoints",
    "the ontological status of consciousness remains debated",
    "the paradigm shift revolutionized scientific understanding",
    "the heuristic algorithm optimizes computational efficiency",
    "the stochastic process models random variables",
    "the asymptotic behavior converges to equilibrium",
    "the covariance matrix captures multivariate relationships",
    "the regression analysis controls for confounding factors",
    "the longitudinal study tracks developmental trajectories",
    "the meta analysis aggregates effect sizes across studies",
    "the systematic review evaluates methodological quality",
]


LEVELS = [
    ("Pre-K", PREK, "sensory"),
    ("Kindergarten", KINDERGARTEN, "language"),
    ("Grade 1-2", GRADE_1_2, "language"),
    ("Grade 3-4", GRADE_3_4, "language"),
    ("Grade 5-6", GRADE_5_6, "language"),
    ("Grade 7-8", GRADE_7_8, "language"),
    ("High School", HIGH_SCHOOL, "language"),
    ("College", COLLEGE, "language"),
]


def main():
    print("=" * 60)
    print("MASTERCLASS ENGLISH TRAINING")
    print("Pre-K to College Level")
    print("=" * 60)
    
    start = time.time()
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    cumulative_corpus = []
    results = {}
    
    for level_name, corpus, learn_type in LEVELS:
        print(f"\n{'='*40}")
        print(f"LEVEL: {level_name.upper()}")
        print(f"{'='*40}")
        
        cumulative_corpus.extend(corpus)
        
        t0 = time.time()
        train(lnn, cumulative_corpus, reps=20, learn_type=learn_type)
        
        word_nodes = sum(1 for n in lnn.nodes if n.startswith("word:"))
        tau3 = sum(1 for sp in lnn.springs.values() if sp.tau == 3)
        
        elapsed = time.time() - t0
        print(f"  Corpus: {len(cumulative_corpus)} sentences")
        print(f"  Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
        print(f"  Words: {word_nodes}, τ=3: {tau3}")
        print(f"  Time: {elapsed:.1f}s")
        
        results[level_name] = {
            "nodes": len(lnn.nodes),
            "springs": len(lnn.springs),
            "words": word_nodes,
            "tau3": tau3,
            "time": elapsed,
        }
    
    total_time = time.time() - start
    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {total_time:.1f}s")
    print(f"{'='*60}")
    
    # Run comprehensive tests
    print("\n=== CATEGORY TEST ===")
    test_categories(lnn)
    
    print("\n=== PHONICS TEST ===")
    test_phonics(lnn)
    
    print("\n=== GRAMMAR TEST ===")
    test_grammar(lnn)
    
    print("\n=== VOCABULARY TEST ===")
    test_vocabulary(lnn)
    
    print("\n=== ANALYSIS TEST ===")
    test_analysis(lnn)
    
    print("\n=== GENERATION TEST ===")
    test_generation(lnn)
    
    return results


def test_categories(lnn):
    tests = {
        "animals": ("word:cat", ["dog", "bird", "fish", "horse", "cow"]),
        "fruits": ("word:apple", ["pear", "banana", "orange", "grape"]),
        "colors": ("word:red", ["blue", "green", "yellow", "orange"]),
        "emotions": ("word:happy", ["sad", "angry", "scared", "excited"]),
        "sizes": ("word:big", ["small", "large", "huge", "tiny"]),
    }
    
    for category, (query, expected) in tests.items():
        for n in lnn.nodes.values():
            n.activation = 0
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            propagate(lnn, n_steps=5)
            
            activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            found = [w for w in expected if w in activated]
            status = "PASS" if len(found) >= 2 else "FAIL"
            print(f"  {category}: {status} ({len(found)}/{len(expected)}: {found})")
        else:
            print(f"  {category}: FAIL")


def test_phonics(lnn):
    tests = [
        ("cat", ["hat", "mat", "sat"]),
        ("bike", ["hike", "like"]),
        ("boat", ["coat", "float"]),
        ("moon", ["spoon", "noon"]),
    ]
    
    for word, family in tests:
        query = f"word:{word}"
        for n in lnn.nodes.values():
            n.activation = 0
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            propagate(lnn, n_steps=5)
            
            activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            found = [w for w in family if w in activated]
            status = "PASS" if found else "FAIL"
            print(f"  {word}: {status} ({found})")
        else:
            print(f"  {word}: FAIL")


def test_grammar(lnn):
    from lrn.grammar_training import infer_pos
    
    test_words = {
        "cat": "noun", "eats": "verb", "big": "adjective",
        "the": "determiner", "runs": "verb", "dog": "noun",
        "fast": "adjective", "sleeps": "verb",
    }
    
    for word, expected in test_words.items():
        result = infer_pos(lnn, word)
        actual = result.get("pos", "unknown")
        status = "PASS" if actual == expected else "FAIL"
        print(f"  {word}: {status} ({expected} → {actual})")


def test_vocabulary(lnn):
    """Test synonym/antonym recognition."""
    tests = {
        "happy_synonyms": ("word:happy", ["glad", "joyful", "cheerful"]),
        "sad_synonyms": ("word:sad", ["unhappy", "sorrowful"]),
        "big_synonyms": ("word:big", ["large", "huge", "massive"]),
        "small_synonyms": ("word:small", ["tiny", "little"]),
    }
    
    for name, (query, expected) in tests.items():
        for n in lnn.nodes.values():
            n.activation = 0
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            propagate(lnn, n_steps=5)
            
            activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            found = [w for w in expected if w in activated]
            status = "PASS" if found else "FAIL"
            print(f"  {name}: {status} ({found})")
        else:
            print(f"  {name}: FAIL")


def test_analysis(lnn):
    """Test literary analysis concepts."""
    tests = {
        "metaphor": ("word:metaphor", ["simile", "symbolism"]),
        "story": ("word:story", ["theme", "plot", "character"]),
        "author": ("word:author", ["book", "write", "read"]),
    }
    
    for name, (query, expected) in tests.items():
        for n in lnn.nodes.values():
            n.activation = 0
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            propagate(lnn, n_steps=5)
            
            activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            found = [w for w in expected if w in activated]
            status = "PASS" if found else "FAIL"
            print(f"  {name}: {status} ({found})")
        else:
            print(f"  {name}: FAIL (node not found)")


def test_generation(lnn):
    from lrn.inference import attention_with_residue
    
    seeds = ["the", "cat", "i", "you", "if", "she", "he", "because"]
    
    for seed in seeds:
        result = attention_with_residue(lnn, f"word:{seed}", propagate_steps=3)
        if result["attention"]:
            words = [n.replace("word:", "") for n, _ in result["attention"][:3]]
            print(f"  '{seed}' → {words}")
        else:
            print(f"  '{seed}' → (none)")


if __name__ == "__main__":
    main()
