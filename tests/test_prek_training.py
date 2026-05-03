"""
Pre-K English Training - Complete Mastery Assessment
Trains and tests all Pre-K level deliverables.
"""
import sys
import time
import os
import json
from datetime import datetime

sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.natural_tokenize import learn_from_text
from lrn.inference import add_word_nodes


# Create report directory
REPORT_DIR = "/Users/tyarc/github/lrn/reports/prek"
os.makedirs(REPORT_DIR, exist_ok=True)


def optimal_rem(lnn, wake_buffer):
    """REM after every sentence - forms τ=3 bridges for all co-occurring words."""
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
                sp.stiffness = max(sp.stiffness, 10)
                sp.tau = 3
            else:
                lnn.add_or_update_spring(a, b, stiffness=10, tau=3, mode="add")


def train(lnn, sentences, reps=30, learn_type="language"):
    """Training with optimal REM after every sentence."""
    wake_buffer = []
    for rep in range(reps):
        for sentence in sentences:
            learn_from_text(lnn, sentence, repetitions=1, learn_type=learn_type)
            add_word_nodes(lnn, [sentence])
            
            wake_buffer.append(sentence)
            if len(wake_buffer) > 20:
                wake_buffer = wake_buffer[-20:]
            
            optimal_rem(lnn, wake_buffer)
        
        propagate(lnn, n_steps=3)
    
    propagate(lnn, n_steps=5)


# Pre-K Corpus - All deliverables
PREK_CORPUS = [
    # Letters (A-Z recognition)
    "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
    "a b c d e f g h i j k l m n o p q r s t u v w x y z",
    "A is for apple", "B is for ball", "C is for cat",
    "D is for dog", "E is for egg", "F is for fish",
    "G is for goat", "H is for hat", "I is for ice",
    "J is for jam", "K is for kite", "L is for lion",
    "M is for moon", "N is for nest", "O is for orange",
    "P is for pig", "Q is for queen", "R is for rat",
    "S is for sun", "T is for tree", "U is for umbrella",
    "V is for van", "W is for watch", "X is for box",
    "Y is for yellow", "Z is for zebra",
    
    # Numbers (1-10)
    "one two three four five six seven eight nine ten",
    "i have one apple", "i have two cats", "i have three dogs",
    "i see four birds", "i see five fish", "i count six stars",
    "seven is lucky", "eight is great", "nine is fine", "ten is the end",
    
    # Colors
    "red blue green yellow orange purple",
    "black white brown pink gray gold",
    "the apple is red", "the sky is blue", "the grass is green",
    "the sun is yellow", "the orange is orange", "the grape is purple",
    "the night is black", "the snow is white", "the bear is brown",
    "the flower is pink", "the sky is gray", "the ring is gold",
    
    # Shapes
    "circle square triangle rectangle star",
    "the ball is a circle", "the box is a square",
    "the roof is a triangle", "the door is a rectangle",
    "the star is a star", "the sun is a circle",
    
    # Basic Animals
    "cat dog bird fish horse cow",
    "sheep pig chicken duck rabbit",
    "lion tiger bear elephant monkey",
    "the cat says meow", "the dog says woof",
    "the bird says tweet", "the cow says moo",
    "the pig says oink", "the duck says quack",
    
    # Basic Emotions
    "happy sad angry scared surprised tired",
    "i feel happy", "i feel sad", "i feel angry",
    "i feel scared", "i feel surprised", "i feel tired",
    "happy is good", "sad is not good",
    
    # Simple Words (CVC - consonant vowel consonant)
    "cat hat mat sat bat",
    "dog log fog hog jog",
    "bed red fed led shed",
    "sun run fun bun pun",
    "big pig dig wig fig",
    "hot pot lot dot got",
    
    # Simple Sentences
    "i see the cat", "i see the dog",
    "the cat is big", "the dog is small",
    "i like the sun", "i like the moon",
    "the bird can fly", "the fish can swim",
]


def main():
    print("=" * 60)
    print("PRE-K ENGLISH TRAINING")
    print("Deliverables: Letters, Numbers, Colors, Shapes, Animals, Emotions, Phonics")
    print("=" * 60)
    
    start_time = time.time()
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    print("\nTraining Pre-K corpus...")
    t0 = time.time()
    train(lnn, PREK_CORPUS, reps=30, learn_type="sensory")
    train_time = time.time() - t0
    
    print(f"Training time: {train_time:.1f}s")
    print(f"Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
    
    # Run all Pre-K assessments
    results = {}
    
    print("\n" + "=" * 60)
    print("PRE-K ASSESSMENT")
    print("=" * 60)
    
    results["letters"] = test_letters(lnn)
    results["numbers"] = test_numbers(lnn)
    results["colors"] = test_colors(lnn)
    results["shapes"] = test_shapes(lnn)
    results["animals"] = test_animals(lnn)
    results["emotions"] = test_emotions(lnn)
    results["phonics"] = test_phonics(lnn)
    results["sentences"] = test_sentences(lnn)
    
    # Calculate total score
    total_score = sum(r["score"] for r in results.values())
    total_possible = sum(r["possible"] for r in results.values())
    mastery_pct = (total_score * 100) // total_possible
    
    print(f"\n{'='*60}")
    print(f"PRE-K TOTAL: {total_score}/{total_possible} ({mastery_pct}%)")
    print(f"{'='*60}")
    
    mastery_status = "MASTERY" if mastery_pct >= 100 else "IN PROGRESS"
    print(f"Status: {mastery_status}")
    
    # Save report
    report = {
        "level": "Pre-K",
        "date": datetime.now().isoformat(),
        "training_time": train_time,
        "nodes": len(lnn.nodes),
        "springs": len(lnn.springs),
        "total_score": total_score,
        "total_possible": total_possible,
        "mastery_pct": mastery_pct,
        "mastery_status": mastery_status,
        "assessments": {},
    }
    
    for name, result in results.items():
        report["assessments"][name] = {
            "score": result["score"],
            "possible": result["possible"],
            "pct": result["pct"],
            "details": result.get("details", []),
        }
    
    report_path = os.path.join(REPORT_DIR, f"prek_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {report_path}")
    
    return results


def test_letters(lnn):
    """Test letter recognition A-Z."""
    print("\n--- LETTERS (A-Z) ---")
    score = 0
    possible = 26
    details = []
    
    for letter in "abcdefghijklmnopqrstuvwxyz":
        query = f"word:{letter}"
        for n in lnn.nodes.values():
            n.activation = 0
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            propagate(lnn, n_steps=1)
            
            activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            # Check if uppercase version also activates
            upper_query = f"word:{letter.upper()}"
            if upper_query in lnn.nodes:
                lnn.nodes[upper_query].activation = 100
                propagate(lnn, n_steps=1)
                upper_activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                                  if node.activation > 0 and n.startswith("word:") and n != upper_query]
                if letter in upper_activated:
                    score += 1
                    details.append(f"{letter}: PASS")
                else:
                    details.append(f"{letter}: FAIL (no case link)")
            else:
                # At least the lowercase exists
                score += 0.5
                details.append(f"{letter}: PARTIAL (lowercase only)")
        else:
            details.append(f"{letter}: FAIL (not found)")
    
    pct = (score * 100) // possible
    print(f"  Score: {score}/{possible} ({pct}%)")
    return {"score": score, "possible": possible, "pct": pct, "details": details[:5]}


def test_numbers(lnn):
    """Test number recognition 1-10."""
    print("\n--- NUMBERS (1-10) ---")
    score = 0
    possible = 10
    details = []
    
    numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    
    for num in numbers:
        query = f"word:{num}"
        for n in lnn.nodes.values():
            n.activation = 0
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            propagate(lnn, n_steps=1)
            
            activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            # Check if other numbers activate (category clustering)
            other_nums = [n for n in numbers if n != num and n in activated]
            if len(other_nums) >= 2:
                score += 1
                details.append(f"{num}: PASS (clustered with {other_nums[:2]})")
            else:
                score += 0.5
                details.append(f"{num}: PARTIAL (exists but not clustered)")
        else:
            details.append(f"{num}: FAIL")
    
    pct = (score * 100) // possible
    print(f"  Score: {score}/{possible} ({pct}%)")
    return {"score": score, "possible": possible, "pct": pct, "details": details}


def test_colors(lnn):
    """Test color recognition."""
    print("\n--- COLORS ---")
    score = 0
    possible = 10
    details = []
    
    colors = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "brown", "pink"]
    
    for color in colors:
        query = f"word:{color}"
        for n in lnn.nodes.values():
            n.activation = 0
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            propagate(lnn, n_steps=1)
            
            activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            other_colors = [c for c in colors if c != color and c in activated]
            if len(other_colors) >= 2:
                score += 1
                details.append(f"{color}: PASS (clustered)")
            else:
                score += 0.5
                details.append(f"{color}: PARTIAL")
        else:
            details.append(f"{color}: FAIL")
    
    pct = (score * 100) // possible
    print(f"  Score: {score}/{possible} ({pct}%)")
    return {"score": score, "possible": possible, "pct": pct, "details": details}


def test_shapes(lnn):
    """Test shape recognition."""
    print("\n--- SHAPES ---")
    score = 0
    possible = 5
    details = []
    
    shapes = ["circle", "square", "triangle", "rectangle", "star"]
    
    for shape in shapes:
        query = f"word:{shape}"
        for n in lnn.nodes.values():
            n.activation = 0
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            propagate(lnn, n_steps=1)
            
            activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            other_shapes = [s for s in shapes if s != shape and s in activated]
            if len(other_shapes) >= 1:
                score += 1
                details.append(f"{shape}: PASS")
            else:
                score += 0.5
                details.append(f"{shape}: PARTIAL")
        else:
            details.append(f"{shape}: FAIL")
    
    pct = (score * 100) // possible
    print(f"  Score: {score}/{possible} ({pct}%)")
    return {"score": score, "possible": possible, "pct": pct, "details": details}


def test_animals(lnn):
    """Test animal recognition."""
    print("\n--- ANIMALS ---")
    score = 0
    possible = 10
    details = []
    
    animals = ["cat", "dog", "bird", "fish", "horse", "cow", "sheep", "pig", "chicken", "duck"]
    
    for animal in animals:
        query = f"word:{animal}"
        for n in lnn.nodes.values():
            n.activation = 0
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            propagate(lnn, n_steps=1)
            
            activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            other_animals = [a for a in animals if a != animal and a in activated]
            if len(other_animals) >= 2:
                score += 1
                details.append(f"{animal}: PASS (clustered with {len(other_animals)})")
            else:
                score += 0.5
                details.append(f"{animal}: PARTIAL")
        else:
            details.append(f"{animal}: FAIL")
    
    pct = (score * 100) // possible
    print(f"  Score: {score}/{possible} ({pct}%)")
    return {"score": score, "possible": possible, "pct": pct, "details": details}


def test_emotions(lnn):
    """Test emotion recognition."""
    print("\n--- EMOTIONS ---")
    score = 0
    possible = 6
    details = []
    
    emotions = ["happy", "sad", "angry", "scared", "surprised", "tired"]
    
    for emotion in emotions:
        query = f"word:{emotion}"
        for n in lnn.nodes.values():
            n.activation = 0
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            propagate(lnn, n_steps=1)
            
            activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            other_emotions = [e for e in emotions if e != emotion and e in activated]
            if len(other_emotions) >= 2:
                score += 1
                details.append(f"{emotion}: PASS")
            else:
                score += 0.5
                details.append(f"{emotion}: PARTIAL")
        else:
            details.append(f"{emotion}: FAIL")
    
    pct = (score * 100) // possible
    print(f"  Score: {score}/{possible} ({pct}%)")
    return {"score": score, "possible": possible, "pct": pct, "details": details}


def test_phonics(lnn):
    """Test phonics word family recognition."""
    print("\n--- PHONICS (CVC) ---")
    score = 0
    possible = 5
    details = []
    
    families = [
        ("cat", ["hat", "mat", "sat", "bat"]),
        ("dog", ["log", "fog", "hog", "jog"]),
        ("bed", ["red", "fed", "led"]),
        ("sun", ["run", "fun", "bun"]),
        ("big", ["pig", "dig", "wig"]),
    ]
    
    for word, family in families:
        query = f"word:{word}"
        for n in lnn.nodes.values():
            n.activation = 0
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            propagate(lnn, n_steps=1)
            
            activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            found = [w for w in family if w in activated]
            if len(found) >= 2:
                score += 1
                details.append(f"{word}: PASS (found {found[:2]})")
            else:
                score += 0.5
                details.append(f"{word}: PARTIAL (found {found})")
        else:
            details.append(f"{word}: FAIL")
    
    pct = (score * 100) // possible
    print(f"  Score: {score}/{possible} ({pct}%)")
    return {"score": score, "possible": possible, "pct": pct, "details": details}


def test_sentences(lnn):
    """Test simple sentence comprehension."""
    print("\n--- SIMPLE SENTENCES ---")
    score = 0
    possible = 5
    details = []
    
    tests = [
        ("word:cat", ["see", "big"]),
        ("word:dog", ["see", "small"]),
        ("word:bird", ["fly"]),
        ("word:fish", ["swim"]),
        ("word:sun", ["like"]),
    ]
    
    for query, expected in tests:
        for n in lnn.nodes.values():
            n.activation = 0
        
        if query in lnn.nodes:
            lnn.nodes[query].activation = 100
            propagate(lnn, n_steps=1)
            
            activated = [n.replace("word:", "") for n, node in lnn.nodes.items() 
                        if node.activation > 0 and n.startswith("word:") and n != query]
            
            found = [w for w in expected if w in activated]
            if found:
                score += 1
                details.append(f"{query.replace('word:', '')}: PASS ({found})")
            else:
                score += 0.5
                details.append(f"{query.replace('word:', '')}: PARTIAL")
        else:
            details.append(f"{query.replace('word:', '')}: FAIL")
    
    pct = (score * 100) // possible
    print(f"  Score: {score}/{possible} ({pct}%)")
    return {"score": score, "possible": possible, "pct": pct, "details": details}


if __name__ == "__main__":
    main()
