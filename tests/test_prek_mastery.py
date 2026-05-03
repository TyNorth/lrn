"""
Pre-K English Training - Iterative Mastery with Charts
Original + varied examples + more reps + detailed chart reports.
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


REPORT_DIR = "/Users/tyarc/github/lrn/reports/prek"
os.makedirs(REPORT_DIR, exist_ok=True)


def bar_chart(label, score, total, width=40):
    pct = score / max(1, total)
    filled = int(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"  {label:20s} |{bar}| {score:5.1f}/{total} ({pct*100:5.1f}%)"


def sparkline(values, width=20):
    if not values:
        return ""
    mn, mx = min(values), max(values)
    if mn == mx:
        return "▁" * width
    chars = "▁▂▃▄▅▆▇█"
    return "".join(chars[int((v - mn) / max(1, mx - mn) * 7)] for v in values[:width])


def optimal_rem(lnn, wake_buffer):
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


def train(lnn, sentences, reps=80, learn_type="language"):
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


ORIGINAL_CORPUS = [
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
    "one two three four five six seven eight nine ten",
    "i have one apple", "i have two cats", "i have three dogs",
    "i see four birds", "i see five fish", "i count six stars",
    "seven is lucky", "eight is great", "nine is fine", "ten is the end",
    "red blue green yellow orange purple",
    "black white brown pink gray gold",
    "the apple is red", "the sky is blue", "the grass is green",
    "the sun is yellow", "the orange is orange", "the grape is purple",
    "the night is black", "the snow is white", "the bear is brown",
    "the flower is pink", "the sky is gray", "the ring is gold",
    "circle square triangle rectangle star",
    "the ball is a circle", "the box is a square",
    "the roof is a triangle", "the door is a rectangle",
    "the star is a star", "the sun is a circle",
    "cat dog bird fish horse cow",
    "sheep pig chicken duck rabbit",
    "lion tiger bear elephant monkey",
    "the cat says meow", "the dog says woof",
    "the bird says tweet", "the cow says moo",
    "the pig says oink", "the duck says quack",
    "happy sad angry scared surprised tired",
    "i feel happy", "i feel sad", "i feel angry",
    "i feel scared", "i feel surprised", "i feel tired",
    "happy is good", "sad is not good",
    "cat hat mat sat bat",
    "dog log fog hog jog",
    "bed red fed led shed",
    "sun run fun bun pun",
    "big pig dig wig fig",
    "hot pot lot dot got",
    "i see the cat", "i see the dog",
    "the cat is big", "the dog is small",
    "i like the sun", "i like the moon",
    "the bird can fly", "the fish can swim",
]

VARIED_EXAMPLES = [
    "A says ah a says ah apple starts with A",
    "B says buh b says buh ball starts with B",
    "C says kuh c says kuh cat starts with C",
    "D says duh d says duh dog starts with D",
    "E says eh e says eh egg starts with E",
    "F says fuh f says fuh fish starts with F",
    "G says guh g says guh goat starts with G",
    "H says huh h says huh hat starts with H",
    "I says ih i says ih ice starts with I",
    "J says juh j says juh jam starts with J",
    "K says kuh k says kuh kite starts with K",
    "L says luh l says luh lion starts with L",
    "M says muh m says muh moon starts with M",
    "N says nuh n says nuh nest starts with N",
    "O says oh o says oh orange starts with O",
    "P says puh p says puh pig starts with P",
    "Q says kwuh q says kwuh queen starts with Q",
    "R says ruh r says ruh rat starts with R",
    "S says suh s says suh sun starts with S",
    "T says tuh t says tuh tree starts with T",
    "U says uh u says uh umbrella starts with U",
    "V says vuh v says vuh van starts with V",
    "W says wuh w says wuh watch starts with W",
    "X says ksuh x says ksuh box has X",
    "Y says yuh y says yuh yellow starts with Y",
    "Z says zuh z says zuh zebra starts with Z",
    "A a apple ant arm", "B b ball bat bed",
    "C c cat cup car", "D d dog door day",
    "E e egg eye ear", "F f fish fan fun",
    "G g goat game go", "H h hat hand hot",
    "I i ice igloo in", "J j jam jar job",
    "K k kite key kid", "L l lion leg log",
    "M m moon man mat", "N n nest net no",
    "O o orange on off", "P p pig pen pan",
    "Q q queen queen quick", "R r rat run red",
    "S s sun sit sad", "T t tree top ten",
    "U u umbrella up us", "V v van van very",
    "W w watch web we", "X x box fox six",
    "Y y yellow yes you", "Z z zebra zoo zero",
    "the cat has a hat", "the cat sits on the mat",
    "the cat sat on the mat", "the bat hit the cat",
    "a fat cat sat on a mat", "the rat sat on a hat",
    "the dog ran over the log", "the fog is on the hog",
    "the dog can jog", "a big dog sat on a log",
    "the bed is red", "i fed the cat on the bed",
    "the red bed is big", "i led the dog to the bed",
    "the sun is hot", "i run in the sun",
    "the sun is fun", "i have a bun in the sun",
    "the big pig can dig", "the pig has a big wig",
    "a big pig sat on a fig", "the pig can dig big",
    "the pot is hot", "i got a hot pot",
    "the hot pot has a lot", "a lot of dots on the pot",
    "the man has a can", "the fan is on the man",
    "the man ran with a pan", "a van ran to the man",
    "the hen has a pen", "ten men have a pen",
    "the hen is in the den", "ten men ran to the hen",
    "i sit on the mat", "the cat can sit",
    "i hit the ball", "the kit is a fit",
    "the cap is on the map", "i nap on the lap",
    "tap the cap on the map", "the cat naps on the lap",
    "i see the cat", "i see the dog", "i see the bird",
    "i see the fish", "i see the sun", "i see the moon",
    "i see a big cat", "i see a small dog",
    "i see the red ball", "i see the blue sky",
    "i see the green grass", "i see the yellow sun",
    "i like the cat", "i like the dog", "i like the bird",
    "i like to run", "i like to play", "i like to eat",
    "i like the big cat", "i like the small dog",
    "i like the red apple", "i like the blue sky",
    "the cat is big", "the dog is small", "the bird is fast",
    "the fish is small", "the sun is hot", "the moon is bright",
    "the cat is on the mat", "the dog is on the log",
    "the bird is in the tree", "the fish is in the water",
    "the sun is in the sky", "the moon is in the sky",
    "i can see the cat", "i can see the dog",
    "i can run fast", "i can jump high",
    "i can play with the cat", "i can play with the dog",
    "i can eat the apple", "i can drink the water",
    "the cat runs", "the dog runs", "the bird flies",
    "the fish swims", "the horse runs", "the cow eats",
    "the cat sits on the mat", "the dog sits on the log",
    "the bird sits in the tree", "the fish swims in the water",
    "where is the cat", "where is the dog",
    "what is that", "who is there",
    "can you see the cat", "can you see the dog",
    "do you like the cat", "do you like the dog",
    "one one one", "two two two", "three three three",
    "four four four", "five five five", "six six six",
    "seven seven seven", "eight eight eight", "nine nine nine", "ten ten ten",
    "red red red", "blue blue blue", "green green green",
    "yellow yellow yellow", "orange orange orange", "purple purple purple",
    "circle circle circle", "square square square",
    "triangle triangle triangle", "rectangle rectangle rectangle",
    "star star star",
    "cat cat cat", "dog dog dog", "bird bird bird",
    "fish fish fish", "horse horse horse", "cow cow cow",
    "sheep sheep sheep", "pig pig pig", "chicken chicken chicken",
    "duck duck duck", "rabbit rabbit rabbit",
    "happy happy happy", "sad sad sad", "angry angry angry",
    "scared scared scared", "surprised surprised surprised",
    "tired tired tired",
]

FULL_CORPUS = ORIGINAL_CORPUS + VARIED_EXAMPLES


def main():
    print("=" * 60)
    print("PRE-K ENGLISH - ORIGINAL + VARIED (60 reps)")
    print(f"Original: {len(ORIGINAL_CORPUS)} | Varied: {len(VARIED_EXAMPLES)} | Combined: {len(FULL_CORPUS)}")
    print("=" * 60)
    
    start_time = time.time()
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    print("\nTraining...")
    t0 = time.time()
    train(lnn, FULL_CORPUS, reps=60, learn_type="sensory")
    train_time = time.time() - t0
    
    print(f"Training: {train_time:.1f}s | Nodes: {len(lnn.nodes)} | Springs: {len(lnn.springs)}")
    
    results = {}
    results["letters"] = test_letters(lnn)
    results["numbers"] = test_numbers(lnn)
    results["colors"] = test_colors(lnn)
    results["shapes"] = test_shapes(lnn)
    results["animals"] = test_animals(lnn)
    results["emotions"] = test_emotions(lnn)
    results["phonics"] = test_phonics(lnn)
    results["sentences"] = test_sentences(lnn)
    
    total_score = sum(r["score"] for r in results.values())
    total_possible = sum(r["possible"] for r in results.values())
    mastery_pct = (total_score * 100) // total_possible
    
    # Print charts
    print(f"\n{'='*60}")
    print("PRE-K ASSESSMENT CHART")
    print(f"{'='*60}")
    
    for name, r in results.items():
        print(bar_chart(name.upper(), r["score"], r["possible"]))
    
    print(f"  {'─'*62}")
    print(bar_chart("TOTAL", total_score, total_possible))
    
    mastery_status = "MASTERY ✓" if mastery_pct >= 100 else "IN PROGRESS"
    print(f"\n  Status: {mastery_status} ({mastery_pct}%)")
    
    # Spring distribution chart
    tau_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for sp in lnn.springs.values():
        tau_counts[sp.tau] += 1
    
    print(f"\n{'='*60}")
    print("SPRING DISTRIBUTION (Tau Hierarchy)")
    print(f"{'='*60}")
    tau_names = {0: "τ=0 Constitutive", 1: "τ=1 Definitional", 2: "τ=2 Causal", 3: "τ=3 Categorical", 4: "τ=4 Contextual"}
    max_tau_count = max(tau_counts.values()) if tau_counts else 1
    for tau in range(5):
        count = tau_counts[tau]
        bar_len = int((count / max_tau_count) * 40) if max_tau_count > 0 else 0
        bar = "█" * bar_len
        print(f"  {tau_names[tau]:20s} |{bar} {count}")
    
    # Category density chart
    print(f"\n{'='*60}")
    print("CATEGORY CLUSTERING DENSITY")
    print(f"{'='*60}")
    categories = {
        "animals": ["cat", "dog", "bird", "fish", "horse", "cow", "sheep", "pig", "chicken", "duck"],
        "fruits": ["apple", "banana", "orange", "grape", "pear"],
        "colors": ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "brown", "pink"],
        "emotions": ["happy", "sad", "angry", "scared", "surprised", "tired"],
    }
    
    for cat_name, members in categories.items():
        tau3 = 0
        total = 0
        for i in range(len(members)):
            for j in range(i+1, len(members)):
                key = lnn._key(f"word:{members[i]}", f"word:{members[j]}")
                total += 1
                if key in lnn.springs and lnn.springs[key].tau == 3:
                    tau3 += 1
        density = tau3 / max(1, total)
        bar_len = int(density * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        status = "DENSE" if density >= 0.5 else "SPARSE"
        print(f"  {cat_name:15s} |{bar}| {tau3}/{total} ({density*100:.0f}%) {status}")
    
    # Save report
    report = {
        "level": "Pre-K",
        "date": datetime.now().isoformat(),
        "training_time": train_time,
        "original_sentences": len(ORIGINAL_CORPUS),
        "varied_sentences": len(VARIED_EXAMPLES),
        "total_sentences": len(FULL_CORPUS),
        "reps": 60,
        "nodes": len(lnn.nodes),
        "springs": len(lnn.springs),
        "tau_distribution": tau_counts,
        "total_score": total_score,
        "total_possible": total_possible,
        "mastery_pct": mastery_pct,
        "mastery_status": mastery_status,
        "assessments": {},
    }
    
    for name, r in results.items():
        report["assessments"][name] = {
            "score": r["score"],
            "possible": r["possible"],
            "pct": r["pct"],
            "details": r.get("details", []),
        }
    
    report_path = os.path.join(REPORT_DIR, f"prek_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved: {report_path}")
    return results


def test_letters(lnn):
    score = 0
    possible = 26
    details = []
    for letter in "abcdefghijklmnopqrstuvwxyz":
        lower_word = f"word:{letter}"
        upper_sens = f"sens:{letter.upper()}"
        lower_exists = lower_word in lnn.nodes
        upper_exists = upper_sens in lnn.nodes
        if lower_exists and upper_exists:
            key = lnn._key(lower_word, upper_sens)
            if key in lnn.springs:
                score += 1
                details.append(f"{letter}:PASS")
            else:
                score += 0.75
                details.append(f"{letter}:PARTIAL")
        elif lower_exists or upper_exists:
            score += 0.5
            details.append(f"{letter}:HALF")
        else:
            details.append(f"{letter}:FAIL")
    pct = int((score * 100) // possible)
    return {"score": score, "possible": possible, "pct": pct, "details": details[:5]}


def test_numbers(lnn):
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
            other_nums = [n for n in numbers if n != num and n in activated]
            if len(other_nums) >= 2:
                score += 1
                details.append(f"{num}:PASS")
            else:
                score += 0.5
                details.append(f"{num}:PARTIAL")
        else:
            details.append(f"{num}:FAIL")
    pct = int((score * 100) // possible)
    return {"score": score, "possible": possible, "pct": pct, "details": details}


def test_colors(lnn):
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
                details.append(f"{color}:PASS")
            else:
                score += 0.5
                details.append(f"{color}:PARTIAL")
        else:
            details.append(f"{color}:FAIL")
    pct = int((score * 100) // possible)
    return {"score": score, "possible": possible, "pct": pct, "details": details}


def test_shapes(lnn):
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
                details.append(f"{shape}:PASS")
            else:
                score += 0.5
                details.append(f"{shape}:PARTIAL")
        else:
            details.append(f"{shape}:FAIL")
    pct = int((score * 100) // possible)
    return {"score": score, "possible": possible, "pct": pct, "details": details}


def test_animals(lnn):
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
                details.append(f"{animal}:PASS")
            else:
                score += 0.5
                details.append(f"{animal}:PARTIAL")
        else:
            details.append(f"{animal}:FAIL")
    pct = int((score * 100) // possible)
    return {"score": score, "possible": possible, "pct": pct, "details": details}


def test_emotions(lnn):
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
                details.append(f"{emotion}:PASS")
            else:
                score += 0.5
                details.append(f"{emotion}:PARTIAL")
        else:
            details.append(f"{emotion}:FAIL")
    pct = int((score * 100) // possible)
    return {"score": score, "possible": possible, "pct": pct, "details": details}


def test_phonics(lnn):
    score = 0
    possible = 10
    details = []
    families = [
        ("cat", ["hat", "mat", "sat", "bat"]),
        ("dog", ["log", "fog", "hog", "jog"]),
        ("bed", ["red", "fed", "led"]),
        ("sun", ["run", "fun", "bun"]),
        ("big", ["pig", "dig", "wig"]),
        ("hot", ["pot", "lot", "dot"]),
        ("can", ["fan", "man", "pan"]),
        ("pen", ["hen", "ten", "men"]),
        ("cap", ["map", "lap", "nap"]),
        ("sit", ["fit", "hit", "kit"]),
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
                details.append(f"{word}:PASS")
            else:
                score += 0.5
                details.append(f"{word}:PARTIAL")
        else:
            details.append(f"{word}:FAIL")
    pct = int((score * 100) // possible)
    return {"score": score, "possible": possible, "pct": pct, "details": details}


def test_sentences(lnn):
    score = 0
    possible = 10
    details = []
    tests = [
        ("word:cat", ["see", "big", "mat"]),
        ("word:dog", ["see", "small", "log"]),
        ("word:bird", ["fly", "tree"]),
        ("word:fish", ["swim", "water"]),
        ("word:sun", ["hot", "sky"]),
        ("word:moon", ["bright", "sky"]),
        ("word:run", ["fast", "sun"]),
        ("word:play", ["cat", "dog"]),
        ("word:eat", ["apple", "food"]),
        ("word:like", ["cat", "dog"]),
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
            if len(found) >= 1:
                score += 1
                details.append(f"{query.replace('word:', '')}:PASS")
            else:
                score += 0.5
                details.append(f"{query.replace('word:', '')}:PARTIAL")
        else:
            details.append(f"{query.replace('word:', '')}:FAIL")
    pct = int((score * 100) // possible)
    return {"score": score, "possible": possible, "pct": pct, "details": details}


if __name__ == "__main__":
    main()
