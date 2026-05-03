"""
LRN Pre-K Curriculum Assessment - Detailed sub-skill breakdown

Modeled after real Pre-K assessment frameworks (DIBELS, TPRI, etc.)
Each domain broken into measurable sub-skills with mastery thresholds.
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import propagate


def assess_level(lnn, level_name):
    """Run all assessments for a level. Returns dict of results."""
    if level_name == "prek":
        return assess_prek(lnn)
    elif level_name == "kindergarten":
        from lrn.assessor_k import assess_kindergarten
        return assess_kindergarten(lnn)
    raise ValueError(f"Unknown level: {level_name}")


def assess_prek(lnn):
    """Full Pre-K curriculum assessment with sub-skill breakdown."""
    results = {}
    
    # === LITERACY DOMAIN ===
    results["letter_recognition"] = test_letter_recognition(lnn)
    results["letter_case_matching"] = test_letter_case_matching(lnn)
    results["phonological_awareness"] = test_phonological_awareness(lnn)
    results["print_awareness"] = test_print_awareness(lnn)
    results["sight_words"] = test_sight_words(lnn)
    
    # === MATHEMATICS DOMAIN ===
    results["number_recognition"] = test_number_recognition(lnn)
    results["counting_sequence"] = test_counting_sequence(lnn)
    results["quantity_correspondence"] = test_quantity_correspondence(lnn)
    results["comparison"] = test_comparison(lnn)
    results["patterns"] = test_patterns(lnn)
    results["position_words"] = test_position_words(lnn)
    results["sorting"] = test_sorting(lnn)
    
    # === SCIENCE/WORLD KNOWLEDGE ===
    results["colors"] = test_colors(lnn)
    results["shapes"] = test_shapes(lnn)
    results["animals"] = test_animals(lnn)
    results["body_parts"] = test_body_parts(lnn)
    results["five_senses"] = test_five_senses(lnn)
    results["weather"] = test_weather(lnn)
    results["seasons"] = test_seasons(lnn)
    results["plants"] = test_plants(lnn)
    results["living_nonliving"] = test_living_nonliving(lnn)
    results["time_concepts"] = test_time_concepts(lnn)
    
    # === SOCIAL STUDIES ===
    results["family"] = test_family(lnn)
    results["community"] = test_community(lnn)
    results["money"] = test_money(lnn)
    
    # === SOCIAL-EMOTIONAL ===
    results["emotions"] = test_emotions(lnn)
    results["social_skills"] = test_social_skills(lnn)
    results["health_safety"] = test_health_safety(lnn)
    
    # === LANGUAGE/COMMUNICATION ===
    results["vocabulary"] = test_vocabulary(lnn)
    results["sentence_comprehension"] = test_sentence_comprehension(lnn)
    
    # === PHYSICAL/ARTS ===
    results["movement"] = test_movement(lnn)
    results["music_art"] = test_music_art(lnn)
    
    return results


def _activate_and_collect(lnn, query_node, steps=1):
    """Activate a node and collect activated word neighbors."""
    for n in lnn.nodes.values():
        n.activation = 0
    if query_node not in lnn.nodes:
        return []
    lnn.nodes[query_node].activation = 100
    propagate(lnn, n_steps=steps)
    return [n.replace("word:", "") for n, node in lnn.nodes.items()
            if node.activation > 0 and n.startswith("word:") and n != query_node]


def _has_spring(lnn, a, b):
    """Check if two nodes have a spring."""
    if a not in lnn.nodes or b not in lnn.nodes:
        return False
    key = lnn._key(a, b)
    return key in lnn.springs


# ============================================================
# LITERACY DOMAIN
# ============================================================

def test_letter_recognition(lnn):
    """Can the lattice recognize each letter as a distinct concept?
    
    Sub-skills:
    - Letter node exists
    - Letter has connections to other letters (alphabet awareness)
    - Letter has connections to words starting with it
    """
    items = []
    for letter in "abcdefghijklmnopqrstuvwxyz":
        node = f"word:{letter}"
        exists = node in lnn.nodes
        
        # Check connections to other letters
        neighbors = _activate_and_collect(lnn, node) if exists else []
        letter_neighbors = [n for n in neighbors if len(n) == 1 and n.isalpha()]
        
        if exists and len(letter_neighbors) >= 2:
            items.append((letter, "PASS", "letter node + alphabet connections"))
        elif exists:
            items.append((letter, "PARTIAL", "letter node exists, weak alphabet links"))
        else:
            items.append((letter, "FAIL", "no letter node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = 26
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Literacy",
        "skill": "Letter Recognition",
    }


def test_letter_case_matching(lnn):
    """Can the lattice match uppercase to lowercase?
    
    Tests: A↔a, B↔b, etc. via spring connections.
    """
    items = []
    for letter in "abcdefghijklmnopqrstuvwxyz":
        lower = f"word:{letter}"
        upper = f"word:{letter.upper()}"
        
        if _has_spring(lnn, lower, upper):
            items.append((f"{letter.upper()}-{letter}", "PASS", "case pair connected"))
        elif lower in lnn.nodes or upper in lnn.nodes:
            items.append((f"{letter.upper()}-{letter}", "PARTIAL", "one case exists"))
        else:
            items.append((f"{letter.upper()}-{letter}", "FAIL", "no case nodes"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = 26
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Literacy",
        "skill": "Letter-Case Matching",
    }


def test_phonological_awareness(lnn):
    """Can the lattice detect rhyming and beginning sounds?
    
    Sub-skills:
    - Rhyme detection (cat↔hat, dog↔log)
    - Beginning sound awareness (cat↔car, dog↔door)
    - Word families (-at, -og, -ed, -un, -ig)
    """
    items = []
    
    # Rhyme families
    rhyme_families = [
        ("-at", ["cat", "hat", "mat", "bat", "sat"]),
        ("-og", ["dog", "log", "fog", "hog"]),
        ("-ed", ["bed", "red", "fed", "led"]),
        ("-un", ["sun", "run", "fun", "bun"]),
        ("-ig", ["big", "pig", "dig", "wig"]),
        ("-ot", ["hot", "pot", "lot", "dot"]),
        ("-an", ["can", "fan", "man", "pan"]),
        ("-en", ["pen", "hen", "ten", "men"]),
        ("-ap", ["cap", "map", "lap", "nap"]),
        ("-it", ["sit", "fit", "hit", "kit"]),
    ]
    
    for family_name, members in rhyme_families:
        # Check if any two members are connected
        connected = 0
        present = [m for m in members if f"word:{m}" in lnn.nodes]
        for i in range(len(present)):
            for j in range(i+1, len(present)):
                if _has_spring(lnn, f"word:{present[i]}", f"word:{present[j]}"):
                    connected += 1
        
        if connected >= 2:
            items.append((family_name, "PASS", f"{connected} rhyme connections"))
        elif connected >= 1:
            items.append((family_name, "PARTIAL", f"{connected} rhyme connection"))
        else:
            items.append((family_name, "FAIL", "no rhyme connections"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(items)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Literacy",
        "skill": "Phonological Awareness",
    }


def test_print_awareness(lnn):
    """Does the lattice understand print concepts?
    
    Tests: word↔letter connections, sentence structure awareness
    """
    items = []
    
    # Check if common words connect to their letters
    test_words = ["cat", "dog", "sun", "big", "red"]
    for word in test_words:
        word_node = f"word:{word}"
        if word_node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, word_node)
            letter_connections = [n for n in neighbors if n in word]
            if len(letter_connections) >= 2:
                items.append((word, "PASS", f"connects to {len(letter_connections)} letters"))
            else:
                items.append((word, "PARTIAL", "weak letter connections"))
        else:
            items.append((word, "FAIL", "word node missing"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(items)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Literacy",
        "skill": "Print Awareness",
    }


# ============================================================
# MATHEMATICS DOMAIN
# ============================================================

def test_number_recognition(lnn):
    """Can the lattice recognize numbers as distinct concepts?"""
    items = []
    numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    
    for num in numbers:
        node = f"word:{num}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            num_neighbors = [n for n in neighbors if n in numbers]
            if len(num_neighbors) >= 2:
                items.append((num, "PASS", f"connected to {', '.join(num_neighbors[:3])}"))
            else:
                items.append((num, "PARTIAL", "exists but isolated"))
        else:
            items.append((num, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(numbers)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Mathematics",
        "skill": "Number Recognition",
    }


def test_counting_sequence(lnn):
    """Does the lattice understand counting order?
    
    Tests: sequential connections (one→two→three...)
    """
    items = []
    sequence = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    
    for i in range(len(sequence) - 1):
        a, b = sequence[i], sequence[i+1]
        if _has_spring(lnn, f"word:{a}", f"word:{b}"):
            items.append((f"{a}→{b}", "PASS", "sequential connection"))
        elif f"word:{a}" in lnn.nodes and f"word:{b}" in lnn.nodes:
            items.append((f"{a}→{b}", "PARTIAL", "both exist, no direct link"))
        else:
            items.append((f"{a}→{b}", "FAIL", "missing nodes"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(items)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Mathematics",
        "skill": "Counting Sequence",
    }


def test_quantity_correspondence(lnn):
    """Does the lattice connect number words to quantities?
    
    Tests: number words connect to count-related concepts
    """
    items = []
    number_quantity_tests = [
        ("one", ["single", "first"]),
        ("two", ["pair", "second"]),
        ("three", ["third"]),
        ("five", ["hand", "fingers"]),
        ("ten", ["fingers", "toes"]),
    ]
    
    for num, quantities in number_quantity_tests:
        node = f"word:{num}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            found = [q for q in quantities if q in neighbors]
            if found:
                items.append((num, "PASS", f"connects to {', '.join(found)}"))
            else:
                items.append((num, "PARTIAL", "no quantity connections"))
        else:
            items.append((num, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(items)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Mathematics",
        "skill": "Quantity Correspondence",
    }


def test_comparison(lnn):
    """Does the lattice understand more/less, big/small?"""
    items = []
    comparison_pairs = [
        ("big", "small"),
        ("more", "less"),
        ("many", "few"),
        ("long", "short"),
        ("tall", "short"),
    ]
    
    for a, b in comparison_pairs:
        # They should be connected (same conceptual domain) but not identical
        a_node, b_node = f"word:{a}", f"word:{b}"
        if a_node in lnn.nodes and b_node in lnn.nodes:
            if _has_spring(lnn, a_node, b_node):
                items.append((f"{a}/{b}", "PASS", "comparison pair connected"))
            else:
                items.append((f"{a}/{b}", "PARTIAL", "both exist, no link"))
        else:
            items.append((f"{a}/{b}", "FAIL", "missing nodes"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(items)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Mathematics",
        "skill": "Comparison Concepts",
    }


# ============================================================
# SCIENCE/WORLD KNOWLEDGE
# ============================================================

def test_colors(lnn):
    """Color naming and categorization."""
    items = []
    colors = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "brown", "pink"]
    
    for color in colors:
        node = f"word:{color}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            color_neighbors = [c for c in colors if c != color and c in neighbors]
            if len(color_neighbors) >= 2:
                items.append((color, "PASS", f"connected to {', '.join(color_neighbors[:3])}"))
            else:
                items.append((color, "PARTIAL", "exists but isolated"))
        else:
            items.append((color, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(colors)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Science",
        "skill": "Color Knowledge",
    }


def test_shapes(lnn):
    """Shape naming and properties."""
    items = []
    shapes = ["circle", "square", "triangle", "rectangle", "star"]
    
    for shape in shapes:
        node = f"word:{shape}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            shape_neighbors = [s for s in shapes if s != shape and s in neighbors]
            if shape_neighbors:
                items.append((shape, "PASS", f"connected to {', '.join(shape_neighbors)}"))
            else:
                items.append((shape, "PARTIAL", "exists but isolated"))
        else:
            items.append((shape, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(shapes)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Science",
        "skill": "Shape Knowledge",
    }


def test_animals(lnn):
    """Animal naming and categorization."""
    items = []
    animals = ["cat", "dog", "bird", "fish", "horse", "cow", "sheep", "pig", "chicken", "duck"]
    
    for animal in animals:
        node = f"word:{animal}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            animal_neighbors = [a for a in animals if a != animal and a in neighbors]
            if len(animal_neighbors) >= 2:
                items.append((animal, "PASS", f"connected to {', '.join(animal_neighbors[:3])}"))
            else:
                items.append((animal, "PARTIAL", "exists but isolated"))
        else:
            items.append((animal, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(animals)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Science",
        "skill": "Animal Knowledge",
    }


def test_body_parts(lnn):
    """Body part naming and categorization."""
    items = []
    body_parts = ["head", "eyes", "nose", "mouth", "ears", "hands", "feet", "arms", "legs", "fingers"]
    
    for part in body_parts:
        node = f"word:{part}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            part_neighbors = [p for p in body_parts if p != part and p in neighbors]
            if part_neighbors:
                items.append((part, "PASS", f"connected to {', '.join(part_neighbors[:2])}"))
            else:
                items.append((part, "PARTIAL", "exists but isolated"))
        else:
            items.append((part, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(body_parts)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Science",
        "skill": "Body Parts",
    }


# ============================================================
# SOCIAL-EMOTIONAL
# ============================================================

def test_emotions(lnn):
    """Emotion naming and categorization."""
    items = []
    emotions = ["happy", "sad", "angry", "scared", "surprised", "tired"]
    
    for emotion in emotions:
        node = f"word:{emotion}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            emotion_neighbors = [e for e in emotions if e != emotion and e in neighbors]
            if len(emotion_neighbors) >= 2:
                items.append((emotion, "PASS", f"connected to {', '.join(emotion_neighbors[:2])}"))
            else:
                items.append((emotion, "PARTIAL", "exists but isolated"))
        else:
            items.append((emotion, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(emotions)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Social-Emotional",
        "skill": "Emotion Knowledge",
    }


# ============================================================
# LANGUAGE/COMMUNICATION
# ============================================================

def test_vocabulary(lnn):
    """General vocabulary breadth."""
    items = []
    vocab_words = ["apple", "ball", "book", "house", "tree", "water", "food", "sleep", "walk", "talk"]
    
    for word in vocab_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 2:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(vocab_words)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Language",
        "skill": "Vocabulary",
    }


def test_sentence_comprehension(lnn):
    """Sentence-level understanding."""
    items = []
    tests = [
        ("cat", ["see", "big", "mat"]),
        ("dog", ["see", "small", "log"]),
        ("bird", ["fly", "tree"]),
        ("fish", ["swim", "water"]),
        ("sun", ["hot", "sky"]),
        ("moon", ["bright", "sky"]),
        ("run", ["fast", "sun"]),
        ("play", ["cat", "dog"]),
        ("eat", ["apple", "food"]),
        ("like", ["cat", "dog"]),
    ]
    
    for word, expected in tests:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            found = [w for w in expected if w in neighbors]
            if found:
                items.append((word, "PASS", f"connects to {', '.join(found)}"))
            else:
                items.append((word, "PARTIAL", "exists but no expected connections"))
        else:
            items.append((word, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(tests)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Language",
        "skill": "Sentence Comprehension",
    }


def test_sight_words(lnn):
    """Common Pre-K sight words."""
    items = []
    sight_words = ["i", "the", "a", "is", "see", "like", "can", "my", "me", "we", "you", "he", "she", "it", "they", "this", "that", "in", "on", "to"]
    
    for word in sight_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 2:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(sight_words)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Literacy",
        "skill": "Sight Words",
    }


def test_patterns(lnn):
    """Pattern recognition (AB, AAB, ABC)."""
    items = []
    pattern_tests = [
        ("red", "blue"),
        ("big", "small"),
        ("one", "two"),
        ("cat", "dog"),
        ("circle", "square"),
    ]
    
    for a, b in pattern_tests:
        a_node, b_node = f"word:{a}", f"word:{b}"
        if a_node in lnn.nodes and b_node in lnn.nodes:
            if _has_spring(lnn, a_node, b_node):
                items.append((f"{a}-{b}", "PASS", "pattern pair connected"))
            else:
                items.append((f"{a}-{b}", "PARTIAL", "both exist, no link"))
        else:
            items.append((f"{a}-{b}", "FAIL", "missing nodes"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(items)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Mathematics",
        "skill": "Patterns",
    }


def test_position_words(lnn):
    """Position/spatial words."""
    items = []
    positions = ["on", "under", "in", "beside", "between", "above", "below", "next", "behind", "front"]
    
    for word in positions:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(positions)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Mathematics",
        "skill": "Position Words",
    }


def test_sorting(lnn):
    """Sorting and classifying concepts."""
    items = []
    sort_concepts = ["sort", "color", "shape", "size", "type", "together", "same", "different", "group", "match"]
    
    for word in sort_concepts:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(sort_concepts)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Mathematics",
        "skill": "Sorting & Classifying",
    }


def test_five_senses(lnn):
    """Five senses awareness."""
    items = []
    senses = [
        ("see", "eyes"),
        ("hear", "ears"),
        ("smell", "nose"),
        ("taste", "tongue"),
        ("touch", "hands"),
    ]
    
    for sense, body_part in senses:
        sense_node = f"word:{sense}"
        body_node = f"word:{body_part}"
        if sense_node in lnn.nodes and body_node in lnn.nodes:
            if _has_spring(lnn, sense_node, body_node):
                items.append((f"{sense}-{body_part}", "PASS", "sense-body connection"))
            else:
                items.append((f"{sense}-{body_part}", "PARTIAL", "both exist, no link"))
        else:
            items.append((f"{sense}-{body_part}", "FAIL", "missing nodes"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(items)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Science",
        "skill": "Five Senses",
    }


def test_weather(lnn):
    """Weather concepts."""
    items = []
    weather_words = ["sunny", "cloudy", "rainy", "snowy", "windy", "stormy", "rain", "snow", "wind", "cloud"]
    
    for word in weather_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(weather_words)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Science",
        "skill": "Weather",
    }


def test_seasons(lnn):
    """Season awareness."""
    items = []
    seasons = ["spring", "summer", "fall", "winter", "hot", "cold", "warm", "cool"]
    
    for word in seasons:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(seasons)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Science",
        "skill": "Seasons",
    }


def test_plants(lnn):
    """Plant parts and growth."""
    items = []
    plant_words = ["seed", "root", "stem", "leaf", "flower", "tree", "grow", "plant", "water", "sun"]
    
    for word in plant_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(plant_words)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Science",
        "skill": "Plants & Growing",
    }


def test_living_nonliving(lnn):
    """Living vs non-living distinction."""
    items = []
    living = ["cat", "dog", "bird", "tree", "flower", "fish"]
    nonliving = ["rock", "toy", "car", "book", "ball", "box"]
    
    for word in living + nonliving:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(living) + len(nonliving)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Science",
        "skill": "Living vs Non-Living",
    }


def test_time_concepts(lnn):
    """Time awareness (morning/night, days, today/tomorrow)."""
    items = []
    time_words = ["morning", "afternoon", "night", "today", "tomorrow", "yesterday", "monday", "friday", "saturday", "sunday"]
    
    for word in time_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(time_words)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Science",
        "skill": "Time Concepts",
    }


def test_family(lnn):
    """Family members and relationships."""
    items = []
    family_words = ["mom", "dad", "sister", "brother", "baby", "grandma", "grandpa", "family", "love", "care"]
    
    for word in family_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(family_words)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Social Studies",
        "skill": "Family",
    }


def test_community(lnn):
    """Community helpers and places."""
    items = []
    community_words = ["teacher", "doctor", "firefighter", "police", "librarian", "farmer", "school", "store", "park", "library"]
    
    for word in community_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(community_words)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Social Studies",
        "skill": "Community",
    }


def test_money(lnn):
    """Basic money recognition."""
    items = []
    money_words = ["penny", "nickel", "dime", "quarter", "cent", "money", "save", "buy", "coin", "count"]
    
    for word in money_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(money_words)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Social Studies",
        "skill": "Money",
    }


def test_social_skills(lnn):
    """Social skills and self-regulation."""
    items = []
    social_words = ["share", "help", "wait", "turn", "please", "thank", "sorry", "kind", "gentle", "listen"]
    
    for word in social_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(social_words)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Social-Emotional",
        "skill": "Social Skills",
    }


def test_health_safety(lnn):
    """Health and safety awareness."""
    items = []
    health_words = ["wash", "hands", "brush", "teeth", "helmet", "seatbelt", "stop", "safe", "careful", "clean"]
    
    for word in health_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(health_words)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Social-Emotional",
        "skill": "Health & Safety",
    }


def test_movement(lnn):
    """Gross motor movement concepts."""
    items = []
    movement_words = ["run", "jump", "hop", "skip", "climb", "throw", "catch", "kick", "balance", "crawl"]
    
    for word in movement_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(movement_words)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Physical",
        "skill": "Movement",
    }


def test_music_art(lnn):
    """Music and art concepts."""
    items = []
    art_words = ["sing", "dance", "clap", "stomp", "draw", "paint", "cut", "paste", "color", "music"]
    
    for word in art_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(art_words)
    
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": "Arts",
        "skill": "Music & Art",
    }
