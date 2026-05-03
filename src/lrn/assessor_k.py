"""
Kindergarten Curriculum Assessment - Detailed sub-skill breakdown

Based on Common Core Kindergarten standards.
Covers: Literacy (phonics/reading/writing), Mathematics (operations/place value),
Science (life cycles/habitats), Social Studies (maps/citizenship), Language.
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import propagate


def _activate_and_collect(lnn, query_node, steps=1):
    for n in lnn.nodes.values():
        n.activation = 0
    if query_node not in lnn.nodes:
        return []
    lnn.nodes[query_node].activation = 100
    propagate(lnn, n_steps=steps)
    return [n.replace("word:", "") for n, node in lnn.nodes.items()
            if node.activation > 0 and n.startswith("word:") and n != query_node]


def _has_spring(lnn, a, b):
    if a not in lnn.nodes or b not in lnn.nodes:
        return False
    key = lnn._key(a, b)
    return key in lnn.springs


def assess_kindergarten(lnn):
    """Full Kindergarten curriculum assessment."""
    results = {}
    
    # === LITERACY DOMAIN ===
    results["phonics_blends"] = test_phonics_blends(lnn)
    results["long_vowels"] = test_long_vowels(lnn)
    results["sight_words"] = test_k_sight_words(lnn)
    results["reading_fluency"] = test_reading_fluency(lnn)
    results["writing_skills"] = test_writing_skills(lnn)
    results["story_elements"] = test_story_elements(lnn)
    results["questioning"] = test_questioning(lnn)
    results["vocabulary_expansion"] = test_vocabulary_expansion(lnn)
    
    # === MATHEMATICS DOMAIN ===
    results["addition"] = test_addition(lnn)
    results["subtraction"] = test_subtraction(lnn)
    results["counting_to_100"] = test_counting_to_100(lnn)
    results["place_value"] = test_place_value(lnn)
    results["measurement"] = test_measurement(lnn)
    results["shapes_3d"] = test_shapes_3d(lnn)
    results["patterns"] = test_k_patterns(lnn)
    results["graphs_data"] = test_graphs_data(lnn)
    
    # === SCIENCE DOMAIN ===
    results["life_cycles"] = test_life_cycles(lnn)
    results["habitats"] = test_habitats(lnn)
    results["earth_science"] = test_earth_science(lnn)
    
    # === SOCIAL STUDIES ===
    results["maps_geography"] = test_maps_geography(lnn)
    results["citizenship"] = test_citizenship(lnn)
    results["holidays_traditions"] = test_holidays_traditions(lnn)
    
    # === SOCIAL-EMOTIONAL ===
    results["self_regulation"] = test_self_regulation(lnn)
    
    return results


def _make_result(items, domain, skill):
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(items)
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible),
        "items": items,
        "domain": domain,
        "skill": skill,
    }


# === LITERACY ===

def test_phonics_blends(lnn):
    items = []
    blends = [
        ("bl", ["blue", "black", "block"]),
        ("sh", ["ship", "shop", "fish"]),
        ("ch", ["chip", "chop", "lunch"]),
        ("th", ["this", "that", "path"]),
        ("ck", ["back", "duck", "sock"]),
        ("ng", ["sing", "ring", "long"]),
        ("tr", ["train", "tree", "truck"]),
        ("dr", ["drum", "drop", "dress"]),
    ]
    for blend, words in blends:
        present = [w for w in words if f"word:{w}" in lnn.nodes]
        if len(present) >= 2:
            connected = sum(1 for i in range(len(present)) for j in range(i+1, len(present)) if _has_spring(lnn, f"word:{present[i]}", f"word:{present[j]}"))
            if connected >= 1:
                items.append((blend, "PASS", f"{connected} connections"))
            else:
                items.append((blend, "PARTIAL", "words exist, no connections"))
        else:
            items.append((blend, "FAIL", "words missing"))
    return _make_result(items, "Literacy", "Phonics Blends & Digraphs")


def test_long_vowels(lnn):
    items = []
    patterns = [
        ("a_e", ["cake", "make", "lake"]),
        ("i_e", ["bike", "like", "time"]),
        ("ai", ["rain", "train", "mail"]),
        ("ee", ["see", "tree", "bee"]),
        ("oa", ["boat", "coat", "road"]),
        ("igh", ["night", "light", "right"]),
    ]
    for pattern, words in patterns:
        present = [w for w in words if f"word:{w}" in lnn.nodes]
        if len(present) >= 2:
            items.append((pattern, "PASS", f"{len(present)} words present"))
        else:
            items.append((pattern, "PARTIAL", f"{len(present)} words present"))
    return _make_result(items, "Literacy", "Long Vowel Patterns")


def test_k_sight_words(lnn):
    items = []
    sight_words = ["all", "are", "but", "came", "did", "eat", "eight", "four", "get", "has",
                   "have", "how", "into", "like", "many", "must", "new", "nine", "now", "on",
                   "our", "out", "please", "pretty", "ran", "ride", "saw", "seven", "six", "so",
                   "soon", "that", "there", "they", "this", "too", "under", "want", "was", "well",
                   "went", "what", "who", "will", "with", "yes"]
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
    return _make_result(items, "Literacy", "Sight Words (Primer)")


def test_reading_fluency(lnn):
    items = []
    sentence_tests = [
        ("read", ["book", "sentence", "word"]),
        ("write", ["name", "sentence", "pencil"]),
        ("spell", ["cat", "dog", "sun"]),
        ("story", ["beginning", "middle", "end"]),
        ("book", ["read", "page", "picture"]),
    ]
    for word, expected in sentence_tests:
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
    return _make_result(items, "Literacy", "Reading Fluency")


def test_writing_skills(lnn):
    items = []
    writing_words = ["write", "pencil", "paper", "name", "letter", "capital", "lowercase", "spell", "sentence", "left", "right", "top"]
    for word in writing_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Literacy", "Writing Skills")


def test_story_elements(lnn):
    items = []
    story_words = ["beginning", "middle", "end", "character", "setting", "problem", "solution", "story", "tell", "retell"]
    for word in story_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Literacy", "Story Elements")


def test_questioning(lnn):
    items = []
    question_words = ["who", "what", "where", "when", "why", "how", "question", "answer", "ask", "learn"]
    for word in question_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Literacy", "Questioning Skills")


def test_vocabulary_expansion(lnn):
    items = []
    vocab_words = ["enormous", "tiny", "rapid", "ancient", "furious", "delighted", "exhausted", "frightened", "synonym", "antonym"]
    for word in vocab_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Literacy", "Vocabulary Expansion")


# === MATHEMATICS ===

def test_addition(lnn):
    items = []
    add_tests = [
        ("plus", ["add", "equals", "sum"]),
        ("add", ["plus", "more", "together"]),
        ("equals", ["is", "same", "answer"]),
        ("one", ["two", "plus", "add"]),
        ("two", ["three", "plus", "one"]),
        ("five", ["ten", "plus", "equals"]),
    ]
    for word, expected in add_tests:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            found = [w for w in expected if w in neighbors]
            if found:
                items.append((word, "PASS", f"connects to {', '.join(found)}"))
            else:
                items.append((word, "PARTIAL", "exists but no math connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Mathematics", "Addition Concepts")


def test_subtraction(lnn):
    items = []
    sub_tests = [
        ("minus", ["subtract", "equals", "difference"]),
        ("subtract", ["minus", "less", "away"]),
        ("take", ["away", "minus", "left"]),
        ("left", ["remain", "minus", "subtract"]),
    ]
    for word, expected in sub_tests:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            found = [w for w in expected if w in neighbors]
            if found:
                items.append((word, "PASS", f"connects to {', '.join(found)}"))
            else:
                items.append((word, "PARTIAL", "exists but no math connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Mathematics", "Subtraction Concepts")


def test_counting_to_100(lnn):
    items = []
    count_words = ["ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred"]
    for word in count_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Mathematics", "Counting to 100")


def test_place_value(lnn):
    items = []
    pv_words = ["tens", "ones", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "twenty", "thirty", "hundred"]
    for word in pv_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Mathematics", "Place Value")


def test_measurement(lnn):
    items = []
    measure_words = ["measure", "ruler", "long", "short", "longer", "heavy", "light", "more", "less", "weigh"]
    for word in measure_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Mathematics", "Measurement")


def test_shapes_3d(lnn):
    items = []
    shapes = ["cube", "sphere", "cylinder", "cone", "face", "edge", "corner", "round", "flat", "solid"]
    for word in shapes:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Mathematics", "3D Shapes")


def test_k_patterns(lnn):
    items = []
    pattern_tests = [
        ("pattern", ["repeat", "next", "sequence"]),
        ("repeat", ["again", "same", "pattern"]),
        ("next", ["comes", "what", "pattern"]),
    ]
    for word, expected in pattern_tests:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            found = [w for w in expected if w in neighbors]
            if found:
                items.append((word, "PASS", f"connects to {', '.join(found)}"))
            else:
                items.append((word, "PARTIAL", "exists but no pattern connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Mathematics", "Patterns (K)")


def test_graphs_data(lnn):
    items = []
    graph_words = ["graph", "tally", "survey", "count", "data", "bar", "chart", "compare", "show", "many"]
    for word in graph_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Mathematics", "Graphs & Data")


# === SCIENCE ===

def test_life_cycles(lnn):
    items = []
    lc_words = ["seed", "grow", "plant", "caterpillar", "butterfly", "tadpole", "frog", "egg", "chick", "cycle"]
    for word in lc_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Life Cycles")


def test_habitats(lnn):
    items = []
    hab_words = ["habitat", "forest", "ocean", "desert", "arctic", "jungle", "shelter", "water", "food", "home"]
    for word in hab_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Habitats")


def test_earth_science(lnn):
    items = []
    earth_words = ["rock", "soil", "water", "mountain", "valley", "river", "ocean", "sun", "moon", "star"]
    for word in earth_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Earth Science")


# === SOCIAL STUDIES ===

def test_maps_geography(lnn):
    items = []
    map_words = ["map", "globe", "north", "south", "east", "west", "continent", "ocean", "town", "country"]
    for word in map_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Studies", "Maps & Geography")


def test_citizenship(lnn):
    items = []
    civics_words = ["citizen", "vote", "rule", "community", "recycle", "save", "leader", "flag", "president", "mayor"]
    for word in civics_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Studies", "Citizenship")


def test_holidays_traditions(lnn):
    items = []
    holiday_words = ["holiday", "celebrate", "tradition", "family", "thanksgiving", "new", "year", "valentine", "share", "special"]
    for word in holiday_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Studies", "Holidays & Traditions")


# === SOCIAL-EMOTIONAL ===

def test_self_regulation(lnn):
    items = []
    sr_words = ["calm", "breathe", "count", "wait", "try", "again", "fail", "solve", "problem", "feel"]
    for word in sr_words:
        node = f"word:{word}"
        if node in lnn.nodes:
            neighbors = _activate_and_collect(lnn, node)
            if len(neighbors) >= 1:
                items.append((word, "PASS", f"{len(neighbors)} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social-Emotional", "Self-Regulation")
