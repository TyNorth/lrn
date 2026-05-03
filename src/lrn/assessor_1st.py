"""
1st Grade Curriculum Assessment - Detailed sub-skill breakdown

Based on Common Core 1st Grade standards.
Uses direct spring lookups for speed (no propagation per test item).
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')


def _has_spring(lnn, a, b):
    if a not in lnn.nodes or b not in lnn.nodes:
        return False
    key = lnn._key(a, b)
    return key in lnn.springs


def _node_exists(lnn, word):
    return f"word:{word}" in lnn.nodes


def _connection_count(lnn, word):
    node = f"word:{word}"
    if node not in lnn.nodes:
        return 0
    return sum(1 for (a, b) in lnn.springs if a == node or b == node)


def _has_any_connection(lnn, word):
    return _connection_count(lnn, word) >= 1


def _has_connections_to(lnn, word, targets):
    """Check if word has springs to any of the target words."""
    node = f"word:{word}"
    if node not in lnn.nodes:
        return False, []
    found = []
    for t in targets:
        if _has_spring(lnn, node, f"word:{t}"):
            found.append(t)
    return len(found) > 0, found


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


def assess_first_grade(lnn):
    """Full 1st Grade curriculum assessment."""
    results = {}
    
    # === LITERACY DOMAIN ===
    results["phonics_complex"] = test_phonics_complex(lnn)
    results["sight_words_1st"] = test_sight_words_1st(lnn)
    results["reading_fluency"] = test_reading_fluency_1st(lnn)
    results["writing_skills"] = test_writing_skills_1st(lnn)
    results["grammar"] = test_grammar(lnn)
    results["complex_sentences"] = test_complex_sentences(lnn)
    
    # === MATHEMATICS DOMAIN ===
    results["addition_within_20"] = test_addition_within_20(lnn)
    results["subtraction_within_20"] = test_subtraction_within_20(lnn)
    results["place_value_to_100"] = test_place_value_to_100(lnn)
    results["measurement"] = test_measurement_1st(lnn)
    results["money"] = test_money_1st(lnn)
    results["data_graphs"] = test_data_graphs_1st(lnn)
    results["geometry"] = test_geometry_1st(lnn)
    
    # === SCIENCE DOMAIN ===
    results["weather_patterns"] = test_weather_patterns(lnn)
    results["plant_animal_needs"] = test_plant_animal_needs(lnn)
    results["light_sound"] = test_light_sound(lnn)
    
    # === SOCIAL STUDIES ===
    results["communities"] = test_communities(lnn)
    results["maps"] = test_maps_1st(lnn)
    results["history"] = test_history(lnn)
    
    # === SOCIAL-EMOTIONAL ===
    results["character_traits"] = test_character_traits(lnn)
    
    return results


# === LITERACY ===

def test_phonics_complex(lnn):
    items = []
    patterns = [
        ("air", ["chair", "hair", "fair"]),
        ("ear", ["hear", "bear", "near"]),
        ("oi", ["coin", "join", "boil"]),
        ("ou", ["cloud", "loud", "proud"]),
        ("aw", ["saw", "paw", "draw"]),
        ("oo", ["moon", "spoon", "boot"]),
        ("ew", ["grew", "flew", "crew"]),
    ]
    for pattern, words in patterns:
        present = [w for w in words if _node_exists(lnn, w)]
        if len(present) >= 2:
            items.append((pattern, "PASS", f"{len(present)} words present"))
        else:
            items.append((pattern, "PARTIAL", f"{len(present)} words present"))
    return _make_result(items, "Literacy", "Complex Phonics Patterns")


def test_sight_words_1st(lnn):
    items = []
    sight_words = ["after", "again", "any", "ask", "away", "back", "been", "both", "buy", "call",
                   "came", "could", "does", "each", "even", "ever", "first", "found", "from", "give",
                   "goes", "going", "had", "has", "her", "him", "his", "just", "keep", "know",
                   "land", "large", "last", "let", "long", "look", "made", "make", "may", "many",
                   "might", "most", "my", "never", "next", "off", "old", "once", "open", "or",
                   "order", "other", "over", "own", "pick", "place", "pretty", "quite", "read",
                   "right", "round", "said", "saw", "say", "set", "shall", "show", "six", "small",
                   "start", "seven", "she", "soon", "sound", "spell", "stop", "such", "take",
                   "tell", "than", "that", "them", "then", "think", "this", "those", "thought",
                   "through", "under", "until", "up", "upon", "us", "use", "very", "walk", "want",
                   "warm", "was", "watch", "water", "way", "we", "well", "were", "what", "when",
                   "which", "while", "wish", "with", "work", "would", "write", "you"]
    for word in sight_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 2:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Literacy", "Sight Words (1st Grade)")


def test_reading_fluency_1st(lnn):
    items = []
    fluency_tests = [
        ("read", ["book", "story", "sentence"]),
        ("fiction", ["story", "made", "character"]),
        ("nonfiction", ["real", "facts", "true"]),
        ("author", ["wrote", "book", "story"]),
        ("character", ["main", "brave", "story"]),
        ("problem", ["solution", "happened", "story"]),
        ("paragraph", ["main", "idea", "sentence"]),
    ]
    for word, expected in fluency_tests:
        if _node_exists(lnn, word):
            found_any, found = _has_connections_to(lnn, word, expected)
            if found_any:
                items.append((word, "PASS", f"connects to {', '.join(found)}"))
            else:
                items.append((word, "PARTIAL", "exists but no expected connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Literacy", "Reading Fluency")


def test_writing_skills_1st(lnn):
    items = []
    writing_words = ["write", "sentence", "capital", "period", "question", "exclamation",
                     "opinion", "reason", "instructions", "describing", "action", "letter",
                     "dictionary", "spell", "sounding"]
    for word in writing_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Literacy", "Writing Skills")


def test_grammar(lnn):
    items = []
    grammar_tests = [
        ("noun", ["person", "place", "thing"]),
        ("verb", ["action", "run", "jump"]),
        ("adjective", ["describes", "big", "red"]),
        ("pronoun", ["he", "she", "they"]),
        ("subject", ["who", "what", "sentence"]),
    ]
    for word, expected in grammar_tests:
        if _node_exists(lnn, word):
            found_any, found = _has_connections_to(lnn, word, expected)
            if found_any:
                items.append((word, "PASS", f"connects to {', '.join(found)}"))
            else:
                items.append((word, "PARTIAL", "exists but no grammar connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Literacy", "Grammar")


def test_complex_sentences(lnn):
    items = []
    connectors = ["because", "although", "if", "while", "after", "before", "since", "unless"]
    for word in connectors:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Literacy", "Complex Sentences")


# === MATHEMATICS ===

def test_addition_within_20(lnn):
    items = []
    add_tests = [
        ("plus", ["add", "equals", "sum", "together"]),
        ("eleven", ["ten", "plus", "one"]),
        ("fifteen", ["ten", "plus", "five"]),
        ("twenty", ["ten", "plus", "ten"]),
        ("equals", ["is", "same", "answer"]),
    ]
    for word, expected in add_tests:
        if _node_exists(lnn, word):
            found_any, found = _has_connections_to(lnn, word, expected)
            if found_any:
                items.append((word, "PASS", f"connects to {', '.join(found)}"))
            else:
                items.append((word, "PARTIAL", "exists but no math connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Mathematics", "Addition Within 20")


def test_subtraction_within_20(lnn):
    items = []
    sub_tests = [
        ("minus", ["subtract", "equals", "difference"]),
        ("subtract", ["minus", "less", "away"]),
        ("take", ["away", "minus", "left"]),
        ("left", ["remain", "minus", "subtract"]),
    ]
    for word, expected in sub_tests:
        if _node_exists(lnn, word):
            found_any, found = _has_connections_to(lnn, word, expected)
            if found_any:
                items.append((word, "PASS", f"connects to {', '.join(found)}"))
            else:
                items.append((word, "PARTIAL", "exists but no math connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Mathematics", "Subtraction Within 20")


def test_place_value_to_100(lnn):
    items = []
    pv_words = ["tens", "ones", "greater", "less", "equals", "twenty", "thirty", "forty", "fifty",
                "sixty", "seventy", "eighty", "ninety", "hundred"]
    for word in pv_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Mathematics", "Place Value to 100")


def test_measurement_1st(lnn):
    items = []
    measure_words = ["measure", "length", "inches", "feet", "yardstick", "longer", "shorter",
                     "same", "time", "hour", "half", "o'clock", "thermometer", "temperature"]
    for word in measure_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Mathematics", "Measurement & Time")


def test_money_1st(lnn):
    items = []
    money_words = ["penny", "nickel", "dime", "quarter", "dollar", "cent", "cents",
                   "coins", "count", "total", "save", "buy", "worth"]
    for word in money_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Mathematics", "Money")


def test_data_graphs_1st(lnn):
    items = []
    graph_words = ["graph", "picture", "bar", "tally", "survey", "data", "categories",
                   "organized", "interpret", "questions", "more", "same", "favorite"]
    for word in graph_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Mathematics", "Data & Graphs")


def test_geometry_1st(lnn):
    items = []
    geo_words = ["triangle", "square", "rectangle", "circle", "hexagon", "sides", "corners",
                 "equal", "halves", "fourths", "divide", "shapes", "together"]
    for word in geo_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Mathematics", "Geometry")


# === SCIENCE ===

def test_weather_patterns(lnn):
    items = []
    weather_words = ["weather", "sunny", "cloudy", "rainy", "snowy", "windy", "seasons",
                     "spring", "summer", "fall", "winter", "temperature", "observe", "record",
                     "pattern", "changes"]
    for word in weather_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Weather Patterns")


def test_plant_animal_needs(lnn):
    items = []
    needs_words = ["plants", "animals", "sunlight", "water", "air", "food", "shelter",
                   "grow", "live", "habitat", "breathe", "fly", "swim", "baby", "parents",
                   "caterpillar", "butterfly"]
    for word in needs_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Plant & Animal Needs")


def test_light_sound(lnn):
    items = []
    ls_words = ["light", "sound", "straight", "reflected", "mirror", "clear", "opaque",
                "shadow", "blocked", "vibrate", "vibration", "loud", "quiet", "communicate",
                "signal", "flashlight", "drum"]
    for word in ls_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Light & Sound")


# === SOCIAL STUDIES ===

def test_communities(lnn):
    items = []
    comm_words = ["community", "rules", "safe", "jobs", "goods", "services", "teaching",
                  "healing", "money", "save", "leaders", "decisions", "vote", "choose",
                  "baker", "nurse", "teacher"]
    for word in comm_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Studies", "Communities")


def test_maps_1st(lnn):
    items = []
    map_words = ["map", "drawing", "above", "key", "symbols", "compass", "rose",
                 "north", "south", "east", "west", "classroom", "neighborhood",
                 "continent", "ocean", "earth", "seven", "five"]
    for word in map_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Studies", "Maps & Geography")


def test_history(lnn):
    items = []
    hist_words = ["history", "past", "present", "future", "today", "happened",
                  "photos", "stories", "important", "people", "changed", "world",
                  "holidays", "celebrate", "events", "family", "country"]
    for word in hist_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Studies", "History")


# === SOCIAL-EMOTIONAL ===

def test_character_traits(lnn):
    items = []
    trait_words = ["honesty", "truth", "respect", "kindly", "responsibility", "courage",
                   "brave", "kindness", "nice", "patience", "waiting", "calmly",
                   "perseverance", "giving", "fairness", "equally", "gratitude",
                   "thankful", "empathy", "understanding"]
    for word in trait_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social-Emotional", "Character Traits")
