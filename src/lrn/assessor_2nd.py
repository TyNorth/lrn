"""
2nd Grade Curriculum Assessment - Detailed sub-skill breakdown

Based on Common Core 2nd Grade standards.
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
    return len(lnn.get_neighbors(node))


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


def assess_second_grade(lnn):
    """Full 2nd Grade curriculum assessment."""
    results = {}
    
    # === LITERACY DOMAIN ===
    results["phonics_advanced"] = test_phonics_advanced(lnn)
    results["sight_words_2nd"] = test_sight_words_2nd(lnn)
    results["reading_comprehension"] = test_reading_comprehension_2nd(lnn)
    results["writing_skills"] = test_writing_skills_2nd(lnn)
    results["grammar"] = test_grammar_2nd(lnn)
    results["sentence_structure"] = test_sentence_structure_2nd(lnn)
    
    # === MATHEMATICS DOMAIN ===
    results["addition_subtraction_100"] = test_addition_subtraction_100(lnn)
    results["place_value_1000"] = test_place_value_1000(lnn)
    results["measurement"] = test_measurement_2nd(lnn)
    results["money"] = test_money_2nd(lnn)
    results["data_graphs"] = test_data_graphs_2nd(lnn)
    results["geometry"] = test_geometry_2nd(lnn)
    
    # === SCIENCE DOMAIN ===
    results["ecosystems"] = test_ecosystems(lnn)
    results["earth_systems"] = test_earth_systems(lnn)
    results["engineering"] = test_engineering(lnn)
    
    # === SOCIAL STUDIES ===
    results["citizenship"] = test_citizenship(lnn)
    results["economics"] = test_economics(lnn)
    results["geography"] = test_geography_2nd(lnn)
    results["history"] = test_history_2nd(lnn)
    
    # === SOCIAL-EMOTIONAL ===
    results["character_traits"] = test_character_traits_2nd(lnn)
    
    return results


# === LITERACY ===

def test_phonics_advanced(lnn):
    items = []
    patterns = [
        ("tion", ["action", "nation", "station"]),
        ("ture", ["picture", "nature", "future"]),
        ("cious", ["delicious", "precious", "spacious"]),
        ("able", ["comfortable", "remarkable", "valuable"]),
        ("ful", ["beautiful", "wonderful", "careful"]),
        ("ness", ["kindness", "happiness", "darkness"]),
        ("ment", ["movement", "excitement", "development"]),
    ]
    for pattern, words in patterns:
        present = [w for w in words if _node_exists(lnn, w)]
        if len(present) >= 2:
            items.append((pattern, "PASS", f"{len(present)} words present"))
        else:
            items.append((pattern, "PARTIAL", f"{len(present)} words present"))
    return _make_result(items, "Literacy", "Advanced Phonics Patterns")


def test_sight_words_2nd(lnn):
    items = []
    sight_words = ["always", "around", "because", "been", "before", "best", "both", "bring",
                   "build", "carry", "clean", "cut", "done", "draw", "drink", "eight", "else",
                   "fall", "far", "full", "got", "grow", "hold", "hot", "if", "keep", "kind",
                   "laugh", "light", "long", "much", "myself", "never", "only", "own", "people",
                   "picture", "plant", "pull", "read", "ready", "real", "right", "round", "seem",
                   "sense", "separate", "seven", "shall", "together", "try", "upon", "warm",
                   "why", "wish", "would"]
    for word in sight_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 2:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Literacy", "Sight Words (2nd Grade)")


def test_reading_comprehension_2nd(lnn):
    items = []
    comprehension_tests = [
        ("main", ["idea", "story", "about"]),
        ("character", ["brave", "story", "describe"]),
        ("setting", ["story", "place", "time"]),
        ("plot", ["happens", "story", "problem"]),
        ("solution", ["problem", "end", "answer"]),
        ("lesson", ["moral", "story", "learn"]),
        ("fantasy", ["cannot", "really", "happen"]),
        ("realistic", ["could", "really", "happen"]),
        ("context", ["clues", "figure", "word"]),
    ]
    for word, expected in comprehension_tests:
        if _node_exists(lnn, word):
            found_any, found = _has_connections_to(lnn, word, expected)
            if found_any:
                items.append((word, "PASS", f"connects to {', '.join(found)}"))
            else:
                items.append((word, "PARTIAL", "exists but no expected connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Literacy", "Reading Comprehension")


def test_writing_skills_2nd(lnn):
    items = []
    writing_words = ["paragraph", "topic", "sentence", "opinion", "reason", "report",
                     "research", "narrative", "sequence", "transition", "first", "next",
                     "finally", "conclusion", "revise", "edit", "thesaurus", "letter",
                     "greeting", "closing"]
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


def test_grammar_2nd(lnn):
    items = []
    grammar_tests = [
        ("noun", ["common", "proper", "person", "place"]),
        ("verb", ["action", "linking", "does", "is"]),
        ("adjective", ["describes", "noun", "big", "red"]),
        ("adverb", ["quickly", "slowly", "how", "when"]),
        ("conjunction", ["and", "but", "or", "joins"]),
        ("interjection", ["wow", "ouch", "feeling"]),
        ("pronoun", ["he", "she", "they", "replaces"]),
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


def test_sentence_structure_2nd(lnn):
    items = []
    structure_words = ["simple", "compound", "subject", "predicate", "statement",
                       "question", "command", "exclamation", "comma", "thought"]
    for word in structure_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Literacy", "Sentence Structure")


# === MATHEMATICS ===

def test_addition_subtraction_100(lnn):
    items = []
    add_sub_tests = [
        ("plus", ["add", "equals", "sum", "together"]),
        ("minus", ["subtract", "equals", "difference", "less"]),
        ("regroup", ["carry", "borrow", "ten", "place"]),
        ("mental", ["math", "add", "subtract", "quick"]),
        ("check", ["answer", "opposite", "operation", "verify"]),
        ("twenty", ["thirty", "forty", "fifty", "sixty"]),
        ("seventy", ["eighty", "ninety", "hundred"]),
    ]
    for word, expected in add_sub_tests:
        if _node_exists(lnn, word):
            found_any, found = _has_connections_to(lnn, word, expected)
            if found_any:
                items.append((word, "PASS", f"connects to {', '.join(found)}"))
            else:
                items.append((word, "PARTIAL", "exists but no math connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Mathematics", "Addition & Subtraction Within 100")


def test_place_value_1000(lnn):
    items = []
    pv_words = ["hundreds", "tens", "ones", "thousand", "greater", "less",
                "count", "read", "write", "number", "digit", "value"]
    for word in pv_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Mathematics", "Place Value to 1000")


def test_measurement_2nd(lnn):
    items = []
    measure_words = ["centimeter", "meters", "estimate", "length", "longer", "shorter",
                     "time", "nearest", "minute", "quarter", "past", "half", "a.m.", "p.m."]
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


def test_money_2nd(lnn):
    items = []
    money_words = ["dollar", "dollars", "cents", "quarter", "dime", "nickel", "penny",
                   "change", "cost", "buy", "price", "total"]
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


def test_data_graphs_2nd(lnn):
    items = []
    graph_words = ["line", "plot", "bar", "graph", "data", "table", "category",
                   "compare", "most", "least", "pattern", "collect", "organize"]
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


def test_geometry_2nd(lnn):
    items = []
    geo_words = ["triangle", "quadrilateral", "square", "rectangle", "rhombus",
                 "pentagon", "hexagon", "partition", "equal", "shares", "halves",
                 "thirds", "fourths", "whole", "angle", "sides"]
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

def test_ecosystems(lnn):
    items = []
    eco_words = ["plants", "animals", "depend", "sunlight", "food", "chain",
                 "decomposer", "pollinator", "bee", "butterfly", "habitat",
                 "hibernate", "migrate", "camouflage", "hide"]
    for word in eco_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Ecosystems")


def test_earth_systems(lnn):
    items = []
    earth_words = ["land", "water", "mountain", "hill", "valley", "plain",
                   "river", "lake", "ocean", "map", "earthquake", "erosion",
                   "fossil", "wind", "soil", "rock"]
    for word in earth_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Earth Systems")


def test_engineering(lnn):
    items = []
    eng_words = ["engineer", "design", "solution", "problem", "test", "improve",
                 "material", "property", "strong", "flexible", "waterproof", "absorb"]
    for word in eng_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Engineering")


# === SOCIAL STUDIES ===

def test_citizenship(lnn):
    items = []
    cit_words = ["citizen", "rights", "responsibilities", "constitution", "amendment",
                 "freedom", "speech", "religion", "vote", "leader", "community",
                 "volunteer", "follow", "rules", "help"]
    for word in cit_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Studies", "Citizenship")


def test_economics(lnn):
    items = []
    econ_words = ["goods", "services", "producer", "consumer", "supply", "demand",
                  "price", "save", "bank", "lend", "money", "business", "buy", "sell"]
    for word in econ_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Studies", "Economics")


def test_geography_2nd(lnn):
    items = []
    geo_words = ["states", "fifty", "capital", "city", "washington", "river",
                 "lake", "mountain", "natural", "human", "made", "feature",
                 "move", "place", "cold", "warm", "dry", "wet", "adapt"]
    for word in geo_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Studies", "Geography")


def test_history_2nd(lnn):
    items = []
    hist_words = ["historian", "past", "primary", "secondary", "source", "timeline",
                  "event", "important", "people", "change", "country", "mistake",
                  "learn", "tradition", "generation", "holiday", "celebrate"]
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

def test_character_traits_2nd(lnn):
    items = []
    trait_words = ["self-control", "manage", "emotion", "perseverance", "giving",
                   "responsibility", "supposed", "cooperation", "working", "together",
                   "respect", "treating", "honesty", "truth", "kindness", "nice",
                   "empathy", "understanding", "feel", "courage", "right", "thing",
                   "gratitude", "thankful", "fairness", "equally", "patience", "waiting"]
    for word in trait_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social-Emotional", "Character Traits & SEL")
