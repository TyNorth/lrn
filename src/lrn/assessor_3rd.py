"""
3rd Grade Curriculum Assessment - Detailed sub-skill breakdown

Based on Common Core 3rd Grade standards.
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


def assess_third_grade(lnn):
    """Full 3rd Grade curriculum assessment."""
    results = {}
    
    # === LITERACY DOMAIN ===
    results["phonics_multisyllable"] = test_phonics_multisyllable(lnn)
    results["sight_words_3rd"] = test_sight_words_3rd(lnn)
    results["reading_fluency"] = test_reading_fluency_3rd(lnn)
    results["reading_comprehension"] = test_reading_comprehension_3rd(lnn)
    results["writing_skills"] = test_writing_skills_3rd(lnn)
    results["grammar"] = test_grammar_3rd(lnn)
    results["vocabulary"] = test_vocabulary_3rd(lnn)
    
    # === MATHEMATICS DOMAIN ===
    results["multiplication"] = test_multiplication(lnn)
    results["division"] = test_division(lnn)
    results["fractions"] = test_fractions(lnn)
    results["measurement"] = test_measurement_3rd(lnn)
    results["data_graphs"] = test_data_graphs_3rd(lnn)
    results["geometry"] = test_geometry_3rd(lnn)
    
    # === SCIENCE DOMAIN ===
    results["forces_motion"] = test_forces_motion(lnn)
    results["weather_climate"] = test_weather_climate(lnn)
    results["ecosystems_heredity"] = test_ecosystems_heredity(lnn)
    
    # === SOCIAL STUDIES ===
    results["government"] = test_government(lnn)
    results["economics"] = test_economics_3rd(lnn)
    results["geography"] = test_geography_3rd(lnn)
    results["history"] = test_history_3rd(lnn)
    
    # === SOCIAL-EMOTIONAL ===
    results["sel"] = test_sel_3rd(lnn)
    
    return results


# === LITERACY ===

def test_phonics_multisyllable(lnn):
    items = []
    patterns = [
        ("tion", ["action", "nation", "station", "celebration"]),
        ("ture", ["picture", "nature", "future", "temperature"]),
        ("able", ["comfortable", "remarkable", "valuable", "reliable"]),
        ("ful", ["beautiful", "wonderful", "careful", "grateful"]),
        ("ness", ["kindness", "happiness", "sadness", "darkness"]),
        ("ment", ["movement", "excitement", "improvement"]),
        ("er", ["teacher", "farmer", "baker", "faster"]),
        ("est", ["fastest", "tallest", "strongest", "biggest"]),
        ("un", ["unhappy", "unfair", "untie", "undo"]),
        ("re", ["reread", "rewrite", "rebuild", "return"]),
    ]
    for pattern, words in patterns:
        present = [w for w in words if _node_exists(lnn, w)]
        if len(present) >= 3:
            items.append((pattern, "PASS", f"{len(present)} words present"))
        else:
            items.append((pattern, "PARTIAL", f"{len(present)} words present"))
    return _make_result(items, "Literacy", "Multi-Syllable Phonics")


def test_sight_words_3rd(lnn):
    items = []
    sight_words = ["about", "better", "bring", "carry", "clean", "cut", "done", "draw",
                   "eight", "fall", "far", "five", "give", "good", "grow", "hold", "if",
                   "keep", "kind", "laugh", "light", "long", "much", "myself", "never",
                   "only", "own", "pick", "seven", "shall", "show", "six", "small",
                   "start", "ten", "today", "together", "try", "upon", "warm", "why",
                   "wish", "would"]
    for word in sight_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 2:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Literacy", "Sight Words (3rd Grade)")


def test_reading_fluency_3rd(lnn):
    items = []
    fluency_words = ["chapter", "expression", "pacing", "accuracy", "rate", "silently",
                     "extended", "periods", "context", "meaning"]
    for word in fluency_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Literacy", "Reading Fluency")


def test_reading_comprehension_3rd(lnn):
    items = []
    comprehension_tests = [
        ("theme", ["lesson", "story", "message"]),
        ("compare", ["contrast", "similar", "different"]),
        ("contrast", ["compare", "alike", "different"]),
        ("challenge", ["character", "overcome", "problem"]),
        ("main", ["idea", "detail", "support"]),
        ("heading", ["section", "title", "text"]),
        ("caption", ["photo", "picture", "explain"]),
        ("glossary", ["word", "meaning", "dictionary"]),
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


def test_writing_skills_3rd(lnn):
    items = []
    writing_words = ["opinion", "essay", "introduction", "body", "conclusion", "reason",
                     "informative", "report", "narrative", "dialogue", "descriptive",
                     "transition", "synonym", "thesaurus", "comma", "quotation",
                     "topic", "sentence", "paragraph", "revise"]
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


def test_grammar_3rd(lnn):
    items = []
    grammar_tests = [
        ("plural", ["regular", "irregular", "children", "adds"]),
        ("conjunction", ["and", "but", "or", "connect"]),
        ("adjective", ["modifies", "noun", "red", "big"]),
        ("adverb", ["modifies", "verb", "quickly", "slowly"]),
        ("preposition", ["in", "on", "at", "under", "over"]),
        ("tense", ["past", "present", "future", "will"]),
        ("clause", ["independent", "simple", "compound"]),
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


def test_vocabulary_3rd(lnn):
    items = []
    vocab_words = ["prefix", "suffix", "context", "clues", "un", "re", "ful", "less",
                   "tion", "ment", "synonym", "antonym", "homophone", "meaning"]
    for word in vocab_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Literacy", "Vocabulary Strategies")


# === MATHEMATICS ===

def test_multiplication(lnn):
    items = []
    mult_tests = [
        ("multiply", ["times", "product", "factor", "groups"]),
        ("times", ["multiply", "equals", "product"]),
        ("product", ["multiply", "times", "answer"]),
        ("factor", ["multiply", "times", "number"]),
        ("six", ["times", "seven", "forty two"]),
        ("eight", ["times", "nine", "seventy two"]),
        ("nine", ["times", "ten", "ninety"]),
    ]
    for word, expected in mult_tests:
        if _node_exists(lnn, word):
            found_any, found = _has_connections_to(lnn, word, expected)
            if found_any:
                items.append((word, "PASS", f"connects to {', '.join(found)}"))
            else:
                items.append((word, "PARTIAL", "exists but no math connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Mathematics", "Multiplication Within 100")


def test_division(lnn):
    items = []
    div_tests = [
        ("divide", ["divided", "quotient", "groups", "equal"]),
        ("divided", ["divide", "equals", "groups"]),
        ("quotient", ["divide", "answer", "result"]),
        ("share", ["divide", "equal", "groups"]),
        ("twenty", ["divided", "four", "five"]),
        ("thirty", ["divided", "five", "six"]),
    ]
    for word, expected in div_tests:
        if _node_exists(lnn, word):
            found_any, found = _has_connections_to(lnn, word, expected)
            if found_any:
                items.append((word, "PASS", f"connects to {', '.join(found)}"))
            else:
                items.append((word, "PARTIAL", "exists but no math connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Mathematics", "Division Within 100")


def test_fractions(lnn):
    items = []
    frac_words = ["fraction", "numerator", "denominator", "half", "third", "fourth",
                  "equal", "parts", "whole", "compare", "greater", "less"]
    for word in frac_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Mathematics", "Fractions")


def test_measurement_3rd(lnn):
    items = []
    measure_words = ["liter", "gram", "kilogram", "minute", "interval", "quarter",
                     "inch", "mass", "volume", "liquid", "time", "nearest"]
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


def test_data_graphs_3rd(lnn):
    items = []
    graph_words = ["line", "plot", "picture", "graph", "bar", "scaled", "data",
                   "measurement", "solve", "problem"]
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


def test_geometry_3rd(lnn):
    items = []
    geo_words = ["rectangle", "rhombus", "square", "parallelogram", "trapezoid",
                 "area", "perimeter", "length", "width", "side", "angle",
                 "partition", "unit", "multiply", "add"]
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

def test_forces_motion(lnn):
    items = []
    force_words = ["force", "motion", "push", "pull", "friction", "magnet", "gravity",
                   "balanced", "unbalanced", "attract", "repel", "pole", "slow"]
    for word in force_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Forces & Motion")


def test_weather_climate(lnn):
    items = []
    weather_words = ["weather", "climate", "pattern", "temperature", "rain", "snow",
                     "wind", "cloud", "thermometer", "gauge", "region", "season",
                     "vapor", "freezing", "record"]
    for word in weather_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Weather & Climate")


def test_ecosystems_heredity(lnn):
    items = []
    eco_words = ["organism", "environment", "survive", "inherit", "trait", "parent",
                 "learned", "variation", "fossil", "ancient", "evidence", "caterpillar",
                 "butterfly", "puppy", "flower"]
    for word in eco_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Ecosystems & Heredity")


# === SOCIAL STUDIES ===

def test_government(lnn):
    items = []
    gov_words = ["government", "legislative", "executive", "judicial", "constitution",
                 "congress", "president", "supreme", "court", "law", "branch",
                 "citizen", "right", "responsibility", "vote", "local", "state", "federal"]
    for word in gov_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Studies", "Government")


def test_economics_3rd(lnn):
    items = []
    econ_words = ["resource", "natural", "human", "capital", "specialization", "trade",
                  "import", "export", "entrepreneur", "profit", "goods", "service",
                  "business", "money", "earn"]
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


def test_geography_3rd(lnn):
    items = []
    geo_words = ["continent", "hemisphere", "equator", "latitude", "longitude", "globe",
                 "scale", "physical", "political", "border", "city", "pacific",
                 "atlantic", "indian", "arctic", "antarctica"]
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


def test_history_3rd(lnn):
    items = []
    hist_words = ["ancient", "civilization", "river", "egyptian", "pyramid", "nile",
                  "greek", "democracy", "roman", "road", "explorer", "discover",
                  "industrial", "revolution", "telegraph", "communication", "civil",
                  "rights", "equality", "leader"]
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

def test_sel_3rd(lnn):
    items = []
    sel_words = ["awareness", "management", "resilience", "integrity", "tolerance",
                 "leadership", "teamwork", "emotion", "strength", "impulse", "perspective",
                 "communicate", "cooperate", "decision", "ethical", "bounce", "challenge",
                 "accept", "difference", "guide", "positive", "goal"]
    for word in sel_words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            if count >= 1:
                items.append((word, "PASS", f"{count} connections"))
            else:
                items.append((word, "PARTIAL", "exists but isolated"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social-Emotional", "SEL Competencies")
