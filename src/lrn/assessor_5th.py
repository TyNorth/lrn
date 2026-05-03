"""
5th Grade Curriculum Assessment
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


def test_algebraic_thinking(lnn):
    items = []
    words = ["parentheses", "brackets", "braces", "expression", "evaluate", "pattern", "sequence", "rule", "ordered", "pair"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Algebraic Thinking")


def test_place_value_decimals(lnn):
    items = []
    words = ["place", "value", "decimal", "thousandth", "hundredth", "tenth", "round", "multiply", "divide", "power"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Place Value & Decimals")


def test_fraction_operations(lnn):
    items = []
    words = ["fraction", "numerator", "denominator", "unlike", "mixed", "number", "division", "scaling", "resizing", "reciprocal"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Fraction Operations")


def test_volume(lnn):
    items = []
    words = ["volume", "cube", "unit", "cubic", "prism", "rectangular", "length", "width", "height", "additive"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Volume")


def test_coordinate_geometry(lnn):
    items = []
    words = ["coordinate", "plane", "axis", "origin", "ordered", "pair", "quadrant", "graph", "plot", "perpendicular"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Coordinate Geometry")


def test_matter(lnn):
    items = []
    words = ["matter", "particle", "atom", "conservation", "mixture", "solution", "dissolve", "property", "conductivity", "substance"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Matter & Interactions")


def test_ecosystems_5th(lnn):
    items = []
    words = ["organism", "food web", "producer", "consumer", "decomposer", "energy", "transfer", "balance", "population", "environment"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Ecosystems")


def test_american_revolution(lnn):
    items = []
    words = ["revolution", "independence", "declaration", "patriot", "loyalist", "boston", "tea", "party", "washington", "treaty"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Studies", "American Revolution")


def test_critical_thinking(lnn):
    items = []
    words = ["analyze", "evaluate", "synthesize", "cite", "evidence", "perspective", "bias", "conclusion", "reasoning", "compare"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Literacy", "Critical Thinking")


def test_programming_logic_5th(lnn):
    items = []
    words = ["boolean", "and", "or", "not", "true", "false", "condition", "loop", "iterate", "variable"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Computer Science", "Programming Logic")


def assess_fifth_grade(lnn):
    results = {}
    results["algebraic_thinking"] = test_algebraic_thinking(lnn)
    results["place_value_decimals"] = test_place_value_decimals(lnn)
    results["fraction_operations"] = test_fraction_operations(lnn)
    results["volume"] = test_volume(lnn)
    results["coordinate_geometry"] = test_coordinate_geometry(lnn)
    results["programming_logic"] = test_programming_logic_5th(lnn)
    results["matter"] = test_matter(lnn)
    results["ecosystems_5th"] = test_ecosystems_5th(lnn)
    results["american_revolution"] = test_american_revolution(lnn)
    results["critical_thinking"] = test_critical_thinking(lnn)
    return results
