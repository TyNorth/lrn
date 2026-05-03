"""
7th Grade Curriculum Assessment
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')


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


def test_proportional_relationships(lnn):
    items = []
    words = ["proportional", "constant", "unit", "rate", "ratio", "slope", "relationship", "equivalent", "scale", "factor"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Proportional Relationships")


def test_rational_numbers(lnn):
    items = []
    words = ["rational", "integer", "fraction", "decimal", "negative", "positive", "absolute", "number", "line", "opposite"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Rational Numbers")


def test_expressions_7th(lnn):
    items = []
    words = ["expression", "variable", "coefficient", "term", "equation", "inequality", "solution", "simplify", "expand", "factor"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Expressions & Equations")


def test_geometry_7th(lnn):
    items = []
    words = ["circle", "radius", "diameter", "area", "perimeter", "angle", "triangle", "cross", "section", "scale"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Geometry")


def test_probability_7th(lnn):
    items = []
    words = ["probability", "chance", "outcome", "sample", "event", "random", "experimental", "theoretical", "compound", "likelihood"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Probability")


def test_life_science_7th(lnn):
    items = []
    words = ["cell", "tissue", "organ", "system", "organism", "homeostasis", "reproduction", "heredity", "gene", "trait"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Life Science")


def test_physical_science_7th(lnn):
    items = []
    words = ["atom", "molecule", "element", "compound", "reaction", "energy", "thermal", "kinetic", "potential", "conservation"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Physical Science")


def test_world_history_7th(lnn):
    items = []
    words = ["medieval", "renaissance", "reformation", "exploration", "empire", "trade", "culture", "dynasty", "revolution", "feudal"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Studies", "World History")


def assess_seventh_grade(lnn):
    results = {}
    results["proportional_relationships"] = test_proportional_relationships(lnn)
    results["rational_numbers"] = test_rational_numbers(lnn)
    results["expressions_7th"] = test_expressions_7th(lnn)
    results["geometry_7th"] = test_geometry_7th(lnn)
    results["probability_7th"] = test_probability_7th(lnn)
    results["life_science_7th"] = test_life_science_7th(lnn)
    results["physical_science_7th"] = test_physical_science_7th(lnn)
    results["world_history_7th"] = test_world_history_7th(lnn)
    return results
