"""
6th Grade Curriculum Assessment
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


def test_ratios_proportions(lnn):
    items = []
    words = ["ratio", "rate", "unit", "proportional", "percent", "quantity", "pricing", "speed"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Ratios & Proportional Relationships")


def test_number_system_6th(lnn):
    items = []
    words = ["quotient", "dividend", "decimal", "factor", "multiple", "absolute", "integer", "rational", "positive", "negative"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Number System")


def test_expressions_equations(lnn):
    items = []
    words = ["exponent", "expression", "variable", "equation", "inequality", "equivalent", "coefficient", "term", "sum", "product"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Expressions & Equations")


def test_geometry_6th(lnn):
    items = []
    words = ["polygon", "surface", "area", "net", "dimensional", "vertex", "coordinate", "triangle", "prism", "volume"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Geometry")


def test_statistics_6th(lnn):
    items = []
    words = ["statistical", "distribution", "median", "mean", "deviation", "histogram", "variability", "center", "spread", "shape"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Statistics")


def test_earth_science_6th(lnn):
    items = []
    words = ["plate", "tectonics", "fossil", "earthquake", "hazard", "water", "cycle", "geoscience", "erosion", "resource"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Earth Science")


def test_life_science_6th(lnn):
    items = []
    words = ["cell", "organism", "cellular", "respiration", "ecosystem", "biodiversity", "population", "food", "energy", "growth"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Life Science")


def test_ancient_civilizations(lnn):
    items = []
    words = ["mesopotamia", "egypt", "pharaoh", "democracy", "athens", "roman", "republic", "silk", "road", "civilization"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Studies", "Ancient Civilizations")


def test_programming_logic_6th(lnn):
    items = []
    words = ["object", "class", "instance", "method", "property", "event", "handler", "stack", "queue", "data"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Computer Science", "Programming Logic")


def assess_sixth_grade(lnn):
    results = {}
    results["ratios_proportions"] = test_ratios_proportions(lnn)
    results["number_system_6th"] = test_number_system_6th(lnn)
    results["expressions_equations"] = test_expressions_equations(lnn)
    results["geometry_6th"] = test_geometry_6th(lnn)
    results["statistics_6th"] = test_statistics_6th(lnn)
    results["programming_logic"] = test_programming_logic_6th(lnn)
    results["earth_science_6th"] = test_earth_science_6th(lnn)
    results["life_science_6th"] = test_life_science_6th(lnn)
    results["ancient_civilizations"] = test_ancient_civilizations(lnn)
    return results
