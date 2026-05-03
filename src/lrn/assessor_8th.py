"""
8th Grade Curriculum Assessment
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


def test_irrational_numbers(lnn):
    items = []
    words = ["irrational", "rational", "square", "root", "decimal", "approximate", "real", "number", "pi", "radical"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Irrational Numbers")


def test_linear_equations(lnn):
    items = []
    words = ["linear", "equation", "slope", "intercept", "graph", "function", "input", "output", "rate", "change"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Linear Equations")


def test_systems_equations(lnn):
    items = []
    words = ["system", "simultaneous", "substitution", "elimination", "solution", "intersection", "parallel", "consistent", "inconsistent", "dependent"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Systems of Equations")


def test_geometry_8th(lnn):
    items = []
    words = ["transformation", "rotation", "reflection", "translation", "congruent", "similar", "pythagorean", "theorem", "volume", "cylinder"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Geometry")


def test_statistics_8th(lnn):
    items = []
    words = ["scatter", "plot", "correlation", "trend", "association", "outlier", "linear", "model", "data", "association"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Statistics")


def test_physical_science_8th(lnn):
    items = []
    words = ["force", "motion", "newton", "momentum", "energy", "wave", "frequency", "amplitude", "electric", "magnetic"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Physical Science")


def test_civics_8th(lnn):
    items = []
    words = ["constitution", "amendment", "rights", "freedom", "government", "democracy", "citizen", "responsibility", "branch", "federal"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Studies", "Civics")


def test_modern_history(lnn):
    items = []
    words = ["industrial", "revolution", "imperialism", "nationalism", "war", "independence", "reform", "movement", "colonial", "modern"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Studies", "Modern History")


def assess_eighth_grade(lnn):
    results = {}
    results["irrational_numbers"] = test_irrational_numbers(lnn)
    results["linear_equations"] = test_linear_equations(lnn)
    results["systems_equations"] = test_systems_equations(lnn)
    results["geometry_8th"] = test_geometry_8th(lnn)
    results["statistics_8th"] = test_statistics_8th(lnn)
    results["physical_science_8th"] = test_physical_science_8th(lnn)
    results["civics_8th"] = test_civics_8th(lnn)
    results["modern_history"] = test_modern_history(lnn)
    return results
