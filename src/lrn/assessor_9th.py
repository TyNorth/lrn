"""
9th Grade Curriculum Assessment
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


def test_algebra_I(lnn):
    items = []
    words = ["polynomial", "quadratic", "factor", "vertex", "parabola", "exponent", "radical", "expression", "coefficient", "discriminant"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Algebra I")


def test_biology_9th(lnn):
    items = []
    words = ["dna", "rna", "protein", "enzyme", "mitosis", "meiosis", "evolution", "natural", "selection", "mutation"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Biology")


def test_world_history_9th(lnn):
    items = []
    words = ["ancient", "classical", "empire", "philosophy", "religion", "trade", "conquest", "dynasty", "culture", "legacy"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Studies", "World History")


def test_english_9th(lnn):
    items = []
    words = ["narrative", "theme", "character", "plot", "conflict", "symbol", "metaphor", "irony", "perspective", "evidence"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "English", "English 9")


def test_geography_9th(lnn):
    items = []
    words = ["latitude", "longitude", "climate", "region", "population", "migration", "urban", "rural", "resource", "environment"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Studies", "Geography")


def assess_ninth_grade(lnn):
    results = {}
    results["algebra_I"] = test_algebra_I(lnn)
    results["biology_9th"] = test_biology_9th(lnn)
    results["world_history_9th"] = test_world_history_9th(lnn)
    results["english_9th"] = test_english_9th(lnn)
    results["geography_9th"] = test_geography_9th(lnn)
    return results
