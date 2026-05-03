"""
11th Grade Curriculum Assessment
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


def test_algebra_II(lnn):
    items = []
    words = ["logarithm", "exponential", "function", "sequence", "series", "matrix", "complex", "imaginary", "polynomial", "rational"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Algebra II")


def test_physics_11th(lnn):
    items = []
    words = ["velocity", "acceleration", "momentum", "gravity", "electromagnetic", "circuit", "optics", "thermodynamics", "quantum", "relativity"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Physics")


def test_us_government(lnn):
    items = []
    words = ["separation", "powers", "checks", "balances", "judicial", "executive", "legislative", "federalism", "sovereignty", "precedent"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Studies", "US Government")


def test_english_11th(lnn):
    items = []
    words = ["american", "literature", "transcendentalism", "modernism", "realism", "narrative", "voice", "context", "criticism", "interpretation"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "English", "English 11")


def test_psychology_11th(lnn):
    items = []
    words = ["cognitive", "behavioral", "developmental", "neuroscience", "consciousness", "memory", "learning", "personality", "disorder", "therapy"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Psychology")


def assess_eleventh_grade(lnn):
    results = {}
    results["algebra_II"] = test_algebra_II(lnn)
    results["physics_11th"] = test_physics_11th(lnn)
    results["us_government"] = test_us_government(lnn)
    results["english_11th"] = test_english_11th(lnn)
    results["psychology_11th"] = test_psychology_11th(lnn)
    return results
