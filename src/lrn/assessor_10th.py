"""
10th Grade Curriculum Assessment
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


def test_geometry_10th(lnn):
    items = []
    words = ["proof", "theorem", "congruence", "similarity", "trigonometry", "sine", "cosine", "tangent", "circle", "arc"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Geometry")


def test_chemistry_10th(lnn):
    items = []
    words = ["periodic", "bond", "ionic", "covalent", "mole", "stoichiometry", "solution", "acid", "base", "reaction"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Chemistry")


def test_us_history_10th(lnn):
    items = []
    words = ["colonial", "declaration", "constitution", "civil", "war", "amendment", "reconstruction", "industrial", "progressive", "century"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Studies", "US History")


def test_english_10th(lnn):
    items = []
    words = ["argument", "rhetoric", "analysis", "synthesis", "citation", "thesis", "structure", "tone", "audience", "purpose"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "English", "English 10")


def test_economics_10th(lnn):
    items = []
    words = ["supply", "demand", "market", "inflation", "gdp", "trade", "capital", "labor", "government", "policy"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Studies", "Economics")


def assess_tenth_grade(lnn):
    results = {}
    results["geometry_10th"] = test_geometry_10th(lnn)
    results["chemistry_10th"] = test_chemistry_10th(lnn)
    results["us_history_10th"] = test_us_history_10th(lnn)
    results["english_10th"] = test_english_10th(lnn)
    results["economics_10th"] = test_economics_10th(lnn)
    return results
