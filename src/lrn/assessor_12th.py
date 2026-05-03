"""
12th Grade Curriculum Assessment
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


def test_calculus_12th(lnn):
    items = []
    words = ["limit", "derivative", "integral", "continuity", "differentiation", "antiderivative", "optimization", "related", "chain", "rule"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Calculus")


def test_statistics_12th(lnn):
    items = []
    words = ["probability", "distribution", "normal", "hypothesis", "confidence", "regression", "variance", "standard", "deviation", "significance"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Statistics")


def test_environmental_science(lnn):
    items = []
    words = ["ecosystem", "biodiversity", "sustainability", "pollution", "climate", "renewable", "conservation", "habitat", "carbon", "cycle"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Environmental Science")


def test_philosophy_12th(lnn):
    items = []
    words = ["ethics", "logic", "metaphysics", "epistemology", "existentialism", "utilitarianism", "virtue", "reason", "morality", "argument"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Humanities", "Philosophy")


def test_comparative_government(lnn):
    items = []
    words = ["authoritarian", "democratic", "parliamentary", "presidential", "ideology", "sovereignty", "international", "diplomacy", "treaty", "globalization"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Studies", "Comparative Government")


def assess_twelfth_grade(lnn):
    results = {}
    results["calculus_12th"] = test_calculus_12th(lnn)
    results["statistics_12th"] = test_statistics_12th(lnn)
    results["environmental_science"] = test_environmental_science(lnn)
    results["philosophy_12th"] = test_philosophy_12th(lnn)
    results["comparative_government"] = test_comparative_government(lnn)
    return results
