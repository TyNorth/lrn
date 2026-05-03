"""
4th Grade Curriculum Assessment
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


def test_multi_digit_arithmetic(lnn):
    items = []
    words = ["multiply", "divide", "product", "quotient", "remainder", "algorithm", "digit", "place", "value", "round"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Multi-Digit Arithmetic")


def test_fractions_4th(lnn):
    items = []
    words = ["fraction", "equivalent", "numerator", "denominator", "decimal", "hundredth", "tenth", "compare", "visual", "model"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Fractions & Decimals")


def test_measurement_data(lnn):
    items = []
    words = ["area", "perimeter", "rectangle", "formula", "measurement", "protractor", "angle", "degree", "line", "plot"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Measurement & Data")


def test_geometry_4th(lnn):
    items = []
    words = ["parallel", "perpendicular", "segment", "ray", "symmetry", "classify", "dimensional", "figure", "point", "line"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Geometry")


def test_reading_literature_4th(lnn):
    items = []
    words = ["theme", "summarize", "character", "setting", "event", "compare", "contrast", "point", "view", "chapter"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Literacy", "Reading Literature")


def test_reading_informational_4th(lnn):
    items = []
    words = ["main", "idea", "evidence", "explain", "describe", "structure", "chronology", "integrate", "information", "topic"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Literacy", "Reading Informational")


def test_writing_4th(lnn):
    items = []
    words = ["opinion", "informative", "narrative", "research", "evidence", "analysis", "convention", "capitalization", "punctuation", "coherent"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Literacy", "Writing")


def test_energy(lnn):
    items = []
    words = ["energy", "transfer", "collision", "motion", "speed", "sound", "light", "circuit", "thermal", "electrical"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Energy")


def test_waves(lnn):
    items = []
    words = ["wave", "amplitude", "wavelength", "reflect", "absorb", "digital", "encode", "pattern", "technology", "information"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Waves & Technology")


def test_earth_systems(lnn):
    items = []
    words = ["rainfall", "erosion", "canyon", "valley", "delta", "map", "landform", "weathering", "flow", "ocean"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Earth Systems")


def test_colonial_america(lnn):
    items = []
    words = ["colony", "colonist", "freedom", "mayflower", "democracy", "tax", "resist", "britain", "compact", "region"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Studies", "Colonial America")


def test_civics_4th(lnn):
    items = []
    words = ["constitution", "amendment", "right", "responsibility", "citizen", "branch", "legislative", "judicial", "voting", "democracy"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Studies", "Civics")


def test_programming_logic_4th(lnn):
    items = []
    words = ["algorithm", "step", "sequence", "pattern", "error", "debug", "function", "loop", "variable", "array"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Computer Science", "Programming Logic")


def assess_fourth_grade(lnn):
    results = {}
    results["multi_digit_arithmetic"] = test_multi_digit_arithmetic(lnn)
    results["fractions_4th"] = test_fractions_4th(lnn)
    results["measurement_data"] = test_measurement_data(lnn)
    results["geometry_4th"] = test_geometry_4th(lnn)
    results["reading_literature_4th"] = test_reading_literature_4th(lnn)
    results["reading_informational_4th"] = test_reading_informational_4th(lnn)
    results["writing_4th"] = test_writing_4th(lnn)
    results["programming_logic"] = test_programming_logic_4th(lnn)
    results["energy"] = test_energy(lnn)
    results["waves"] = test_waves(lnn)
    results["earth_systems"] = test_earth_systems(lnn)
    results["colonial_america"] = test_colonial_america(lnn)
    results["civics_4th"] = test_civics_4th(lnn)
    return results
