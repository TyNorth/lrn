"""
College Level Curriculum Assessment
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


def test_linear_algebra(lnn):
    items = []
    words = ["vector", "matrix", "eigenvalue", "eigenvector", "determinant", "subspace", "basis", "dimension", "linear", "transformation"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Math", "Linear Algebra")


def test_organic_chemistry(lnn):
    items = []
    words = ["hydrocarbon", "functional", "stereochemistry", "synthesis", "mechanism", "nucleophile", "electrophile", "aromatic", "carbonyl", "isomer"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Organic Chemistry")


def test_quantum_physics(lnn):
    items = []
    words = ["wavefunction", "schrodinger", "uncertainty", "superposition", "entanglement", "photon", "electron", "orbital", "tunneling", "hamiltonian"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Science", "Quantum Physics")


def test_computer_science(lnn):
    items = []
    words = ["algorithm", "complexity", "recursion", "data", "structure", "sorting", "graph", "tree", "dynamic", "programming"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Computer Science", "Computer Science")


def test_economics_college(lnn):
    items = []
    words = ["microeconomics", "macroeconomics", "equilibrium", "elasticity", "monetary", "fiscal", "opportunity", "marginal", "utility", "game"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Social Science", "Economics")


def test_research_methods(lnn):
    items = []
    words = ["methodology", "quantitative", "qualitative", "variable", "control", "sample", "validity", "reliability", "correlation", "causation"]
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Research", "Research Methods")


def assess_college(lnn):
    results = {}
    results["linear_algebra"] = test_linear_algebra(lnn)
    results["organic_chemistry"] = test_organic_chemistry(lnn)
    results["quantum_physics"] = test_quantum_physics(lnn)
    results["computer_science"] = test_computer_science(lnn)
    results["economics_college"] = test_economics_college(lnn)
    results["research_methods"] = test_research_methods(lnn)
    return results
