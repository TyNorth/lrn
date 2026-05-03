"""
Lattice Probing - Query what the lattice knows
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import propagate
from lrn.charts import probe_result, compare_result


def probe(lnn, query, depth=1):
    """
    Probe a single node - show activation paths.
    
    Returns list of connection dicts with name, stiffness, tau.
    """
    query_node = f"word:{query}"
    
    if query_node not in lnn.nodes:
        return []
    
    for n in lnn.nodes.values():
        n.activation = 0
    
    lnn.nodes[query_node].activation = 100
    propagate(lnn, n_steps=depth)
    
    connections = []
    neighbors = lnn.get_neighbors(query_node)
    for neighbor_name, sp in neighbors:
        if neighbor_name.startswith("word:"):
            name = neighbor_name.replace("word:", "")
            connections.append({
                "name": name,
                "stiffness": sp.stiffness,
                "tau": sp.tau,
                "activation": lnn.nodes.get(neighbor_name, None).activation if neighbor_name in lnn.nodes else 0,
            })
    
    connections.sort(key=lambda c: c["stiffness"], reverse=True)
    return connections


def probe_category(lnn, category_name):
    """Test category clustering density."""
    categories = {
        "animals": ["cat", "dog", "bird", "fish", "horse", "cow", "sheep", "pig", "chicken", "duck"],
        "colors": ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "brown", "pink"],
        "emotions": ["happy", "sad", "angry", "scared", "surprised", "tired"],
        "numbers": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
    }
    
    members = categories.get(category_name, [])
    if not members:
        return None
    
    tau3 = 0
    total = 0
    for i in range(len(members)):
        for j in range(i+1, len(members)):
            key = lnn._key(f"word:{members[i]}", f"word:{members[j]}")
            total += 1
            if key in lnn.springs and lnn.springs[key].tau == 3:
                tau3 += 1
    
    return {
        "category": category_name,
        "members": members,
        "tau3": tau3,
        "total": total,
        "density": tau3 / max(1, total),
    }


def compare(lnn, a, b):
    """Compare two nodes - show shared vs unique associations."""
    a_node = f"word:{a}"
    b_node = f"word:{b}"
    
    if a_node not in lnn.nodes or b_node not in lnn.nodes:
        return None
    
    a_neighbors = set()
    for neighbor_name, sp in lnn.get_neighbors(a_node):
        if neighbor_name.startswith("word:"):
            a_neighbors.add(neighbor_name.replace("word:", ""))
    
    b_neighbors = set()
    for neighbor_name, sp in lnn.get_neighbors(b_node):
        if neighbor_name.startswith("word:"):
            b_neighbors.add(neighbor_name.replace("word:", ""))
    
    shared_names = a_neighbors & b_neighbors
    only_a_names = a_neighbors - b_neighbors
    only_b_names = b_neighbors - a_neighbors
    
    def get_connections(names, target_node):
        connections = []
        for name in names:
            key = lnn._key(target_node, f"word:{name}")
            if key in lnn.springs:
                sp = lnn.springs[key]
                connections.append({
                    "name": name,
                    "stiffness": sp.stiffness,
                    "tau": sp.tau,
                })
        connections.sort(key=lambda c: c["stiffness"], reverse=True)
        return connections
    
    return {
        "a": a,
        "b": b,
        "shared": get_connections(shared_names, a_node),
        "only_a": get_connections(only_a_names, a_node),
        "only_b": get_connections(only_b_names, b_node),
    }
