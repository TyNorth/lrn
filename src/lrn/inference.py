"""
LRN Inference System - Attention as Residue

KEY INSIGHT: When you pin a node (query) and propagate, energy flows through
the network and leaves a RESIDUE along the path back to the pinned node.
This residue IS attention - the trace of where activation traveled.

Mechanisms:
1. Pin node → creates query anchor
2. Propagate → energy flows through springs
3. Track path → residue back to pinned node = attention
4. Stronger residue = stronger attention (more frequently traversed)
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate, FLAG_PINNED
from lrn.natural_tokenize import learn_from_text, discover_words


def attention_with_residue(lnn: LatticeNN, query_node: str, propagate_steps: int = 3) -> dict:
    """
    Compute attention as residue path from activated nodes back to query.
    
    Process:
    1. Pin the query node (creates anchor)
    2. Propagate activation through network
    3. Track which springs are traversed
    4. The residue path (springs connecting activated → query) = attention
    
    Returns: {
        "query": pinned node,
        "activated": list of activated nodes,
        "attention_paths": {node: path_back_to_query},
        "attention_strength": {node: residue_score}
    }
    """
    # Reset and pin query
    for node in lnn.nodes.values():
        node.pinned = False
        node.activation = 0
    
    if query_node in lnn.nodes:
        lnn.nodes[query_node].pinned = True
        lnn.nodes[query_node].activation = 100
    
    # Track activation paths: which spring led to each node
    path_tracker = {query_node: [query_node]}  # node -> path from query
    
    # Propagate and track paths
    for step in range(propagate_steps):
        new_activations = {}
        
        for name, node in lnn.nodes.items():
            if node.pinned or node.activation >= 100:
                continue
            
            neighbors = lnn.get_neighbors(name)
            if not neighbors:
                continue
            
            weighted_sum = 0
            stiff_total = 0
            
            for neighbor_name, sp in neighbors:
                neighbor = lnn.nodes.get(neighbor_name)
                if not neighbor or neighbor.activation == 0:
                    continue
                
                eff_k = sp.stiffness * (1.0 / (sp.tau + 1))
                weighted_sum += eff_k * neighbor.activation
                stiff_total += abs(eff_k)
            
            if stiff_total > 0:
                new_act = min(100, (weighted_sum * 6) // stiff_total)
            else:
                new_act = 0
            
            if new_act > 10:  # Threshold for "attended"
                new_activations[name] = new_act
        
        # Apply new activations and track paths
        for name, act in new_activations.items():
            # Find which neighbor contributed most (for path tracking)
            lnn.nodes[name].activation = act
            
            # Track path from query
            neighbors = lnn.get_neighbors(name)
            for neighbor_name, sp in neighbors:
                if neighbor_name in path_tracker:
                    path_tracker[name] = path_tracker[neighbor_name] + [name]
                    break
    
    # Compute attention as residue strength
    # Residue = how many times spring was used in path + activation level
    attention = {}
    for node, path in path_tracker.items():
        if node != query_node:
            # Residue strength = activation * path length factor
            if node in lnn.nodes:
                attention[node] = {
                    "activation": lnn.nodes[node].activation,
                    "path": " → ".join(path),
                    "residue": lnn.nodes[node].activation * (1.0 / len(path))
                }
    
    # Sort by residue strength
    sorted_attention = sorted(attention.items(), 
                             key=lambda x: -x[1]["residue"])
    
    return {
        "query": query_node,
        "activated_count": len(path_tracker) - 1,
        "attention": sorted_attention[:10]
    }


def add_word_nodes(lnn, samples: list, prefix: str = "word"):
    """Add word-level nodes for inference."""
    for text in samples:
        words = text.lower().split()
        for word in words:
            word_node = f"{prefix}:{word}"
            if word_node not in lnn.nodes:
                lnn.add_node(word_node)
        
        for i in range(len(words) - 1):
            a = f"{prefix}:{words[i]}"
            b = f"{prefix}:{words[i+1]}"
            if a in lnn.nodes and b in lnn.nodes:
                lnn.add_or_update_spring(a, b, stiffness=10, tau=2, mode="add")


def test_attention_residue():
    print("=" * 60)
    print("ATTENTION AS RESIDUE TEST")
    print("=" * 60)
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    # Train on sensory + causation
    SENSORY = [
        "fire is hot", "fire burns wood", "water is wet", "water cools fire",
        "ice is cold", "sun heats ground", "rock is hard", "soft pillow",
    ]
    
    CAUSATION = [
        "fire burns wood and wood becomes black",
        "water cools fire and fire becomes small",
        "sun heats ground and ground becomes warm",
    ]
    
    print("\n--- Training ---")
    for text in SENSORY:
        learn_from_text(lnn, text, repetitions=10, learn_type="sensory")
    for text in CAUSATION:
        learn_from_text(lnn, text, repetitions=10, learn_type="causation")
    
    # Add word nodes
    all_texts = SENSORY + CAUSATION
    add_word_nodes(lnn, all_texts)
    
    print(f"Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
    
    # Test attention on different query nodes
    print("\n--- Attention Query: 'word:fire' ---")
    result = attention_with_residue(lnn, "word:fire", propagate_steps=3)
    print(f"Query: {result['query']}")
    print(f"Activated: {result['activated_count']} nodes")
    print("Top attention (residue paths):")
    for node, info in result['attention']:
        print(f"  {node}: act={info['activation']}, path={info['path']}")
    
    print("\n--- Attention Query: 'word:water' ---")
    result = attention_with_residue(lnn, "word:water", propagate_steps=3)
    print(f"Activated: {result['activated_count']} nodes")
    for node, info in result['attention'][:5]:
        print(f"  {node}: {info['path']}")
    
    print("\n--- Attention Query: 'word:burns' ---")
    result = attention_with_residue(lnn, "word:burns", propagate_steps=3)
    print(f"Activated: {result['activated_count']} nodes")
    for node, info in result['attention'][:5]:
        print(f"  {node}: {info['path']}")
    
    return result


if __name__ == "__main__":
    test_attention_residue()