"""
Letter Sensory Grounding - Letters as visual/electromagnetic events

Letters aren't abstract symbols — they're visual patterns on the retina.
Each letter has geometric properties:
- Strokes: vertical, horizontal, diagonal lines
- Curves: open, closed, partial arcs
- Intersections: T-junctions, X-crossings, Y-junctions
- Enclosed spaces: holes (a, b, d, e, g, o, p, q)
- Spatial features: ascenders (b, d, f, h, k, l, t), descenders (g, j, p, q, y)

Letters that share visual features should have constructive interference.
Letters that look different should have destructive interference.
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn.sensory_wave import sensory_environment, create_lnn
from lrn.charts import bar_chart, tau_distribution


# Letters grounded in visual geometry
# Phase alignment = shared visual features
# Each letter has multiple feature-waves that fire together

def make_letter_concepts():
    """Create sensory wave concepts for all 26 letters, grounded in visual features."""
    
    # Visual feature groups — letters sharing features fire in phase
    concepts = []
    
    # === VERTICAL LINE letters (ascenders) ===
    # b, d, f, h, k, l, t — tall, vertical stroke dominant
    vertical_ascenders = [
        {"name": "b", "amplitude": 0.8, "phase": 0.0, "rise_time": 2, "peak_time": 4, "fall_time": 3, "cooldown": 10},
        {"name": "d", "amplitude": 0.8, "phase": 0.02, "rise_time": 2, "peak_time": 4, "fall_time": 3, "cooldown": 10},
        {"name": "f", "amplitude": 0.7, "phase": 0.04, "rise_time": 2, "peak_time": 4, "fall_time": 3, "cooldown": 10},
        {"name": "h", "amplitude": 0.8, "phase": 0.01, "rise_time": 2, "peak_time": 4, "fall_time": 3, "cooldown": 10},
        {"name": "k", "amplitude": 0.7, "phase": 0.03, "rise_time": 2, "peak_time": 4, "fall_time": 3, "cooldown": 10},
        {"name": "l", "amplitude": 0.9, "phase": 0.0, "rise_time": 2, "peak_time": 4, "fall_time": 3, "cooldown": 10},
        {"name": "t", "amplitude": 0.7, "phase": 0.05, "rise_time": 2, "peak_time": 4, "fall_time": 3, "cooldown": 10},
    ]
    
    # === CLOSED LOOP letters ===
    # a, b, d, e, g, o, p, q — have enclosed spaces
    closed_loop = [
        {"name": "a", "amplitude": 0.8, "phase": 0.3, "rise_time": 3, "peak_time": 4, "fall_time": 3, "cooldown": 12},
        {"name": "b", "amplitude": 0.8, "phase": 0.31, "rise_time": 3, "peak_time": 4, "fall_time": 3, "cooldown": 12},
        {"name": "d", "amplitude": 0.8, "phase": 0.32, "rise_time": 3, "peak_time": 4, "fall_time": 3, "cooldown": 12},
        {"name": "e", "amplitude": 0.7, "phase": 0.33, "rise_time": 3, "peak_time": 4, "fall_time": 3, "cooldown": 12},
        {"name": "g", "amplitude": 0.7, "phase": 0.34, "rise_time": 3, "peak_time": 4, "fall_time": 3, "cooldown": 12},
        {"name": "o", "amplitude": 0.9, "phase": 0.3, "rise_time": 3, "peak_time": 4, "fall_time": 3, "cooldown": 12},
        {"name": "p", "amplitude": 0.7, "phase": 0.35, "rise_time": 3, "peak_time": 4, "fall_time": 3, "cooldown": 12},
        {"name": "q", "amplitude": 0.7, "phase": 0.36, "rise_time": 3, "peak_time": 4, "fall_time": 3, "cooldown": 12},
    ]
    
    # === CURVE-ONLY letters ===
    # c, o, s, u — no straight lines, all curves
    curve_only = [
        {"name": "c", "amplitude": 0.8, "phase": 0.6, "rise_time": 3, "peak_time": 5, "fall_time": 3, "cooldown": 14},
        {"name": "o", "amplitude": 0.9, "phase": 0.61, "rise_time": 3, "peak_time": 5, "fall_time": 3, "cooldown": 14},
        {"name": "s", "amplitude": 0.7, "phase": 0.62, "rise_time": 3, "peak_time": 5, "fall_time": 3, "cooldown": 14},
        {"name": "u", "amplitude": 0.7, "phase": 0.63, "rise_time": 3, "peak_time": 5, "fall_time": 3, "cooldown": 14},
    ]
    
    # === DIAGONAL letters ===
    # k, v, w, x, y, z — diagonal strokes dominant
    diagonal = [
        {"name": "k", "amplitude": 0.7, "phase": 0.8, "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 8},
        {"name": "v", "amplitude": 0.8, "phase": 0.81, "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 8},
        {"name": "w", "amplitude": 0.7, "phase": 0.82, "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 8},
        {"name": "x", "amplitude": 0.8, "phase": 0.83, "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 8},
        {"name": "y", "amplitude": 0.7, "phase": 0.84, "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 8},
        {"name": "z", "amplitude": 0.7, "phase": 0.85, "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 8},
    ]
    
    # === DESCENDER letters ===
    # g, j, p, q, y — go below the baseline
    descenders = [
        {"name": "g", "amplitude": 0.7, "phase": 0.9, "rise_time": 2, "peak_time": 3, "fall_time": 4, "cooldown": 15},
        {"name": "j", "amplitude": 0.6, "phase": 0.91, "rise_time": 2, "peak_time": 3, "fall_time": 4, "cooldown": 15},
        {"name": "p", "amplitude": 0.7, "phase": 0.92, "rise_time": 2, "peak_time": 3, "fall_time": 4, "cooldown": 15},
        {"name": "q", "amplitude": 0.7, "phase": 0.93, "rise_time": 2, "peak_time": 3, "fall_time": 4, "cooldown": 15},
        {"name": "y", "amplitude": 0.7, "phase": 0.94, "rise_time": 2, "peak_time": 3, "fall_time": 4, "cooldown": 15},
    ]
    
    # === HORIZONTAL letters ===
    # e, f, t, z — horizontal stroke dominant
    horizontal = [
        {"name": "e", "amplitude": 0.7, "phase": 0.15, "rise_time": 2, "peak_time": 4, "fall_time": 2, "cooldown": 10},
        {"name": "f", "amplitude": 0.7, "phase": 0.16, "rise_time": 2, "peak_time": 4, "fall_time": 2, "cooldown": 10},
        {"name": "t", "amplitude": 0.7, "phase": 0.17, "rise_time": 2, "peak_time": 4, "fall_time": 2, "cooldown": 10},
        {"name": "z", "amplitude": 0.7, "phase": 0.18, "rise_time": 2, "peak_time": 4, "fall_time": 2, "cooldown": 10},
    ]
    
    # === SIMPLE letters (few strokes) ===
    # c, i, j, l, o, s, u, v — 1-2 strokes
    simple = [
        {"name": "c", "amplitude": 0.8, "phase": 0.45, "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 8},
        {"name": "i", "amplitude": 0.9, "phase": 0.46, "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 8},
        {"name": "j", "amplitude": 0.6, "phase": 0.47, "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 8},
        {"name": "l", "amplitude": 0.9, "phase": 0.45, "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 8},
        {"name": "o", "amplitude": 0.9, "phase": 0.46, "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 8},
        {"name": "s", "amplitude": 0.7, "phase": 0.48, "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 8},
        {"name": "u", "amplitude": 0.7, "phase": 0.49, "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 8},
        {"name": "v", "amplitude": 0.8, "phase": 0.47, "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 8},
    ]
    
    # Merge: letters appear in multiple groups (multi-feature)
    # Use the first occurrence's amplitude, average the phase
    letter_map = {}
    for group in [vertical_ascenders, closed_loop, curve_only, diagonal, descenders, horizontal, simple]:
        for c in group:
            name = c["name"]
            if name not in letter_map:
                letter_map[name] = c.copy()
                letter_map[name]["_count"] = 1
            else:
                # Average phase, keep max amplitude
                letter_map[name]["phase"] = (letter_map[name]["phase"] + c["phase"]) / 2
                letter_map[name]["amplitude"] = max(letter_map[name]["amplitude"], c["amplitude"])
                letter_map[name]["_count"] += 1
    
    # Normalize: letters in more groups get slightly higher amplitude (more visual features)
    for name, c in letter_map.items():
        c["amplitude"] = min(1.0, c["amplitude"] * (0.8 + 0.2 * c["_count"] / 3))
        del c["_count"]
    
    return list(letter_map.values())


def test_letter_sensory():
    """Test letter sensory grounding."""
    print("=" * 60)
    print("SENSORY WAVE MODEL: Letter Visual Grounding")
    print("=" * 60)
    
    concepts = make_letter_concepts()
    print(f"\n  Letters: {len(concepts)}")
    print(f"  Simulation: 2000 ticks")
    print()
    
    lnn = create_lnn()
    sensory_environment(lnn, concepts, total_ticks=2000, verbose=False)
    
    letters = [c["name"] for c in concepts]
    
    # Tau distribution
    tau_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for sp in lnn.springs.values():
        tau_counts[sp.tau] += 1
    print(tau_distribution(tau_counts))
    
    # Visual similarity clusters
    clusters = {
        "vertical_ascenders": {"b", "d", "f", "h", "k", "l", "t"},
        "closed_loop": {"a", "b", "d", "e", "g", "o", "p", "q"},
        "curve_only": {"c", "o", "s", "u"},
        "diagonal": {"k", "v", "w", "x", "y", "z"},
        "descenders": {"g", "j", "p", "q", "y"},
    }
    
    print(f"\n  Visual Cluster Strengths:")
    for cluster_name, members in clusters.items():
        internal = 0
        cross = 0
        members_list = list(members)
        for i in range(len(members_list)):
            for j in range(i+1, len(members_list)):
                key = lnn._key(f"word:{members_list[i]}", f"word:{members_list[j]}")
                if key in lnn.springs:
                    internal += lnn.springs[key].stiffness
        
        # Cross-cluster: connections to non-members
        for m in members_list:
            node = f"word:{m}"
            if node in lnn.nodes:
                for n, sp in lnn.get_neighbors(node):
                    if n.startswith("word:") and n.replace("word:", "") not in members:
                        cross += sp.stiffness
        
        total = internal + cross
        pct = internal * 100 // max(1, total)
        print(f"    {cluster_name:25s}: internal={internal:5d}, cross={cross:5d}, {pct:3d}% internal")
    
    # Top visually similar pairs
    print(f"\n  Top Visually Similar Pairs:")
    all_pairs = []
    for i in range(len(letters)):
        for j in range(i+1, len(letters)):
            key = lnn._key(f"word:{letters[i]}", f"word:{letters[j]}")
            if key in lnn.springs:
                all_pairs.append((letters[i], letters[j], lnn.springs[key].stiffness))
    
    all_pairs.sort(key=lambda x: x[2], reverse=True)
    for a, b, stiff in all_pairs[:15]:
        print(f"    {a}-{b}: {stiff}")


if __name__ == "__main__":
    test_letter_sensory()
