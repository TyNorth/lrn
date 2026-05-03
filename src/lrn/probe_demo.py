"""
Direct probe of the 1st Grade lattice - show what it knows
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn.sensory_grounding import sensory_grounding
from lrn.harmonic_lessons import harmonic_video_training
from lrn.trainer import train
from lrn.corpora import get_corpus
from lrn import propagate

print("Loading 1st Grade lattice...")
lnn = sensory_grounding(verbose=False)
harmonic_video_training(lnn, 'first_grade', verbose=False)
corpus = get_corpus('first_grade')
train(lnn, corpus, reps=3, rem_interval='end', verbose=False)
print(f"Loaded: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs\n")


def probe(lnn, word, steps=2):
    """Probe a specific word node and show its strongest connections."""
    node = f"word:{word}"
    if node not in lnn.nodes:
        return None, []
    
    # Reset and activate
    for n in lnn.nodes.values():
        n.activation = 0
    lnn.nodes[node].activation = 100
    
    propagate(lnn, n_steps=steps)
    
    # Get neighbors sorted by activation
    neighbors = []
    for (a, b), sp in lnn.springs.items():
        if a == node:
            neighbor = b
        elif b == node:
            neighbor = a
        else:
            continue
        
        if neighbor.startswith("word:"):
            w = neighbor.replace("word:", "")
            act = lnn.nodes.get(neighbor, None)
            if act:
                neighbors.append((w, sp.stiffness, act.activation, sp.tau))
    
    neighbors.sort(key=lambda x: x[1], reverse=True)
    return lnn.nodes[node], neighbors


print("=" * 60)
print("PROBE: What does the 1st Grade lattice know?")
print("=" * 60)

probes = [
    ("red", "Color (sensory grounded)"),
    ("hot", "Temperature (sensory grounded)"),
    ("fast", "Speed (sensory grounded)"),
    ("cat", "Animal (text corpus)"),
    ("triangle", "Shape (sensory + text)"),
    ("plus", "Math operation (harmonic video)"),
    ("community", "Social studies (harmonic video + text)"),
    ("honesty", "Character trait (harmonic video + text)"),
    ("weather", "Science (text corpus)"),
    ("story", "Literacy (text corpus)"),
    ("because", "Complex connector (harmonic video)"),
    ("empathy", "Character trait (harmonic video)"),
]

for word, desc in probes:
    node, neighbors = probe(lnn, word)
    
    print(f"\n  ── {word.upper()} ({desc}) ──")
    
    if node is None:
        print(f"    Node not found")
        continue
    
    # Show spring count and avg stiffness
    if neighbors:
        avg_stiff = sum(s for _, s, _, _ in neighbors) // len(neighbors)
        print(f"    Springs: {len(neighbors)}, Avg stiffness: {avg_stiff}")
        print(f"    Top connections:")
        for w, stiff, act, tau in neighbors[:8]:
            tau_label = {0: "const", 1: "def", 2: "causal", 3: "cat", 4: "ctx"}[tau]
            print(f"      {w:15s} stiffness={stiff:5d}  τ={tau_label}  activation={act}")
    else:
        print(f"    No connections")
