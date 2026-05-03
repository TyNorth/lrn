"""
Direct probe of the 2nd Grade lattice
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn.sensory_grounding import sensory_grounding
from lrn.harmonic_lessons import harmonic_video_training
from lrn.physical_manipulation import physical_manipulation
from lrn.social_interaction import social_interaction
from lrn.trainer import train
from lrn.corpora import get_corpus
from lrn import propagate

print("Loading 2nd Grade lattice...")
lnn = sensory_grounding(verbose=False)
harmonic_video_training(lnn, 'second_grade', verbose=False)
physical_manipulation(lnn, verbose=False)
social_interaction(lnn, verbose=False)
corpus = get_corpus('second_grade')
train(lnn, corpus, reps=3, rem_interval='end', verbose=False)
print(f"Loaded: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs\n")


def probe(lnn, word, steps=2):
    """Probe a specific word node and show its strongest connections."""
    node = f"word:{word}"
    if node not in lnn.nodes:
        return None, []
    
    for n in lnn.nodes.values():
        n.activation = 0
    lnn.nodes[node].activation = 100
    
    propagate(lnn, n_steps=steps)
    
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
print("PROBE: What does the 2nd Grade lattice know?")
print("=" * 60)

probes = [
    ("character", "Reading comprehension"),
    ("plot", "Reading comprehension"),
    ("theme", "Reading comprehension"),
    ("noun", "Grammar"),
    ("conjunction", "Grammar"),
    ("multiplication", "Math"),
    ("fraction", "Math"),
    ("area", "Math"),
    ("numerator", "Math"),
    ("ecosystem", "Science"),
    ("climate", "Science"),
    ("force", "Science"),
    ("gravity", "Science"),
    ("government", "Social Studies"),
    ("constitution", "Social Studies"),
    ("continent", "Geography"),
    ("empathy", "SEL"),
    ("resilience", "SEL"),
    ("integrity", "SEL"),
    ("teamwork", "SEL"),
    ("penny", "Money - check if it's flooding"),
]

for word, desc in probes:
    node, neighbors = probe(lnn, word)
    
    print(f"\n  ── {word.upper()} ({desc}) ──")
    
    if node is None:
        print(f"    Node not found")
        continue
    
    if neighbors:
        avg_stiff = sum(s for _, s, _, _ in neighbors) // len(neighbors)
        print(f"    Springs: {len(neighbors)}, Avg stiffness: {avg_stiff}")
        print(f"    Top connections:")
        for w, stiff, act, tau in neighbors[:8]:
            tau_label = {0: "const", 1: "def", 2: "causal", 3: "cat", 4: "ctx"}[tau]
            print(f"      {w:15s} stiffness={stiff:5d}  τ={tau_label}  activation={act}")
    else:
        print(f"    No connections")
