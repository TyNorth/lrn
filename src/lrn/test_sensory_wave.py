"""
Test: Sensory Wave Model for Color Learning

Colors as electromagnetic sensory events with wave properties.
Tests constructive vs destructive interference patterns.
"""
import sys
import math
import random
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn.sensory_wave import sensory_environment, create_lnn
from lrn.charts import bar_chart, tau_distribution

# Color concepts as episodic sensory waves
# Warm colors share phase (sunset context) → constructive interference
# Cool colors share phase (sky/water context) → constructive interference
# Warm vs cool are out of phase → destructive interference (minimal overlap)

COLOR_CONCEPTS = [
    # Warm colors - fire/sunset context, phase-aligned, fire together
    {"name": "red", "amplitude": 0.9, "frequency": 0.8, "phase": 0.0, 
     "rise_time": 3, "peak_time": 5, "fall_time": 4, "cooldown": 12},
    {"name": "orange", "amplitude": 0.8, "frequency": 0.8, "phase": 0.02, 
     "rise_time": 3, "peak_time": 5, "fall_time": 4, "cooldown": 12},
    {"name": "yellow", "amplitude": 0.9, "frequency": 0.8, "phase": 0.04, 
     "rise_time": 3, "peak_time": 5, "fall_time": 4, "cooldown": 12},
    
    # Cool colors - sky/water context, phase-aligned with each other
    # Phase 0.5 = half cycle offset from warm → minimal temporal overlap
    {"name": "blue", "amplitude": 0.9, "frequency": 0.7, "phase": 0.5, 
     "rise_time": 4, "peak_time": 6, "fall_time": 5, "cooldown": 15},
    {"name": "green", "amplitude": 0.8, "frequency": 0.7, "phase": 0.52, 
     "rise_time": 4, "peak_time": 6, "fall_time": 5, "cooldown": 15},
    {"name": "purple", "amplitude": 0.7, "frequency": 0.7, "phase": 0.54, 
     "rise_time": 4, "peak_time": 6, "fall_time": 5, "cooldown": 15},
    
    # Neutrals - different contexts, low frequency, high cooldown
    {"name": "black", "amplitude": 0.4, "frequency": 0.3, "phase": 0.25, 
     "rise_time": 2, "peak_time": 3, "fall_time": 3, "cooldown": 25},
    {"name": "white", "amplitude": 0.4, "frequency": 0.3, "phase": 0.27, 
     "rise_time": 2, "peak_time": 3, "fall_time": 3, "cooldown": 25},
    {"name": "brown", "amplitude": 0.5, "frequency": 0.4, "phase": 0.75, 
     "rise_time": 3, "peak_time": 4, "fall_time": 4, "cooldown": 20},
    {"name": "pink", "amplitude": 0.3, "frequency": 0.2, "phase": 0.1, 
     "rise_time": 2, "peak_time": 2, "fall_time": 2, "cooldown": 30},
]

print("=" * 60)
print("SENSORY WAVE MODEL: Color Learning (Episodic)")
print("=" * 60)
print(f"\n  Concepts: {len(COLOR_CONCEPTS)}")
print(f"  Simulation: 1000 ticks")
print()

lnn = create_lnn()
sensory_environment(lnn, COLOR_CONCEPTS, total_ticks=1000, verbose=True)

# Assess color clustering
colors = [c["name"] for c in COLOR_CONCEPTS]
print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")

# Tau distribution
tau_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
for sp in lnn.springs.values():
    tau_counts[sp.tau] += 1
print(tau_distribution(tau_counts))

# Color-to-color spring strengths
print(f"\n  Color Spring Strengths (top connections):")
for c in colors:
    node = f"word:{c}"
    if node in lnn.nodes:
        neighbors = lnn.get_neighbors(node)
        color_neighbors = []
        for n, sp in neighbors:
            if n.startswith("word:") and n.replace("word:", "") in colors:
                color_neighbors.append((n.replace("word:", ""), sp.stiffness))
        color_neighbors.sort(key=lambda x: x[1], reverse=True)
        
        if color_neighbors:
            top = color_neighbors[:3]
            top_str = ", ".join(f"{name}({stiff})" for name, stiff in top)
            print(f"    {c:8s} → {top_str}")
        else:
            print(f"    {c:8s} → (no color connections)")

# Warm vs cool clustering
warm = {"red", "orange", "yellow"}
cool = {"blue", "green", "purple"}

warm_internal = 0
cool_internal = 0
cross = 0

for i in range(len(colors)):
    for j in range(i+1, len(colors)):
        key = lnn._key(f"word:{colors[i]}", f"word:{colors[j]}")
        if key in lnn.springs:
            sp = lnn.springs[key]
            a, b = colors[i], colors[j]
            if a in warm and b in warm:
                warm_internal += sp.stiffness
            elif a in cool and b in cool:
                cool_internal += sp.stiffness
            else:
                cross += sp.stiffness

print(f"\n  {'='*40}")
print(f"  Warm cluster (red/orange/yellow): {warm_internal}")
print(f"  Cool cluster (blue/green/purple): {cool_internal}")
print(f"  Cross-cluster connections:        {cross}")
print(f"  {'='*40}")

if warm_internal > cross and cool_internal > cross:
    print(f"\n  ✓ Constructive interference within clusters")
    print(f"  ✓ Destructive interference between clusters")
else:
    print(f"\n  ✗ Clustering not yet established")
