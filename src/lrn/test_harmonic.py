"""
Test: Harmonic Video - Attaching semantic labels to pre-grounded concepts
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn.sensory_grounding import sensory_grounding
from lrn.harmonic_video import make_label_lesson, harmonic_training

# Start with a fully grounded lattice (interoceptive + external sensory)
print("=" * 60)
print("HARMONIC VIDEO: Attaching labels to grounded concepts")
print("=" * 60)
print()

print("Step 1: Sensory grounding (the lattice already KNOWS these concepts)")
lnn = sensory_grounding(verbose=True)

print("\nStep 2: Harmonic video lessons (attaching WORDS to known concepts)")
print()

# Temperature labels - the lattice knows hot/cold waves, now we attach words
temp_lesson = make_label_lesson(
    ["hot", "warm", "cool", "cold", "freezing", "boiling"],
    "Temperature Words"
)

# Weight/size labels - lattice knows heavy/light waves, attach words
weight_lesson = make_label_lesson(
    ["heavy", "light", "big", "small", "tiny"],
    "Weight & Size Words"
)

# Speed labels - lattice knows fast/slow waves, attach words
speed_lesson = make_label_lesson(
    ["fast", "slow", "quick", "crawl", "run", "walk"],
    "Speed Words"
)

# Distance labels
distance_lesson = make_label_lesson(
    ["near", "far", "close", "away", "here", "there"],
    "Distance Words"
)

# Brightness labels
brightness_lesson = make_label_lesson(
    ["bright", "dim", "dark", "light", "shine", "glow", "shadow"],
    "Brightness Words"
)

# Sound labels
sound_lesson = make_label_lesson(
    ["loud", "quiet", "high", "low", "noisy", "whisper", "music"],
    "Sound Words"
)

# Texture labels
texture_lesson = make_label_lesson(
    ["soft", "hard", "smooth", "rough", "sticky", "wet", "fuzzy"],
    "Texture Words"
)

lessons = [temp_lesson, weight_lesson, speed_lesson, distance_lesson,
           brightness_lesson, sound_lesson, texture_lesson]

harmonic_training(lnn, lessons, verbose=True)

# Check: do the word nodes now exist and connect to each other?
print(f"\n{'='*60}")
print(f"RESULTS: After harmonic labeling")
print(f"{'='*60}")
print(f"\n  Lattice: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")

# Count cross-modal nodes
modal_nodes = sum(1 for n in lnn.nodes if ":" in n and not n.startswith("word:"))
word_nodes = sum(1 for n in lnn.nodes if n.startswith("word:"))
print(f"  Word nodes: {word_nodes}")
print(f"  Cross-modal nodes: {modal_nodes}")

# Check label-to-concept connections for key pairs
print(f"\n  Label connections (word node springs):")
test_pairs = [
    ("hot", "cold"), ("warm", "cool"), ("freezing", "boiling"),
    ("heavy", "light"), ("big", "small"), ("tiny", "big"),
    ("fast", "slow"), ("quick", "crawl"), ("run", "walk"),
    ("near", "far"), ("close", "away"), ("here", "there"),
    ("bright", "dark"), ("dim", "shine"), ("glow", "shadow"),
    ("loud", "quiet"), ("high", "low"), ("noisy", "whisper"),
    ("soft", "hard"), ("smooth", "rough"), ("sticky", "wet"),
]

for a, b in test_pairs:
    key = lnn._key(f"word:{a}", f"word:{b}")
    if key in lnn.springs:
        sp = lnn.springs[key]
        print(f"    {a:10s} ↔ {b:10s}: stiffness={sp.stiffness:5d}, tau={sp.tau}")
    else:
        print(f"    {a:10s} ↔ {b:10s}: (no spring)")
