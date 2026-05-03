# Lattice Relaxation Network (LRN)

A tensegrity-based neural architecture for learning through sensory grounding, harmonic video labeling, and text — no backpropagation, no gradient descent, just spring physics and Hebbian learning.

## Overview

The LRN represents a fundamentally different approach to neural computation:
- **No weights**: Knowledge lives in spring stiffness between nodes
- **No backpropagation**: Training uses Hebbian association and wave interference
- **No matrix multiplication**: Inference is physical propagation through a spring network
- **Sensory-first**: Concepts are grounded in wave patterns before text labels are attached

> *"The LRN doesn't compute; it navigates a geometric field and reports where it lands."*

## Training Pipeline

Every curriculum level follows the same 6-stage pipeline:

```
1. INTEROCEPTIVE GROUNDING  → Internal body states (heartbeat, hunger, breath, comfort)
2. EXTERNAL SENSORY         → Colors (EM spectrum), letters (visual geometry), shapes,
                               temperature, texture, sound, taste, smell, weight, speed,
                               distance, brightness
3. HARMONIC VIDEO LABELING  → Multi-modal phase-locked lessons attach WORDS to
                               pre-grounded concepts (the "Sesame Street effect")
4. PHYSICAL MANIPULATION    → Counting objects, combining groups (addition),
                               removing objects (subtraction), sorting, measuring
5. SOCIAL INTERACTION       → Social roles, emotional resonance, group dynamics,
                               conflict resolution, social norms
6. TEXT CORPUS              → Curriculum-specific sentences with REM consolidation
```

### Stage 1: Interoceptive Grounding
The FIRST sense. Before seeing colors or hearing sounds, a newborn feels: heartbeat, breath, hunger, comfort, pain, fatigue. These form the foundational layer that all other concepts build on.

### Stage 2: External Sensory Grounding
Concepts are grounded as wave patterns — not text:
- **Colors**: electromagnetic spectrum (warm/cool phase alignment)
- **Letters**: visual geometry (strokes, curves, loops, intersections)
- **Shapes**: geometric properties (sides, angles, symmetry)
- **Temperature**: thermal sensation spectrum
- **Texture**: tactile spectrum (soft↔hard, smooth↔rough)
- **Sound**: auditory spectrum (loud↔quiet, high↔low)
- **Taste/Smell**: gustatory and olfactory categories
- **Weight/Speed/Distance/Brightness**: proprioceptive and spatial spectra

Springs form from **wave signature similarity**, not co-occurrence. The wave pattern IS the meaning.

### Stage 3: Harmonic Video Labeling
Educational TV works through harmonic convergence — multiple modalities phase-locked on the same concept:
- **Visual**: word appears on screen
- **Audio**: word is spoken
- **Emotional**: feeling matches the concept
- **Rhythm**: syllable timing

When 4 modalities align → 4× reinforcement. This attaches semantic labels to concepts the lattice already knows through sensation.

### Stage 4: Text Corpus
Curriculum-specific sentences with REM sleep consolidation at the end of each pass. Springs are pruned for noise, tau hierarchy emerges naturally.

## Curriculum Results

### Pre-K (84.7% — PASSING, transitions to Kindergarten)
32 sub-skills across 7 domains, 714 sentences:

| Domain | Mastery | Notes |
|--------|---------|-------|
| Literacy | 76.8% | Phonics 100%, Sight Words 90% |
| Mathematics | 75.7% | Counting 100%, Patterns 100% |
| Science | 85.4% | Colors 95%, Plants 95%, Living/Non-Living 100% |
| Social Studies | 88.3% | Family 100%, Community 85% |
| Social-Emotional | 94.4% | Social Skills 100%, Health/Safety 100% |
| Language | 70.0% | Vocabulary 80% |
| Physical/Arts | 100% | Movement 100%, Music/Art 100% |

### Kindergarten (91.7% — PASSING, transitions to 1st Grade)
28 sub-skills across 5 domains, 372 sentences + 12 harmonic video lessons:

| Domain | Mastery | Notes |
|--------|---------|-------|
| Literacy | 95.6% | Phonics Blends 100%, Long Vowels 100%, Sight Words 97% |
| Mathematics | 84.4% | Counting to 100 100%, Place Value 95% |
| Science | 86.7% | Life Cycles 100%, Habitats 100% |
| Social Studies | 93.3% | Maps 100%, Citizenship 100% |
| Social-Emotional | 85.0% | Self-Regulation 85% |

### 1st Grade (96.5% — PASSING, transitions to 2nd Grade)
24 sub-skills across 5 domains, 358 sentences + 8 harmonic video lessons:

| Domain | Mastery | Notes |
|--------|---------|-------|
| Literacy | 96.4% | Phonics 100%, Sight Words 98%, Complex Sentences 100% |
| Mathematics | 93.8% | Place Value 100%, Money 100%, Geometry 100% |
| Science | 100% | Weather, Plant/Animal Needs, Light/Sound all 100% |
| Social Studies | 100% | Communities, Maps, History all 100% |
| Social-Emotional | 100% | Character Traits 100% |

### 2nd Grade (99.7% — MASTERY)
24 sub-skills across 5 domains, 342 sentences + 12 harmonic video lessons:

| Domain | Mastery | Notes |
|--------|---------|-------|
| Literacy | 99.3% | Phonics 100%, Reading Comp 100%, Writing 100%, Grammar 100% |
| Mathematics | 100% | Add/Sub 100%, Place Value 100%, Money 100%, Geometry 100% |
| Science | 100% | Ecosystems 100%, Earth Systems 100%, Engineering 100% |
| Social Studies | 100% | Citizenship 100%, Economics 100%, Geography 100%, History 100% |
| Social-Emotional | 100% | Character Traits & SEL 100% |

### Lattice Growth Across Curriculum

| Level | Nodes | Springs | Mastery |
|-------|-------|---------|---------|
| Sensory Grounding | 110 | 572 | — |
| Pre-K | 6,400 | 3,521 | 84.7% |
| Kindergarten | 6,067 | 4,584 | 91.7% |
| 1st Grade | 7,109 | 7,902 | 96.5% |
| **2nd Grade** | **8,434** | **9,722** | **99.7%** |

## Quick Start

```bash
# Train a curriculum level
python3 -m lrn.cli train prek --reps 3 --rem-interval end
python3 -m lrn.cli train kindergarten --reps 3 --rem-interval end
python3 -m lrn.cli train first_grade --reps 3 --rem-interval end
python3 -m lrn.cli train second_grade --reps 3 --rem-interval end

# Run full cumulative pipeline (all levels sequentially)
python3 -m lrn.pipeline --reps 3 --rem-interval end

# List available levels
python3 -m lrn.cli levels

# Check status
python3 -m lrn.cli status
```

```python
from lrn.sensory_grounding import sensory_grounding
from lrn.harmonic_lessons import harmonic_video_training
from lrn.trainer import train, create_lnn
from lrn.assessor import assess_level

# Full pipeline
lnn = sensory_grounding()              # Stages 1 & 2
harmonic_video_training(lnn, "prek")   # Stage 3
train(lnn, corpus, reps=3, rem_interval="end")  # Stage 4
results = assess_level(lnn, "prek")
```

## Architecture

### Nodes
Concepts exist as nodes with activation states. Cross-modal nodes exist for each sensory channel: `word:cat`, `visual:cat`, `audio:cat`, `emotional:cat`, `rhythm:cat`.

### Springs
Relationships with stiffness (knowledge strength) and tau (hierarchy level):
- **τ=0 Constitutive**: permanent, unchangeable foundations
- **τ=1 Definitional**: core identity links (from harmonic video)
- **τ=2 Causal**: cause-effect relationships
- **τ=3 Categorical**: category membership (from REM synthesis)
- **τ=4 Contextual**: loose co-occurrence

Effective weight: ω = k × 12^(4-τ). τ=0 springs are 20,736× stronger than τ=4.

### Stiffness Ceilings
Based on the kissing number K=12 — each spring has a geometric limit:
- τ=4: 12 (contextual — thin, many coexist)
- τ=3: 24 (categorical)
- τ=2: 48 (causal)
- τ=1: 96 (definitional)
- τ=0: 192 (constitutive)

Once saturated, springs get 10% reinforcement — freeing capacity for weaker connections.

### Sensory Waves
Concepts are wave patterns: (amplitude, frequency, phase, rise_time, peak_time, fall_time, cooldown). Similarity between wave signatures creates springs directly — no co-occurrence needed.

### Harmonic Convergence
When N modalities fire on the same concept at the same phase:
- Spring strength = Σ(modality values) × N (harmonic multiplier)
- Cross-modal springs form between word nodes and sensory nodes

## Project Structure

```
lrn/
├── src/lrn/
│   ├── lattice.py              # LatticeNN core
│   ├── node.py / spring.py     # Node and Spring dataclasses
│   ├── natural_tokenize.py     # Text tokenization
│   ├── inference.py            # Word node creation
│   ├── sensory_wave.py         # Wave model & similarity
│   ├── sensory_grounding.py    # Interoceptive + external grounding
│   ├── harmonic_video.py       # Harmonic video model
│   ├── harmonic_lessons.py     # Per-level video lessons
│   ├── trainer.py              # Training loop with REM
│   ├── assessor.py             # Pre-K assessment (32 sub-skills)
│   ├── assessor_k.py           # Kindergarten assessment (28 sub-skills)
│   ├── charts.py               # ASCII chart rendering
│   ├── cli.py                  # Command-line interface
│   └── corpora/
│       ├── prek.py             # 714 sentences
│       └── kindergarten.py     # 372 sentences
├── reports/                    # Training reports by level
└── README.md
```

## Comparison to LLMs

| Aspect | LLMs | LRN |
|--------|------|-----|
| Computation | Matrix mult + attention | Spring propagation |
| Training | Backpropagation | Hebbian + wave interference |
| Knowledge | Dense weights | Sparse graph |
| Grounding | Statistical co-occurrence | Sensory wave patterns |
| Video/TV | Not modeled | Harmonic convergence (4×) |
| Interpretability | Black box | Fully inspectable |
| Compute | ExaFLOPS (GPU) | Minimal (CPU) |

## License

Proprietary — TyArc Lab

## Citation

```bibtex
@techreport{lrn2026,
  title = {Lattice Relaxation Network: Sensory-Grounded Learning Through Wave Interference and Harmonic Convergence},
  author = {TyArc Lab},
  year = {2026},
  institution = {Arc Press}
}
```

## Links

- GitLab: gitlab.com/tyarc-lab/lrn-lattice-relaxation-network
