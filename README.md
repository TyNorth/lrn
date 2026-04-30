# Lattice Relaxation Network (LRN)

A tensegrity-based neural architecture for learning and reasoning — no backpropagation, no gradient descent, just spring physics and Hebbian learning.

## Overview

The LRN represents a fundamentally different approach to neural computation:
- **No weights**: Knowledge lives in spring stiffness between nodes
- **No backpropagation**: Training uses Hebbian association ("neurons that fire together, wire together")
- **No matrix multiplication**: Inference is physical propagation through a spring network

> *"The LRN doesn't compute; it navigates a geometric field and reports where it lands."*

## Quick Start

```python
from lrn import LatticeNN, add_sentence, generate

# Create network
lnn = LatticeNN()

# Train (Hebbian learning)
add_sentence(lnn, "the cat meows")
add_sentence(lnn, "the dog barks")

# Generate
candidates = generate(lnn, ["the", "cat"], top_k=3)
print(candidates)  # [{"word": "meows", "score": 85}, ...]
```

## Key Features

### Language Learning
- 12-prompt benchmark: **100%** after Teacher Curriculum
- Trained on curated examples, not massive corpora
- Demonstrates both exact matching and generalization

### Mathematical Operations
- Basic arithmetic via number line traversal
- Complex math (fractions, decimals, percentages): **32/32 (100%)**
- No symbolic computation — physical walking on number line

### Logical Reasoning
- Syllogisms, transitivity, causal chains: **100%**
- Negation, contradictions, necessary conditions: **100%**

## Project Structure

```
lrn/
├── src/lrn/              # Core implementation
│   ├── lattice.py        # LatticeNN (nodes, springs, propagation)
│   ├── training.py       # add_sentence (Hebbian learning)
│   ├── generate.py       # generate function
│   └── math_lattice.py   # Math module (NumberLine, EquationSolver)
├── scripts/
│   ├── teacher_*.py      # Teacher Curriculum lessons
│   ├── logic_*.py        # Logical reasoning tests
│   ├── complex_math.py   # Fractions, decimals, percentages
│   └── lrn_paper.py      # Paper generator
├── docs/
│   ├── LRN_Architecture.md    # Full architecture doc
│   ├── TEACHER_CURRICULUM.md  # Training methodology
│   └── lrn_paper.pdf          # Published paper
├── sys_test/             # Test results
└── checkpoints/          # Saved network states
```

## Teacher Curriculum

Structured lessons with curated examples:

| Lesson | Focus | Result |
|--------|-------|--------|
| 1 | Basic vocabulary | 5/5 ✓ |
| 2 | Simple relationships | 4/4 ✓ |
| 4 | Gap filling | 10/12 |
| 7 | Bridge & reinforce | 11/12 |
| 10 | Exact phrase 15x | **12/12 (100%)** |

Key insight: **Quality over quantity** — 50 curated examples outperform noisy large corpora.

## Results

| Task | Score |
|------|-------|
| Language benchmark | 12/12 (100%) |
| Basic math | 38/38 (100%) |
| Complex math | 32/32 (100%) |
| Logical reasoning | 9/9 (100%) |
| Combined language + math | 8/9 (89%) |

## Architecture

- **Nodes**: Concepts (words, numbers, balance nodes)
- **Springs**: Relationships (stiffness = knowledge)
- **Propagation**: Activation flows through springs until equilibrium
- **Equilibrium**: The "answer" emerges from minimum-energy configuration

See `docs/LRN_Architecture.md` for full technical specification.

## Paper

Published as `docs/lrn_paper.pdf` — includes:
- Introduction & motivation
- Architecture (tensegrity, nodes/springs)
- Training methodology (Hebbian + Teacher Curriculum)
- Experimental results
- Comparison to modern AI (LLMs)

## Comparison to LLMs

| Aspect | LLMs | LRN |
|--------|------|-----|
| Computation | Matrix mult + attention | Spring propagation |
| Training | Backpropagation | Hebbian |
| Knowledge | Dense weights | Sparse graph |
| Interpretability | Black box | Inspectable |
| Compute | ExaFLOPS (GPU) | Minimal (CPU) |

LRN excels at transparency, energy efficiency, and biological plausibility. LLMs remain superior for general-purpose language tasks.

## License

MIT

## Citation

```bibtex
@techreport{lrn2026,
  title = {Lattice Relaxation Network: A Tensegrity-Based Neural Architecture},
  author = {TyArc Lab},
  year = {2026},
  institution = {Arc Press}
}
```

## Links

- GitHub: github.com/TyNorth/lrn
- Paper: `docs/lrn_paper.pdf`
- Architecture: `docs/LRN_Architecture.md`