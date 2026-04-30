# LRN Teacher Curriculum - Documentation

## Overview
The LRN Teacher approach treats the network like a student being taught by a teacher - structured lessons, curated examples, repetition, and progressive complexity.

## Key Principles

### 1. Curated Corpus (Quality over Quantity)
- Each example is carefully crafted to teach a specific concept
- Avoid noise and ambiguous examples
- Use clear subject-relationship-object patterns

### 2. Structured Lessons (Progressive Learning)
Each lesson builds on previous ones:
- **Lesson 1**: Basic vocabulary (nouns, verbs)
- **Lesson 2**: Simple relationships (spatial, causal, temporal)
- **Lesson 3**: Categories and properties
- **Lesson 4**: Gap filling (address benchmark failures)
- **Lesson 5-6**: Advanced concepts (time, quantity, comparisons)
- **Lesson 7**: Bridge & reinforce (connect existing concepts)
- **Lesson 8-10**: Targeted fixes (resolve remaining failures)

### 3. Repetition with Purpose
- 3-5x for normal concepts
- 8-15x for hard cases requiring override
- Each repetition strengthens Hebbian springs

### 4. Context Resolution
When ambiguous, add more context examples:
- "the sun" → give more "sky" context examples
- Avoid generic patterns that cause interference

### 5. Bridge Concepts
Adding new concepts reinforces existing ones:
- Connect actions to objects
- Connect categories to specifics
- This explains why new lessons don't break old ones

## Results

### Benchmark Progression
| Lesson | Parity |
|--------|--------|
| 1 | 5/5 (vocab only) |
| 4 | 83% |
| 7 | 92% |
| 10 | **100%** |

### Final Network
- Nodes: 130 (vs ~179 in phase-based)
- Springs: 442 (vs ~1933 in phase-based)
- More efficient: smaller network, better results

## Key Insight
**"Junk in, junk out"** - the quality of training data matters more than quantity. A curated 50 examples with proper repetition beats a noisy 10,000 example corpus.

## Generalization Findings

### Exact Phrase Matching (Lesson 10)
- **100%** on original 12-prompt benchmark
- Network learns specific phrases, not abstract rules

### Novel Combinations (Generalization Test)
- **1/13 (7.7%)** - network cannot generalize to unseen combinations
- Hebbian learning creates associations, not abstractions

### Trade-off: Diversity vs Accuracy
| Approach | Exact Match | Generalization |
|----------|-------------|-----------------|
| Exact phrase (L10) | 100% | ~8% |
| Diverse training (L13) | 44% | ~44% |

**Conclusion**: Current LRN excels at exact phrase completion. True generalization requires either:
1. Much more diverse training (many variations of each pattern)
2. Additional abstraction layer for rule extraction
3. Larger network with more training iterations

## Files
- `scripts/teacher_lesson*.py`: Each lesson
- `checkpoints/lesson*.pkl`: Saved network states
## Experiment Results

### Exact vs Diverse Trade-off
| Experiment | Description | Exact | Generalization | Total |
|------------|-------------|-------|-----------------|-------|
| L10 | Exact only (15x) | 12/12 | 1/13 | 13/25 (52%) |
| L13 | Diverse only (3x) | 7/16 | ~50% | 7/16 (44%) |
| Exp A | Diverse (5 epochs) | 8/15 | 53% | 8/15 (53%) |
| Exp B | Exact + Diverse | 11/12 | 2/4 | 13/16 (81%) |
| Exp B2 | B with more epochs | 11/12 | 2/6 | 13/18 (72%) |

**Best: Experiment B (81%)** - exact first (15x), then diverse (3x), preserves exact while adding some generalization.

### Key Finding
More epochs on exact doesn't help generalization. Sweet spot: exact 15x + diverse 3x.

## Files
## Logical Reasoning Results

### Lesson 1: Syllogisms (All A are B → X is B)
| Pattern | Result |
|---------|--------|
| "all dogs are animals" → animals | ✓ |
| "birds are animals" → animals | ✓ |
| "the dog is an animal" | ✓ (inference) |

### Lesson 2: If-Then & Comparisons
| Pattern | Result |
|---------|--------|
| If-Then ("if rain then wet") | 5/6 (83%) |
| Comparisons ("bigger than small") | 2/3 (67%) |

### Lesson 3: Transitivity & Chains
| Pattern | Result |
|---------|--------|
| Transitivity (A>B, B>C → A>C) | 3/3 (100%) |
| Causal chains (rain→wet→cold) | 3/3 (100%) |
| Sequences (morning→noon→night) | 1/2 (50%) |

### Combined: Language + Logic
| Feature | Result |
|---------|--------|
| Language benchmark | 6/7 (86%) |
| Logical reasoning | 5/5 (100%) |

### Logical Reasoning Advanced (Lesson 4)
| Pattern | Result |
|---------|--------|
| Negation ("not", "opposite") | 4/4 (100%) |
| Necessary ("necessary for") | 3/3 (100%) |
| Contradictions ("but", "however") | 2/2 (100%) |

### Math + Language Combined
| Component | Result |
|-----------|--------|
| Language (word associations) | 3/4 (75%) |
| Math (word problems) | **5/5 (100%)** |
| **Combined** | **8/9 (89%)** |

**Key finding**: 
- Math word problems work perfectly with number line traversal
- Word-to-number bridge (one→1, two→2) enables language→math connection
- LRN handles both pattern recognition and numeric computation

**Key finding**: LRN learns logical patterns well - transitivity and causal chains work better than abstract sequences.

## Files
- `scripts/teacher_lesson*.py`: Each lesson
- `scripts/logic_lesson*.py`: Logical reasoning lessons
- `scripts/experiment_*.py`: Experiments
- `scripts/combined_*.py`: Combined language + logic
- `checkpoints/`: Saved network states
- `sys_test/`: Test results