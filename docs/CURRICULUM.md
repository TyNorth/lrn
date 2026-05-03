# Educational Lattice - Curriculum Documentation

## Philosophy
The LatticeNN educational system builds knowledge cumulatively from sensory grounding upward through language, grammar, and eventually abstract concepts like computer science and advanced mathematics. Each level builds on the previous, simulating real child development.

## Curriculum Spheres

### 1. Sensory Grounding (Foundation)
**When**: Pre-birth through Kindergarten

**Purpose**: Multi-modal concept grounding before text training

**Components**:
- **Interoception** (internal body states): hunger, thirst, comfort, heartbeat, breath, fatigue, warmth, energy, pain
- **External Senses**: Colors (warm/cool), Letters (visual geometry), Shapes (geometric properties), Temperature (thermal), Texture (tactile), Sound (auditory), Taste (gustatory), Smell (olfactory), Weight (proprioceptive), Speed (kinesthetic), Distance (spatial), Brightness (visual)
- **Sensory Interoceptive Lessons** (K-3rd): Body awareness, self-regulation, sensory processing

**Files**:
- `sensory_grounding.py` - Core sensory wave grounding
- `sensory_interoceptive.py` - K-3rd interoceptive curriculum lessons

### 2. Language Corpus (All Grades)
**When**: Pre-K through College

**Purpose**: Build vocabulary and comprehension through grade-level corpora

**Corpora**: `corpora/` directory with level-specific modules:
- Pre-K: 714 sentences (phonics, colors, shapes, basic vocabulary)
- Kindergarten: 372 sentences (letters, numbers, sight words, simple sentences)
- 1st-8th Grade: Increasing complexity across literacy, math, science, social studies
- High School: Algebra, geometry, biology, chemistry, physics, history, government
- College: Calculus, differential equations, organic chemistry, electricity and magnetism, thermodynamics, quantum mechanics, linear algebra, world history, microeconomics, macromarketing, differential psychology, philosophy, political science

### 3. Grammar Spiral (1st-8th Grade)
**When**: Integrated with language corpus at each grade

**Purpose**: Cumulative grammar concepts building year over year

**Scope**:
- **1st Grade** (40 lessons): Sentences, subjects/predicates, nouns (people/places/animals/things), verbs, adjectives, capitalization, punctuation, pronouns, simple plurals
- **2nd Grade** (40 lessons): Complete vs incomplete, common vs proper nouns, past/present/future tense, singular/plural, articles a/an/the, contractions, pronoun-antecedent, adverbs
- **3rd Grade** (40 lessons): Subject-verb agreement, progressive tenses, complex sentences (because, when, although), adverbs, apostrophes, conjunctions, four sentence types, quotation marks
- **4th Grade** (15 lessons): Simple/compound/complex sentences, pronouns (subjective/objective/possessive), verbs (action/linking/helping), adjectives (comparative/superlative), prepositions, conjunctions, interjections, commas/quotes/apostrophes
- **5th Grade** (15 lessons): Verbals (participles, gerunds, infinitives), clauses (independent/dependent/noun), pronoun-antecedent agreement, precise word choice, sentence combining, semicolons/colons, quotation marks, dashes/hyphens, parallel structure, active/passive voice
- **6th-8th Grade** (4 lessons each): Analysis and evaluation, research and source evaluation, argument and persuasion, figurative language, advanced grammar structures, complex sentence structure, formal writing style, synthesis and integration, rhetorical analysis, literary criticism

**Files**: `grammar_lessons*.py` for each grade level

### 4. Computer Logic and Programming (K-8th)
**When**: Integrated with language corpus starting at K

**Purpose**: Build computational thinking and programming concepts cumulatively

**Scope**:
- **K-1st**: Sequential thinking, patterns, step-by-step instructions, following directions, finding errors
- **2nd-3rd**: If/then logic, loops, task decomposition, true/false evaluation
- **4th-5th**: Algorithms, debugging, functions, arrays/lists, Boolean logic (AND/OR/NOT), variables/data types, loops (for/while), pseudo-code
- **6th-8th**: Objects/classes/instances, events/event handlers, stacks/queues, Big O notation, recursion, searching/sorting algorithms, OOP design, APIs, Boolean algebra/logic gates, problem decomposition/abstraction

**Files**:
- `computer_logic.py` - K-3rd programming foundations
- `computer_logic_4th_8th.py` - 4th-8th advanced concepts

## Pipeline Execution Order

1. **PHASE 0**: Sensory Grounding (interoception + 12 external modalities)
2. **PHASE 0.5**: Sensory/Interoceptive Lessons (K-3rd curriculum)
3. **PHASE 1**: Pre-K corpus training
4. **PHASE 2**: Kindergarten corpus training
5. **PHASE 3-N**: 1st through 8th grade (each adds grammar + programming + corpus)

## File Structure

```
src/lrn/
├── sensory_grounding.py          # Foundation: interoception + modalities
├── sensory_interoceptive.py       # K-3rd body awareness curriculum
├── computer_logic.py             # K-3rd programming foundations
├── computer_logic_4th_8th.py     # 4th-8th programming concepts
├── grammar_lessons.py            # 1st grade grammar (40 lessons)
├── grammar_lessons_2nd.py        # 2nd grade grammar (40 lessons)
├── grammar_lessons_3rd.py        # 3rd grade grammar (40 lessons)
├── grammar_lessons_4th.py         # 4th grade grammar
├── grammar_lessons_5th.py        # 5th grade grammar
├── grammar_lessons_6th_8th.py     # 6th-8th grade grammar
├── grammar_workbook.py           # Practice/assessment generation
├── grammar_assessor.py           # Grammar-specific assessment
├── corpora/                      # Grade-level language corpora
│   ├── prek.py, kindergarten.py
│   ├── first_grade.py, second_grade.py, third_grade.py
│   ├── fourth_grade.py, fifth_grade.py, sixth_grade.py, seventh_grade.py, eighth_grade.py
│   └── ...
├── assessor_*.py                 # Grade-level curriculum assessors
└── full_k_to_8th.py              # Main pipeline script
```

## Running the Pipeline

```bash
# Full K-8th pipeline with grammar and programming
python3 src/lrn/full_k_to_8th.py

# Assessment for specific grade
python3 -c "
import pickle
import sys
sys.path.insert(0, 'src')
from lrn.assessor_4th import assess_fourth_grade
# load lattice and assess
"
```

## Assessment

Each grade assessor (e.g., `assessor_4th.py`) includes domain-specific tests:
- Math
- Literacy (reading, writing, vocabulary)
- Science
- Social Studies
- Grammar (grammar spiral)
- Programming Logic (computer science spiral)

Assessment uses spring connection counts to determine mastery:
- PASS: Node exists with 2+ connections to expected neighbors
- PARTIAL: Node exists but limited connections
- FAIL: Node does not exist