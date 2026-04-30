# Phase 2 Report: Training Pipeline

## Date: 2026-04-30

## Goal
Process corpus, form springs via Hebbian co-activation, build n-gram table.

## Implementation
- `src/lrn/corpus.py` - CorpusExpander with 38 core + 462 expanded sentences
- `src/lrn/training.py` - add_sentence(), add_negative_sentence(), train_corpus()
- Added identity:self anchor node

## Results

| Metric | Value |
|--------|-------|
| Sentences processed | 500 |
| Nodes created | 179 |
| Springs created | 1933 |
| Trigrams stored | 1618 |
| Positive springs | 1923 |
| Negative springs | 10 |

### Sample Data
- Node "fly" roles: `{1: 4, 4: 9, 2: 4, 3: 3}` (ACTOR, CLOSER, LINKER, SETTLER)
- Spring "fly-in": k=43, tau=4 (strong contextual)
- Trigram "in the sky": 22 occurrences

## Test
- Propagation test: "birds fly in the sky" → 170 active nodes
- Negative springs: fish↔fly, burn↔ice, etc. (k=-5)

## Status: ✓ PASS

## Scripts
- `scripts/train_phase2.py` - ✓ PASSED
- `scripts/eval_phase2.py` - ✓ PASSED

## Checkpoint
`checkpoints/phase2.pkl`

## Next Phase
Phase 3: Generation - gravity formula, generate() function

## Status: APPROVED