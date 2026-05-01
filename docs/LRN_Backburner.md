# LRN Backburner: Deferred Capabilities

**Date**: 2026-05-01
**Status**: Documented for future integration

This document captures capabilities discovered in Sovereign Mind v31/v32 and Sovereign Data Science notebooks that are **not yet integrated** into the LRN architecture. They are deferred, not discarded.

---

## 1. Vision Capabilities (Sovereign DLSS)

**Source**: `/Users/tyarc/github/ste/public/8/SI/sovereign_dlss.ipynb`, `sovereign_dlss2.ipynb`

### Overview
Integer-only visual processing pipeline implementing a DLSS-like upscaler entirely with integer arithmetic. No floats, no neural networks.

### Key Components

**1.1 IVM Frame Generation**
- Generates full-resolution integer pixel frames from lattice parameters (rho, sigma, gamma)
- Three biome generators: `gen_groundweave` (lattice lines), `gen_deepcant` (crystalline shimmer), `gen_deadcant` (superposed strands)
- Pixel computation: saturation occlusion → base brightness → phase shimmer → strand weighting (Z-buffer logic)

**1.2 Integer Downsampler/Upsampler**
- 2x downsample via block averaging: `block_sum >> 2` (integer-only)
- Nearest-neighbor upsample baseline for comparison

**1.3 Integer Upscale Kernel**
- 9-tap, 4-subpixel-position (TL, TR, BL, BR) weight matrix = 36 integer parameters
- Each row sums to 256 (power-of-2 for bit-shift divide)
- Initial kernel: bilinear `[16, 32, 16, 32, 64, 32, 16, 32, 16]`

**1.4 Jitterbug Kernel Trainer**
- Simulated annealing with integer-only operations
- Mutation: pick random position + two taps, apply +delta/-delta (maintains row sum)
- Metropolis acceptance: `exp(-delta / temperature)` approximated as integer comparison
- Cooling: `temp = max(1, temp * 95 // 100)` (5% per tick)
- 600 iterations, 240 training frames, 48 validation frames

**1.5 Temporal EMA Accumulator**
- `ctx = (7 * ctx + new_frame) // 8` (alpha = 7/8 = 87.5% previous frame weight)
- Context stored as scaled integer (×8) — no floats
- Pipeline: spatial kernel → temporal EMA → output

**1.6 Frame Budget Anomaly Detection**
- Three anomaly types: POINT (latency spike), CONTEXTUAL (strand imbalance), COLLECTIVE (trend creep)
- 1024-bin histogram over [0, 50ms] range
- 99th percentile threshold for point anomalies
- Integer EMA per strand (6 strands)
- Warmup: 64 ticks before flagging begins

**1.7 Zig Export**
- Trained kernel exported as `const` array for direct Zig renderer integration
- Zero-copy blit to CoreGraphics

### Constants
| Constant | Value | Purpose |
|----------|-------|---------|
| W_FULL/H_FULL | 128 | Full resolution |
| W_HALF/H_HALF | 64 | Render target |
| KERNEL_R | 1 | 9-tap kernel |
| UPSCALE_DENOM | 256 | Bit-shift divide |
| N_STRANDS | 6 | IVM strand arrays |
| UPSILON_VAC | 1000 | Sigma saturation |
| GAMMA_MAX | 1800 | Max torsional phase |
| JB_TEMP_INIT | 4000 | Annealing start temp |
| JB_COOL_NUM/DEN | 95/100 | 0.95 cooling |
| JB_ITERS | 600 | Training iterations |
| EMA_ALPHA_NUM/DEN | 7/8 | Temporal weight |
| HIST_BINS | 1024 | Latency histogram |
| Q_THRESHOLD | 990 | 99.0th percentile |

### Integration Path
- Vision nodes would use `modality: "V"` (already in Node dataclass)
- Visual tokens would form springs with text tokens (bridge nodes, `modality: "B"`)
- Same spring hierarchy applies: visual co-occurrence → τ=4, witnessed visual facts → τ=0
- Frame anomaly detection maps to LRN epistemic foraging (unresolved_tension accumulator)

---

## 2. SMT Solver (Standalone)

**Source**: `/Users/tyarc/github/ste/public/8/SI/SMT.ipynb`

**Note**: This will be documented in the **Sovereign Data Science paper** as a standalone module, not integrated into LRN.

### Overview
Complete Satisfiability Modulo Theories engine with:
- Pratt precedence climbing parser (5 levels: IFF, IMPLIES, OR, AND, NOT)
- Unicode + ASCII operator support (¬, ∧, ∨, →, ↔)
- AST-to-CNF via Tseitin transformation
- DPLL SAT solver with unit propagation and MOM heuristic
- Bounded integer constraint solver with interval pruning
- Reflective domain-switching architecture (Trusted Kernel + Sovereign Controller)
- Telemetry-driven heuristic switching (stagnation detection)
- Proof objects and UNSAT core extraction

### Constants
| Constant | Value | Purpose |
|----------|-------|---------|
| max_steps | 5000 | Solver step limit |
| max_depth | 50 | Branch depth limit |
| Stagnation trigger | ≥3 | Heuristic switch |
| EMA telemetry weights | 0.8/0.2 | Signal smoothing |

---

## 3. Audio Capabilities

**Source**: Deferred — not yet extracted from source materials.

### Planned Approach
Audio would follow the same LRN paradigm:
- Audio sensor channels: `sensor:audio:{frequency_band}:{level}` (already in §3.6)
- Frequency bands map to spring formation (co-occurring sounds → springs)
- Temporal patterns in audio → n-gram table (same as text/code)
- Audio-visual-text bridge nodes → cross-modal spring formation

---

## 4. Additional Deferred Items

**4.1 Geometric Tokenization (3D Ternary Crystals)**
- Source: `02_ternary_tokenization.ipynb`
- SHA-256 hashed words → 3D ternary crystal growth via IVM random walks
- Currently LRN uses simple string-based node names
- Could replace or augment node naming with geometric crystal signatures
- Lattice dim: 12, token_radius: 3, growth steps: 12 + len(word)

**4.2 Discrete Attention (Integer-Only)**
- Source: `03_discrete_architecture.ipynb`
- Q@K.T with integer matrix multiply, hard Top-K=12 gating
- Could replace or augment the gravity formula (§6.3)
- Token dim: 27 (3x3x3 neighborhood), attention threshold: 3

**4.3 Anomaly Detection (General)**
- Source: `anomaly_detection.ipynb`
- Three-tier anomaly detection (POINT/CONTEXTUAL/COLLECTIVE)
- Could map to LRN's unresolved_tension and epistemic foraging
- FrameBudgetMonitor patterns applicable to training pipeline monitoring

---

## Priority Order for Future Integration

1. **Vision** — Highest impact, already fully implemented in sovereign_dlss
2. **Audio** — Natural extension of existing sensor channels
3. **Geometric Tokenization** — Replaces string-based node naming with lattice-native representation
4. **Discrete Attention** — Potential optimization for candidate scoring
5. **General Anomaly Detection** — Training pipeline robustness

---

**End of Backburner Document**
