"""
Generation module for LatticeNN
- Gravity formula for candidate scoring
- generate() - single token prediction
- generate_sequence() - multi-token generation
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, propagate, _is_stable
from lrn.lattice import TAU_W, E_THRESHOLD


ALPHA = 410
BETA = 307
GAMMA = 307
DELTA = 461
CAUSAL_BOOST = 5120
K_BASE = 1024


def _next_expected_role(context: list) -> int:
    if not context:
        return 0
    n = len(context)
    frac = n / max(1, n + 1)
    if frac < 0.15:
        return 0
    elif frac < 0.35:
        return 1
    elif frac < 0.65:
        return 2
    elif frac < 0.85:
        return 3
    else:
        return 4


def _calculate_node_energy(lnn: LatticeNN, node_name: str) -> int:
    total_energy = 0
    degree = 0
    for (a, b), sp in lnn.springs.items():
        if a == node_name or b == node_name:
            if a in lnn.nodes and b in lnn.nodes:
                node_a = lnn.nodes[a]
                node_b = lnn.nodes[b]
                if node_a.activation > 0 or node_b.activation > 0:
                    total_energy += sp.energy
                    degree += 1
    if degree == 0:
        return 0
    return total_energy // degree


def _score_candidates(lnn: LatticeNN, context: list, top_k: int = 5) -> list:
    candidates = []
    last_word = context[-1] if context else None
    expected_role = _next_expected_role(context)

    for name, node in lnn.nodes.items():
        if name.startswith("k:") or name.startswith("sensor:"):
            continue
        if name.startswith("identity:"):
            continue
        if name in context:
            continue
        if node.activation < 5:
            continue
        if node.pinned:
            continue

        H = 0
        for n in [3, 4, 5]:
            if len(context) >= n - 1:
                gram = tuple(context[-(n-1):] + [name])
                count = lnn.trigrams.get(gram, 0)
                H += count * 50

        S = 0
        if last_word:
            sp = lnn.springs.get(lnn._key(last_word, name))
            if sp:
                S = max(0, sp.stiffness) * node.activation // 100

        D = node.activation

        C = 0
        for i, ctx_word in enumerate(context):
            sp = lnn.springs.get(lnn._key(ctx_word, name))
            if sp and sp.stiffness > 0:
                decay = max(1, len(context) - i)
                C += sp.stiffness * node.activation // (decay * 100)

        raw_score = (H * ALPHA + S * BETA + D * GAMMA + C * DELTA) // K_BASE

        c_boost = K_BASE
        if last_word:
            sp = lnn.springs.get(lnn._key(last_word, name))
            if sp and sp.tau <= 2 and sp.directional:
                c_boost = CAUSAL_BOOST

        phi = K_BASE
        if node.dominant_role == expected_role:
            phi = K_BASE + 300
        elif abs(node.dominant_role - expected_role) == 1:
            phi = K_BASE + 150

        final_score = (raw_score * c_boost * phi) // (K_BASE * K_BASE)

        node_energy = _calculate_node_energy(lnn, name)
        if node_energy > E_THRESHOLD:
            continue

        candidates.append({
            "word": name,
            "score": final_score,
            "energy": node_energy,
            "role": node.dominant_role,
            "activation": node.activation,
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:top_k]


def generate(lnn: LatticeNN, prompt: list, top_k: int = 5, verbose: bool = False) -> list:
    lnn.reset()

    for tok in prompt:
        node = lnn.nodes.get(tok)
        if node:
            node.activation = 100
            node.pinned = True

    if "identity:self" in lnn.nodes:
        lnn.nodes["identity:self"].activation = 100
        lnn.nodes["identity:self"].pinned = True

    for step in range(5):
        propagate(lnn, n_steps=1, verbose=False)
        if _is_stable(lnn):
            if verbose:
                print(f"  Stable at step {step + 1}")
            break

    return _score_candidates(lnn, prompt, top_k)


def generate_sequence(lnn: LatticeNN, prompt: str, max_tokens: int = 10, top_k: int = 5) -> str:
    context = prompt.lower().split()
    generated = []

    for _ in range(max_tokens):
        candidates = generate(lnn, context, top_k=top_k)
        if not candidates:
            break

        best = candidates[0]["word"]
        generated.append(best)
        context.append(best)

    return " ".join(generated)