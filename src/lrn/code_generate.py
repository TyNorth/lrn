"""
Code generation for LRN.
- generate_code: same as generate() but stricter energy gate (20 vs 48)
- validate_syntax: post-generation τ=0 spring validation
- generate_code_sequence: autoregressive code generation with syntax checks
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN
from lrn.generate import generate, _calculate_node_energy
from lrn.lattice import E_THRESHOLD

CODE_E_THRESHOLD = 20
CODE_CANDIDATE_MULTIPLIER = 3


def generate_code(lnn: LatticeNN, prompt: list, top_k: int = 5) -> list:
    """
    Code generation: identical to language generation but with stricter energy gate.

    τ=0 syntax springs with k=100: a broken spring adds 50 energy minimum.
    Only candidates with ALL syntax springs closed pass E < 20.
    """
    # Get broader candidate pool, then filter by stricter energy
    candidates = generate(lnn, prompt, top_k=top_k * CODE_CANDIDATE_MULTIPLIER)

    return [c for c in candidates if c["energy"] < CODE_E_THRESHOLD][:top_k]


def validate_syntax(lnn: LatticeNN, generated_token: str, context: list) -> bool:
    """
    Check if generated_token closes all open syntax springs.

    Returns True if all τ=0 springs involving generated_token
    and tokens in context have zero residual tension.
    """
    for ctx_token in context:
        sp = lnn.springs.get(lnn._key(ctx_token, generated_token))
        if sp and sp.tau == 0 and sp.stiffness > 0:
            ctx_node = lnn.nodes.get(ctx_token)
            gen_node = lnn.nodes.get(generated_token)
            if ctx_node and gen_node:
                if ctx_node.activation < 50 and gen_node.activation < 50:
                    return False
    return True


def generate_code_sequence(lnn: LatticeNN, prompt: str, max_tokens: int = 20,
                            top_k: int = 5) -> str:
    """
    Generate code autoregressively with syntax validation.

    Stop conditions:
    - max_tokens reached
    - no valid candidate (epistemic honesty)
    - validate_syntax fails for all candidates
    """
    context = prompt.lower().split()
    generated = []

    for _ in range(max_tokens):
        candidates = generate_code(lnn, context, top_k=top_k)

        if not candidates:
            break

        best = candidates[0]["word"]

        if not validate_syntax(lnn, best, context):
            break

        generated.append(best)
        context.append(best)

    return " ".join(generated)


def test_generation(lnn: LatticeNN, prompt: str, expected_tokens: list = None) -> dict:
    """
    Test code generation for a given prompt.

    Returns:
    {
        "prompt": str,
        "candidates": list,
        "sequence": str,
        "expected_match": bool,
    }
    """
    context = prompt.split()
    candidates = generate_code(lnn, context, top_k=5)
    sequence = generate_code_sequence(lnn, prompt, max_tokens=10)

    result = {
        "prompt": prompt,
        "candidates": candidates,
        "sequence": sequence,
    }

    if expected_tokens:
        result["expected_match"] = (sequence.strip() == " ".join(expected_tokens))

    return result
