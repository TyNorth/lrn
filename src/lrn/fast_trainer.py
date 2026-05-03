"""
Fast word-level training for higher grade levels.
Skips character n-gram processing (already done in Pre-K/K).
Directly forms word-word springs based on co-occurrence.
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, propagate, assign_roles
from lrn.trainer import optimal_rem, prune_springs


def train_fast(lnn: LatticeNN, sentences: list, reps: int = 10, verbose: bool = True):
    """Fast training: word-level springs only, no character n-grams."""
    wake_buffer = []
    total = len(sentences) * reps

    for rep in range(reps):
        for idx, sentence in enumerate(sentences):
            tokens = sentence.lower().strip().split()
            if not tokens:
                continue

            for tok in tokens:
                word_node = f"word:{tok}"
                if word_node not in lnn.nodes:
                    lnn.add_node(word_node)

            roles = assign_roles(tokens)
            for tok, role in zip(tokens, roles):
                wn = f"word:{tok}"
                if wn in lnn.nodes:
                    if role not in lnn.nodes[wn].role_counts:
                        lnn.nodes[wn].role_counts[role] = 0
                    lnn.nodes[wn].role_counts[role] += 1

            for i in range(len(tokens)):
                for j in range(i + 1, min(i + 6, len(tokens))):
                    a = f"word:{tokens[i]}"
                    b = f"word:{tokens[j]}"
                    distance = j - i
                    k = max(1, 10 // distance)
                    if roles[i] == roles[j]:
                        k += 3
                    lnn.add_or_update_spring(a, b, stiffness=k, tau=4, mode="add")

            for size in [3, 4, 5]:
                for i in range(len(tokens) - size + 1):
                    gram = tuple(tokens[i:i+size])
                    lnn.trigrams[gram] = lnn.trigrams.get(gram, 0) + 1

            wake_buffer.append(sentence)
            if len(wake_buffer) > 20:
                wake_buffer = wake_buffer[-20:]

        optimal_rem(lnn, wake_buffer)
        propagate(lnn, n_steps=3)

    optimal_rem(lnn, wake_buffer)
    pruned = prune_springs(lnn)
    propagate(lnn, n_steps=5)

    if verbose:
        if pruned:
            print(f"  Pruned: {pruned} noise springs")
