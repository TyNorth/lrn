"""
Training functions for LatticeNN
- add_sentence: Hebbian structural imprinting
- add_negative_sentence: Repulsive springs
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, Node, assign_roles


def add_sentence(lnn: LatticeNN, sentence: str, reality: float = 1.0) -> None:
    """
    Ingest one sentence into the lattice.
    
    Steps:
    1. Tokenize
    2. Create nodes for each token
    3. Assign roles (STARTER, ACTOR, LINKER, SETTLER, CLOSER)
    4. Form sequential springs between adjacent tokens (τ=4 contextual)
    5. Form skip springs for non-adjacent tokens with role match
    6. Update trigram table
    """
    tokens = sentence.lower().strip().split()
    n = len(tokens)

    if n == 0:
        return

    for tok in tokens:
        if tok not in lnn.nodes:
            lnn.add_node(tok)

    roles = assign_roles(tokens)
    for tok, role in zip(tokens, roles):
        if role not in lnn.nodes[tok].role_counts:
            lnn.nodes[tok].role_counts[role] = 0
        lnn.nodes[tok].role_counts[role] += 1

    WINDOW_SIZE = 5

    for i in range(n):
        for j in range(i + 1, min(i + WINDOW_SIZE + 1, n)):
            a, b = tokens[i], tokens[j]
            distance = j - i

            k = max(1, 10 // distance)

            if roles[i] == roles[j]:
                k += 3

            k = int(k * reality)

            lnn.add_or_update_spring(a, b, stiffness=k, tau=4)

    for size in [3, 4, 5]:
        for i in range(n - size + 1):
            gram = tuple(tokens[i:i+size])
            lnn.trigrams[gram] = lnn.trigrams.get(gram, 0) + 1


def add_negative_sentence(lnn: LatticeNN, sentence: str) -> None:
    """
    Parse 'X don't Y' → add k=-5 spring between X and Y.
    """
    tokens = sentence.lower().split()
    if "don't" in tokens:
        idx = tokens.index("don't")
        subject = tokens[idx - 1] if idx > 0 else None
        predicate = tokens[idx + 1] if idx + 1 < len(tokens) else None
        if subject and predicate:
            lnn.add_or_update_spring(subject, predicate, stiffness=-5, tau=4,
                                      mode="neg_override")


def add_identity_anchor(lnn: LatticeNN) -> None:
    """
    Add identity:self node - the self-reference anchor.
    Always pinned at activation=100.
    """
    lnn.add_node("identity:self")
    lnn.nodes["identity:self"].activation = 100
    lnn.nodes["identity:self"].pinned = True


def train_corpus(lnn: LatticeNN, sentences: list, negatives: list = None,
                 reality: float = 1.0) -> dict:
    """
    Train the lattice on a corpus of sentences.
    
    Returns statistics about the training.
    """
    stats = {
        'sentences_processed': 0,
        'nodes_created': 0,
        'springs_created': 0,
        'trigrams_added': 0
    }

    initial_nodes = len(lnn.nodes)
    initial_springs = len(lnn.springs)

    for sentence in sentences:
        add_sentence(lnn, sentence, reality=reality)
        stats['sentences_processed'] += 1

    if negatives:
        for neg in negatives:
            add_negative_sentence(lnn, neg)

    final_nodes = len(lnn.nodes)
    final_springs = len(lnn.springs)

    stats['nodes_created'] = final_nodes - initial_nodes
    stats['springs_created'] = final_springs - initial_springs
    stats['trigrams_added'] = len(lnn.trigrams)

    return stats