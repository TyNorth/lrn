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

    # Scale stiffness by exposure count (Hebbian strengthening)
    # On repetition, add to existing stiffness
    base_k = 10
    
    for i in range(n):
        for j in range(i + 1, min(i + WINDOW_SIZE + 1, n)):
            a, b = tokens[i], tokens[j]
            distance = j - i

            k = max(1, base_k // distance)

            if roles[i] == roles[j]:
                k += 3

            k = int(k * reality)

            # Use "add" mode to accumulate stiffness on repetition
            lnn.add_or_update_spring(a, b, stiffness=k, tau=4, mode="add")

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


# --- NATIVE LANGUAGE TRAINING (Fluent Polyglot) ---
# Language learned from raw text - no hardcoded tokenization or rules

from lrn.natural_tokenize import learn_from_text, discover_words, CHAR_NGRAM_SIZES, TAU_THRESHOLD
from lrn.language_corpus import NATIVE_LANGUAGE_SAMPLES, GRADE_THRESHOLDS, get_grade, get_language_samples, get_mixed_samples


def train_language_native(lnn: LatticeNN, language: str, repetitions: int = 60, min_frequency: int = 3) -> dict:
    """
    Learn a language NATIVELY from raw text.
    Words/morphemes emerge from character n-gram patterns through repetition.
    
    After TAU_THRESHOLD (30) repetitions, word order springs become rigid (grammar learned).
    """
    stats = {"samples": 0, "char_ngrams": 0, "springs": 0, "words": 0}
    
    samples = get_language_samples(language)
    if not samples:
        return stats
    
    for _ in range(repetitions):
        for text in samples:
            result = learn_from_text(lnn, text, repetitions=1, language=language)
            stats["samples"] += 1
            stats["char_ngrams"] += result["char_ngrams"]
            stats["springs"] += result["springs"]
    
    # Discover what vocabulary emerged
    words = discover_words(lnn, samples, min_frequency=min_frequency)
    stats["words"] = len(words)
    stats["vocabulary"] = list(words.keys())[:100]  # Top 100 words
    
    return stats


def train_all_languages_native(lnn: LatticeNN, repetitions: int = 60) -> dict:
    """
    Train all 3 languages (English, Spanish, Mandarin) + mixed code-switching.
    Target: 5000 words per language for fluent polyglot.
    """
    stats = {
        "languages": 0,
        "samples": 0,
        "char_ngrams": 0,
        "springs": 0,
        "vocabulary": {},
        "grades": {},
    }
    
    # Train pure languages
    for lang in ["english", "spanish", "mandarin"]:
        lang_stats = train_language_native(lnn, lang, repetitions=repetitions)
        stats["languages"] += 1
        stats["samples"] += lang_stats["samples"]
        stats["char_ngrams"] += lang_stats["char_ngrams"]
        stats["springs"] += lang_stats["springs"]
        
        word_count = lang_stats["words"]
        stats["vocabulary"][lang] = lang_stats.get("vocabulary", [])
        stats["grades"][lang] = get_grade(word_count)
    
    # Train mixed languages (Spanglish, Chinglish, etc.)
    mixed_stats = train_mixed_language_native(lnn, repetitions=repetitions)
    stats["samples"] += mixed_stats["samples"]
    stats["char_ngrams"] += mixed_stats["char_ngrams"]
    stats["springs"] += mixed_stats["springs"]
    stats["mixed_vocabulary"] = mixed_stats.get("vocabulary", {})
    
    return stats


def train_mixed_language_native(lnn: LatticeNN, repetitions: int = 30) -> dict:
    """
    Train code-switching patterns (Spanglish, Chinglish, etc.)
    Polyglot must handle mixed language naturally.
    """
    stats = {"samples": 0, "char_ngrams": 0, "springs": 0, "vocabulary": {}}
    
    mixed_samples = get_mixed_samples()
    
    for mix_type, texts in mixed_samples.items():
        for _ in range(repetitions):
            for text in texts:
                result = learn_from_text(lnn, text, repetitions=1)
                stats["samples"] += 1
                stats["char_ngrams"] += result["char_ngrams"]
                stats["springs"] += result["springs"]
        
        # Discover vocabulary for this mixed type
        words = discover_words(lnn, texts, min_frequency=2)
        stats["vocabulary"][mix_type] = list(words.keys())[:20]
    
    return stats


def train_language_to_grade(lnn: LatticeNN, language: str, target_grade: str) -> dict:
    """
    Train a language until reaching a specific vocabulary grade.
    
    Grades:
    - grade1: 250 words
    - grade2: 500 words
    - grade3: 1000 words
    - grade4: 2000 words
    - grade5: 3500 words
    - grade6: 5000 words (fluent polyglot)
    """
    target_words = GRADE_THRESHOLDS.get(target_grade, 250)
    min_freq = 3
    
    stats = {"samples": 0, "char_ngrams": 0, "springs": 0, "words": 0, "grade": "grade0"}
    
    samples = get_language_samples(language)
    if not samples:
        return stats
    
    # Start with 60 repetitions, increase until target reached
    repetitions = 60
    
    while stats["words"] < target_words and repetitions < 500:
        for _ in range(repetitions):
            for text in samples:
                result = learn_from_text(lnn, text, repetitions=1)
                stats["samples"] += 1
                stats["char_ngrams"] += result["char_ngrams"]
                stats["springs"] += result["springs"]
        
        words = discover_words(lnn, samples, min_frequency=min_freq)
        stats["words"] = len(words)
        stats["grade"] = get_grade(stats["words"])
        
        if stats["words"] < target_words:
            repetitions += 20  # Increase training
    
    return stats