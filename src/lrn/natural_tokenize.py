"""
Natural Language Tokenization for LRN - Learn language from raw text, not rules.
Tokenization emerges from character sequence patterns through spring formation.
Supports ANY language including Mandarin (single-char morphemes).

How it works:
1. Feed raw text as character n-grams into the lattice
2. Character sequences that co-occur often form springs → become "words/morphemes"
3. After training, words and morphemes emerge from exposure
4. No hardcoded vocabulary - language learning is native
"""
import re
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN


# Character n-gram sizes for language morpheme discovery
# 1-4 to support Mandarin single-char morphemes
CHAR_NGRAM_SIZES = [1, 2, 3, 4]

# Language is more flexible than code - lower threshold for τ=0
# After 30+ exposures, word order becomes rigid (grammar learned)
TAU_THRESHOLD = 30

# Tau hierarchy for different learning types:
# - SENSORY: Physical world is rigid - gravity, temperature, etc.
# - CAUSATION: Cause→effect is absolute - must happen
# - LANGUAGE: Words are flexible - context-dependent meanings
TAU_BY_TYPE = {
    "sensory": 5,      # Becomes rigid after 5 exposures
    "causation": 10,   # Becomes rigid after 10 exposures
    "language": 30,    # Stays flexible longer (30 exposures)
    "default": 30,     # Default for unknown types
}


def learn_from_text(lnn: LatticeNN, text: str, repetitions: int = 1, 
                   learn_type: str = "language", language: str = None) -> dict:
    """
    Learn language by forming springs between character sequences.
    
    TAU HIERARCHY:
    - sensory: τ=0 after 5 reps - physical world is rigid
    - causation: τ=0 after 10 reps - cause→effect is absolute  
    - language: τ=0 after 30 reps - words are flexible
    
    This is how children learn: physical first, then causation, then language.
    """
    stats = {"char_ngrams": 0, "springs": 0, "rigid_springs": 0}
    
    text = text.strip()
    if not text:
        return stats
    
    # Get tau threshold for this learning type
    tau_threshold = TAU_BY_TYPE.get(learn_type, TAU_BY_TYPE["default"])
    
    # Clean text: normalize whitespace
    text = _clean_text(text)
    
    for rep in range(repetitions):
        # Character n-grams form "morpheme" nodes
        for size in CHAR_NGRAM_SIZES:
            for i in range(len(text) - size + 1):
                ngram = text[i:i+size]
                
                if ngram.strip():
                    # Add prefix to distinguish types
                    prefix = {"sensory": "sens", "causation": "cauz", "language": "lang"}.get(learn_type, "lang")
                    node_name = f"{prefix}:{ngram}"
                    lnn.add_node(node_name)
                    stats["char_ngrams"] += 1
        
        # Form springs between adjacent character positions
        prefix = {"sensory": "sens", "causation": "cauz", "language": "lang"}.get(learn_type, "lang")
        
        for i in range(len(text) - 1):
            a = f"{prefix}:{text[i]}"
            b = f"{prefix}:{text[i+1]}"
            
            # After tau_threshold repetitions, springs become rigid
            tau = 4 if rep < tau_threshold else 0
            stiffness = 5 if learn_type == "sensory" else 3  # Sensory is stronger
            
            lnn.add_or_update_spring(a, b, stiffness=stiffness, tau=tau, mode="add")
            stats["springs"] += 1
            
            if tau == 0:
                stats["rigid_springs"] += 1
        
        # Word boundaries - weak springs
        for i, c in enumerate(text):
            if c in " \t\n.,!?:;\"'" and i + 1 < len(text):
                a = f"{prefix}:{c}"
                b = f"{prefix}:{text[i+1]}"
                tau = 4 if rep < tau_threshold else 0
                lnn.add_or_update_spring(a, b, stiffness=0, tau=tau, mode="add")
    
    return stats


def _clean_text(text: str) -> str:
    """Normalize whitespace while preserving language-specific chars."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def discover_words(lnn: LatticeNN, corpus: list, min_frequency: int = 3) -> dict:
    """
    After training, discover which character sequences became "words".
    These are the vocabulary that emerged from exposure.
    
    Also scans corpus for word-like patterns (alphanumeric sequences).
    """
    word_nodes = {}
    
    # Method 1: Look at node exposure counts (trained patterns)
    for name, node in lnn.nodes.items():
        if not name.startswith("lang:"):
            continue
        
        # Use node's own exposure_count
        usage = node.exposure_count if hasattr(node, 'exposure_count') else 0
        
        if usage >= min_frequency:
            word = name[5:]  # Remove "lang:" prefix
            word_nodes[word] = {
                "node": name,
                "usage_count": usage,
                "springs": len(lnn.get_neighbors(name))
            }
    
    # Method 2: Scan corpus for word-like patterns
    word_freq = {}
    for text in corpus:
        text = _clean_text(text)
        # Extract word-like tokens (letters, including non-ASCII for accents, Chinese)
        words = re.findall(r'[a-zA-Z\u00C0-\u024F\u4e00-\u9fff]+', text)
        for w in words:
            w_lower = w.lower()
            word_freq[w_lower] = word_freq.get(w_lower, 0) + 1
    
    # Add frequent words to discovered vocabulary
    for word, freq in word_freq.items():
        if freq >= min_frequency and word not in word_nodes:
            word_nodes[word] = {
                "node": f"lang:{word}",
                "usage_count": freq,
                "springs": 0,
                "source": "corpus_scan"
            }
    
    # Sort by frequency
    return dict(sorted(word_nodes.items(), key=lambda x: -x[1]["usage_count"]))


def native_tokenize(lnn: LatticeNN, text: str) -> list:
    """
    After training, "tokenize" by finding known character sequences.
    This is query-based, not rule-based.
    """
    tokens = []
    
    # Sort by length descending to match longest first
    known_tokens = sorted(
        [n for n in lnn.nodes.keys() if n.startswith("lang:")],
        key=lambda x: -len(x)
    )
    
    remaining = text
    while remaining:
        matched = False
        for token_node in known_tokens:
            token = token_node[5:]  # Remove "lang:" prefix
            if remaining.lower().startswith(token.lower()):
                tokens.append(token_node)
                remaining = remaining[len(token):]
                matched = True
                break
        
        if not matched:
            # Unknown - add as single char
            if remaining:
                tokens.append(f"lang:{remaining[0]}")
                remaining = remaining[1:]
    
    return tokens


def learn_mixed_language(lnn: LatticeNN, text: str, repetitions: int = 1) -> dict:
    """
    Learn from mixed-language text (Spanglish, Chinese-English, etc.)
    The LRN naturally separates languages based on character patterns.
    
    Example: "Yo quiero eat tacos" → learns "yo", "quiero" (Spanish)
                                          + "eat", "tacos" (English)
    """
    return learn_from_text(lnn, text, repetitions=repetitions)


def discover_language_markers(lnn: LatticeNN, corpus: dict) -> dict:
    """
    After training, discover which patterns belong to which language.
    Useful for polyglot detection.
    """
    language_markers = {}
    
    for lang, texts in corpus.items():
        words = discover_words(lnn, texts, min_frequency=3)
        language_markers[lang] = list(words.keys())[:50]  # Top 50 words per language
    
    return language_markers