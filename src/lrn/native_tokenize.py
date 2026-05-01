"""
Native Tokenization for LRN - Learn tokens from raw code, not rules.
Tokenization emerges from character sequence patterns through spring formation.

How it works:
1. Feed raw code as character n-grams into the lattice
2. Character sequences that co-occur often form springs → become "tokens"
3. After training, querying "def" finds the node representing that character sequence
4. No hardcoded keywords - grammar emerges from examples
"""
import re
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN


# Character n-gram sizes for code token discovery
CHAR_NGRAM_SIZES = [2, 3, 4, 5, 6]

# SYNTAX PENALTY: Higher repetitions needed before springs become τ=0
# Code syntax is rigid - no partial compliance. Compile or don't.
TAU_THRESHOLD = 50  # Higher = stricter syntax learning


def learn_from_code(lnn: LatticeNN, code: str, repetitions: int = 1) -> dict:
    """
    Learn code by forming springs between character sequences.
    No explicit tokenization - patterns emerge from repetition.
    
    SYNTAX IS RIGID: Higher repetitions required before springs become τ=0.
    Code either compiles or doesn't - no partial syntax.
    
    This is how children learn language: not by rules, but by hearing
    words repeated until the brain forms "word" patterns.
    """
    stats = {"char_ngrams": 0, "springs": 0}
    
    code = code.strip()
    if not code:
        return stats
    
    # Clean code: remove comments, normalize whitespace
    code = _clean_code(code)
    
    for rep in range(repetitions):
        # Character n-grams form "morpheme" nodes
        for size in CHAR_NGRAM_SIZES:
            for i in range(len(code) - size + 1):
                ngram = code[i:i+size]
                
                # Skip whitespace-only n-grams
                if ngram.strip():
                    node_name = f"char:{ngram}"
                    lnn.add_node(node_name)
                    stats["char_ngrams"] += 1
        
        # Form springs between adjacent character positions
        # Character position i connects to position i+1
        # This creates "flow" through the code
        for i in range(len(code) - 1):
            a = f"char:{code[i]}"
            b = f"char:{code[i+1]}"
            # Higher tau threshold = stricter syntax (more repetitions needed)
            tau = 4 if rep < TAU_THRESHOLD else 0
            lnn.add_or_update_spring(a, b, stiffness=1, tau=tau, mode="add")
            stats["springs"] += 1
        
        # Word boundaries (spaces/punctuation) act as separators
        # But adjacent chars across boundaries still form weak springs
        for i, c in enumerate(code):
            if c in " \t\n{}():;," and i + 1 < len(code):
                a = f"char:{c}"
                b = f"char:{code[i+1]}"
                tau = 4 if rep < TAU_THRESHOLD else 0
                lnn.add_or_update_spring(a, b, stiffness=0, tau=tau, mode="add")
    
    return stats


def _clean_code(code: str) -> str:
    """Remove comments and normalize whitespace."""
    # Remove single-line comments
    if "//" in code:
        code = code[:code.index("//")]
    if "#" in code:
        code = code[:code.index("#")]
    
    # Normalize whitespace
    code = re.sub(r'\s+', ' ', code)
    return code.strip()


def discover_tokens(lnn: LatticeNN, code_samples: list, min_frequency: int = 10) -> dict:
    """
    After training, discover which character sequences became "tokens".
    These are the patterns that accumulated high exposure counts.
    
    Also scan code for word patterns that appear frequently.
    """
    token_nodes = {}
    
    # Method 1: Look at node exposure counts (trained patterns)
    for name, node in lnn.nodes.items():
        if not name.startswith("char:"):
            continue
        
        # Use node's own exposure_count (from training)
        usage = node.exposure_count if hasattr(node, 'exposure_count') else 0
        
        if usage >= min_frequency:
            token = name[5:]  # Remove "char:" prefix
            token_nodes[token] = {
                "node": name,
                "usage_count": usage,
                "springs": len(lnn.get_neighbors(name))
            }
    
    # Method 2: Scan code samples for word-like patterns
    word_freq = {}
    for code in code_samples:
        code = _clean_code(code)
        # Extract word-like tokens (alphanumeric + underscore, length 2-15)
        import re
        words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]{1,14}', code)
        for w in words:
            word_freq[w] = word_freq.get(w, 0) + 1
    
    # Add frequent words to discovered tokens
    for word, freq in word_freq.items():
        if freq >= 3 and word not in token_nodes:
            token_nodes[word] = {
                "node": f"char:{word}",
                "usage_count": freq,
                "springs": 0,
                "source": "corpus_scan"
            }
    
    # Sort by frequency
    return dict(sorted(token_nodes.items(), key=lambda x: -x[1]["usage_count"]))


def learn_keywords_from_code(lnn: LatticeNN, code_corpus: list, min_occurrences: int = 20) -> dict:
    """
    After sufficient training, keywords emerge as high-frequency tokens.
    These are the "words" the LRN has learned from code.
    """
    # Train on all code samples
    for code in code_corpus:
        learn_from_code(lnn, code, repetitions=1)
    
    # Discover tokens after training
    tokens = discover_tokens(lnn, code_corpus, min_frequency=min_occurrences)
    
    # Filter to likely keywords (short, high frequency)
    keywords = {
        token: info for token, info in tokens.items()
        if len(token) <= 10 and info["usage_count"] >= min_occurrences
    }
    
    return keywords


def native_tokenize(lnn: LatticeNN, code: str) -> list:
    """
    After training, "tokenize" by finding known character sequences.
    This is query-based, not rule-based.
    """
    # Try to match known tokens (learned from training)
    tokens = []
    
    # Sort by length descending to match longest first
    known_tokens = sorted(
        [n for n in lnn.nodes.keys() if n.startswith("char:")],
        key=lambda x: -len(x)
    )
    
    remaining = code
    while remaining:
        matched = False
        for token_node in known_tokens:
            token = token_node[5:]  # Remove "char:" prefix
            if remaining.startswith(token):
                tokens.append(token_node)
                remaining = remaining[len(token):]
                matched = True
                break
        
        if not matched:
            # Unknown character - add as single char
            if remaining:
                tokens.append(f"char:{remaining[0]}")
                remaining = remaining[1:]
    
    return tokens


# Example: Learn code from raw text
def train_native_code(lnn, code_samples, repetitions=10):
    """Train LRN to learn code tokens natively."""
    for code in code_samples:
        learn_from_code(lnn, code, repetitions=repetitions)
    
    # After training, keywords emerge naturally
    keywords = discover_tokens(lnn, code_samples, min_frequency=5)
    return keywords