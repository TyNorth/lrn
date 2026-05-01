"""
Code training for LRN.
Programming syntax IS grammar - learned through springs like natural language.
No language specification needed - tau hierarchy handles context naturally.
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN
from lrn.training import add_sentence
from lrn.code_tokenizer import tokenize_code_line, assign_code_roles

# Code-specific constants
CODE_SYNTAX_K = 100
CODE_TYPE_K = 50
CODE_SCOPE_K = 20
CODE_SEMANTIC_K = 10
CODE_PROMOTION_SAMPLES = 50
SCOPE_STEP = 4
SCOPE_MAX = 32
CODE_NGRAM_SIZES = [3, 4, 5, 7, 10]


def is_pre_tokenized(text: str) -> bool:
    """Check if text is already pre-tokenized node names."""
    return "code:" in text and all(t.startswith("code:") or t == "" for t in text.split())


def add_code_file(lnn: LatticeNN, source_code: str, language: str = "python") -> dict:
    """
    Ingest code into the lattice.
    
    Handles two cases:
    1. Real Python source -> tokenize -> springs
    2. Pre-tokenized node names -> split -> springs
    
    Grammar is learned naturally through spring formation and promotion.
    """
    stats = {"tokens": 0, "springs": 0, "ngrams": 0}

    if is_pre_tokenized(source_code):
        # Pre-tokenized: split into node names
        tokens = source_code.split()
    else:
        # Real code: tokenize
        lines = source_code.split("\n")
        tokens = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            line_tokens = tokenize_code_line(stripped, language)
            tokens.extend(line_tokens)

    if not tokens:
        return stats

    roles = assign_code_roles(tokens)

    # Create nodes and update role counts
    for tok, role in zip(tokens, roles):
        if tok not in lnn.nodes:
            lnn.add_node(tok)
        if role not in lnn.nodes[tok].role_counts:
            lnn.nodes[tok].role_counts[role] = 0
        lnn.nodes[tok].role_counts[role] += 1

    # Sequential springs - grammar patterns form here
    # After repetition, these promote to τ=0 (rigid syntax)
    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        lnn.add_or_update_spring(a, b, stiffness=20, tau=4, mode="add")
        stats["springs"] += 1

    # Skip springs for non-adjacent tokens
    for i in range(len(tokens)):
        for j in range(i + 2, min(i + 6, len(tokens))):
            a, b = tokens[i], tokens[j]
            distance = j - i
            k = max(1, 20 // distance)
            lnn.add_or_update_spring(a, b, stiffness=k, tau=4, mode="add")
            stats["springs"] += 1

    # N-grams for grammar patterns
    for size in CODE_NGRAM_SIZES:
        for i in range(len(tokens) - size + 1):
            gram = tuple(tokens[i:i + size])
            lnn.trigrams[gram] = lnn.trigrams.get(gram, 0) + 1
            stats["ngrams"] += 1

    stats["tokens"] = len(tokens)
    return stats


def promote_code_springs(lnn: LatticeNN, threshold: int = CODE_PROMOTION_SAMPLES) -> int:
    """Grammar pattern springs get promoted to τ=0."""
    promoted = 0
    for key, sp in lnn.springs.items():
        if sp.exposure_count >= threshold and sp.tau > 0:
            sp.tau = 0
            sp.stiffness = max(sp.stiffness, CODE_SYNTAX_K)
            promoted += 1
    return promoted


def initialize_scope_axis(lnn: LatticeNN) -> None:
    """Create scope geometry for variable binding."""
    for depth in range(SCOPE_MAX):
        node = lnn.get_or_create(f"code:block:{depth}")
        node.x = depth * SCOPE_STEP
        node.modality = "B"

    for depth in range(SCOPE_MAX - 1):
        lnn.add_or_update_spring(
            f"code:block:{depth}",
            f"code:block:{depth + 1}",
            stiffness=30,
            tau=1
        )


# --- Code Curriculum: Grammar patterns, not spec ---
# Grammar learned through repetition, not language specification

GRAMMAR_PATTERNS = [
    # Function definitions
    "code:kw:def code:var:add code:sym:lparen code:var:a code:sym:comma code:var:b code:sym:rparen code:sym:colon code:kw:return code:var:a code:op:plus code:var:b",
    "code:kw:def code:var:foo code:sym:lparen code:sym:rparen code:sym:colon code:kw:return code:lit:int",
    "code:kw:def code:var:hello code:sym:lparen code:var:name code:sym:rparen code:sym:colon code:kw:return code:lit:str",
    "code:kw:def code:var:identity code:sym:lparen code:var:x code:sym:rparen code:sym:colon code:kw:return code:var:x",
    "code:kw:def code:var:square code:sym:lparen code:var:x code:sym:rparen code:sym:colon code:kw:return code:var:x code:op:star code:var:x",
    
    # Conditionals
    "code:kw:if code:var:condition code:sym:colon",
    "code:kw:if code:var:x code:op:gt code:lit:int code:sym:colon code:kw:return code:lit:int",
    "code:kw:if code:var:n code:sym:lte code:lit:int code:sym:colon code:kw:return code:lit:int code:kw:else code:kw:return code:var:x code:op:star code:var:x",
    
    # Loops
    "code:kw:for code:var:i code:kw:in code:var:range code:sym:lparen code:var:n code:sym:rparen code:sym:colon",
    "code:kw:while code:var:condition code:sym:colon",
    
    # Assignments
    "code:var:x code:sym:assign code:lit:int",
    "code:var:result code:sym:assign code:var:x code:op:plus code:var:y",
    
    # Return statements
    "code:kw:return code:var:x",
    "code:kw:return code:var:a code:op:plus code:var:b",
    "code:kw:return code:lit:int",
    "code:kw:return code:lit:str",
    
    # Class definitions
    "code:kw:class code:var:MyClass code:sym:colon code:kw:def code:sym:lparen code:var:self code:sym:rparen code:sym:colon",
    "code:kw:class code:var:MyClass code:sym:colon code:kw:def code:var:get_value code:sym:lparen code:var:self code:sym:rparen code:sym:colon code:kw:return code:var:self code:sym:dot code:var:value",
    
    # Method calls
    "code:func:print code:sym:lparen code:var:x code:sym:rparen",
    "code:func:len code:sym:lparen code:var:x code:sym:rparen",
]

OLD_SYLLABUS = []  # Deprecated - spec text no longer needed


def train_code_grammar(lnn: LatticeNN, repetitions: int = 80) -> dict:
    """
    Train code grammar through repetition.
    Grammar patterns (springs) form through co-occurrence.
    After repetition, promote to τ=0 for rigid syntax.
    """
    stats = {"patterns": 0, "tokens": 0, "springs": 0, "ngrams": 0}
    
    for _ in range(repetitions):
        for pattern in GRAMMAR_PATTERNS:
            result = add_code_file(lnn, pattern)
            stats["patterns"] += 1
            stats["tokens"] += result["tokens"]
            stats["springs"] += result["springs"]
            stats["ngrams"] += result["ngrams"]
    
    stats["promoted"] = promote_code_springs(lnn, threshold=50)
    return stats
