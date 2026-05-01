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


# --- Multi-Language Grammar Patterns ---
# Each language has distinct syntax - all learned through springs

PYTHON_PATTERNS = [
    # Function definitions
    "code:kw:def code:var:add code:sym:lparen code:var:a code:sym:comma code:var:b code:sym:rparen code:sym:colon code:kw:return code:var:a code:op:plus code:var:b",
    "code:kw:def code:var:foo code:sym:lparen code:sym:rparen code:sym:colon code:kw:return code:lit:int",
    "code:kw:def code:var:hello code:sym:lparen code:var:name code:sym:rparen code:sym:colon code:kw:return code:lit:str",
    # Conditionals
    "code:kw:if code:var:condition code:sym:colon",
    "code:kw:if code:var:x code:op:gt code:lit:int code:sym:colon code:kw:return code:lit:int",
    # Loops
    "code:kw:for code:var:i code:kw:in code:var:range code:sym:lparen code:var:n code:sym:rparen code:sym:colon",
    # Assignments
    "code:var:x code:sym:assign code:lit:int",
]

RUST_PATTERNS = [
    # Function definitions with types and return type
    "code:kw:fn code:var:add code:sym:lparen code:var:a code:sym:: code:type:i32 code:sym:, code:var:b code:sym:: code:type:i32 code:sym:rparen code:sym:-> code:type:i32 code:sym:lbrace code:kw:return code:var:a code:op:+ code:var:b code:sym:rbrace",
    "code:kw:fn code:var:main code:sym:lparen code:sym:rparen code:sym:-> code:type:i32 code:sym:lbrace code:kw:return code:literal:0 code:sym:rbrace",
    # Conditionals
    "code:kw:if code:var:x code:op:gt code:literal:0 code:sym:lbrace code:kw:return code:literal:1 code:sym:rbrace",
    "code:kw:if code:var:x code:op:gt code:literal:0 code:sym:lbrace code:kw:return code:literal:1 code:sym:rbrace code:kw:else code:sym:lbrace code:kw:return code:literal:0 code:sym:rbrace",
    # Let binding
    "code:kw:let code:kw:mut code:var:x code:sym::= code:literal:5",
]

JAVASCRIPT_PATTERNS = [
    # Function definitions
    "code:kw:function code:var:add code:sym:lparen code:var:a code:sym:, code:var:b code:sym:rparen code:sym:lbrace code:kw:return code:var:a code:op:+ code:var:b code:sym:rbrace",
    # Arrow function
    "code:kw:const code:var:add code:sym::= code:sym:lparen code:var:a code:sym:, code:var:b code:sym:rparen code:op:=> code:sym:lbrace code:kw:return code:var:a code:op:+ code:var:b code:sym:rbrace",
    # Conditionals
    "code:kw:if code:sym:lparen code:var:x code:op:gt code:literal:0 code:sym:rparen code:sym:lbrace code:kw:return code:literal:1 code:sym:rbrace",
    # Variable declarations
    "code:kw:let code:var:x code:sym::= code:literal:5",
]

GO_PATTERNS = [
    # Function definitions - no type in param after name
    "code:kw:func code:var:add code:sym:lparen code:var:a code:var:int code:sym:, code:var:b code:var:int code:sym:rparen code:var:int code:sym:lbrace code:kw:return code:var:a code:op:+ code:var:b code:sym:rbrace",
    "code:kw:func code:var:main code:sym:lparen code:sym:rparen code:sym:lbrace code:kw:return code:sym:rbrace",
    # Conditionals
    "code:kw:if code:var:x code:op:gt code:literal:0 code:sym:lbrace code:kw:return code:literal:1 code:sym:rbrace",
    # Short variable declaration
    "code:var:x code:sym::= code:literal:5",
]

RUBY_PATTERNS = [
    # Function definitions - no braces, inline params, end keyword
    "code:kw:def code:var:add code:var:a code:sym:, code:var:b code:kw:return code:var:a code:op:+ code:var:b code:kw:end",
    "code:kw:def code:var:main code:kw:return code:sym:nil code:kw:end",
    # Conditionals
    "code:kw:if code:var:x code:op:gt code:literal:0 code:kw:return code:literal:1 code:kw:end",
    "code:kw:if code:var:x code:op:gt code:literal:0 code:kw:then code:kw:return code:literal:1 code:kw:else code:kw:return code:literal:0 code:kw:end",
]

COBOL_PATTERNS = [
    # COBOL - column-based, verbose, uppercase keywords
    "code:kw:IDENTIFICATION code:kw:DIVISION code:kw:PROGRAM-ID code:sym:. code:var:PROGNAME",
    "code:kw:PROCEDURE code:kw:DIVISION code:sym:.",
    "code:kw:IF code:var:X code:op:gt code:literal:ZERO code:kw:THEN code:kw:DISPLAY code:literal:HELLO code:kw:END-IF",
    "code:kw:MOVE code:var:VALUE code:kw:TO code:var:TARGET",
    "code:kw:DISPLAY code:literal:HELLO",
    "code:kw:PERFORM code:var:ROUTINE code:kw:UNTIL code:var:DONE",
]

ZIG_PATTERNS = [
    # Function definitions - explicit error handling
    "code:kw:fn code:var:add code:sym:lparen code:var:a code:sym:: code:type:i32 code:sym:, code:var:b code:sym:: code:type:i32 code:sym:rparen code:sym:! code:type:error code:sym:-> code:type:i32 code:sym:lbrace code:kw:return code:var:a code:op:+ code:var:b code:sym:rbrace",
    # Main function
    "code:kw:pub code:kw:fn code:var:main code:sym:lparen code:sym:rparen code:type:void code:sym:lbrace code:kw:return code:sym:rbrace",
    # Conditionals with parentheses required
    "code:kw:if code:sym:lparen code:var:x code:op:> code:literal:0 code:sym:rparen code:sym:lbrace code:kw:return code:var:x code:sym:rbrace code:kw:else code:sym:lbrace code:kw:return code:literal:0 code:sym:rbrace",
    # Variable declarations
    "code:kw:const code:var:x code:sym::= code:literal:5",
    "code:kw:var code:var:y code:sym::= code:literal:10",
]


# Logic patterns per language (boolean operators differ)
PYTHON_LOGIC = [
    "code:kw:if code:var:x code:op:gt code:lit:int code:kw:and code:var:y code:op:lt code:lit:int code:sym:colon",
    "code:kw:if code:var:x code:op:eq code:lit:int code:kw:or code:var:y code:op:neq code:lit:int code:sym:colon",
]

RUST_LOGIC = [
    "code:kw:if code:var:x code:op:> code:literal:0 code:kw:&& code:var:y code:op:< code:literal:10 code:sym:lbrace code:sym:rbrace",
    "code:kw:match code:var:x code:sym:lbrace code:kw:Some code:sym:lparen code:var:v code:sym:rparen code:op:=> code:var:v code:sym:, code:kw:None code:op:=> code:literal:0 code:sym:rbrace",
]

JAVASCRIPT_LOGIC = [
    "code:kw:if code:sym:lparen code:var:x code:op:> code:literal:0 code:op:&& code:var:y code:op:< code:literal:10 code:sym:rparen code:sym:lbrace code:sym:rbrace",
    "code:kw:if code:var:x code:op:=== code:literal:5 code:sym:lbrace code:sym:rbrace",
]

GO_LOGIC = [
    "code:kw:if code:var:x code:op:> code:literal:0 code:op:&& code:var:y code:op:< code:literal:10 code:sym:lbrace code:sym:rbrace",
    "code:kw:switch code:var:x code:sym:lbrace code:kw:case code:literal:1 code:sym:: code:kw:return code:literal:1 code:sym:, code:kw:default code:sym:: code:kw:return code:literal:0 code:sym:rbrace",
]

RUBY_LOGIC = [
    "code:kw:if code:var:x code:op:> code:literal:0 code:kw:and code:var:y code:op:< code:literal:10 code:kw:then code:kw:end",
    "code:kw:if code:var:x code:op:== code:literal:5 code:kw:unless code:var:y code:op:!= code:literal:0 code:kw:end",
]

COBOL_LOGIC = [
    "code:kw:IF code:var:X code:op:GT code:literal:ZERO code:kw:AND code:var:Y code:op:LESS code:literal:TEN code:kw:THEN code:kw:DISPLAY code:literal:YES code:kw:END-IF",
    "code:kw:IF code:var:X code:op:EQ code:literal:FIVE code:kw:OR code:var:Y code:op:NE code:literal:ZERO code:kw:THEN code:kw:SET code:var:RESULT code:kw:TO code:literal:TRUE code:kw:END-IF",
]

ZIG_LOGIC = [
    "code:kw:if code:sym:lparen code:var:x code:op:> code:literal:0 code:op:and code:var:y code:op:< code:literal:10 code:sym:rparen code:sym:lbrace code:kw:return code:var:x code:sym:rbrace",
    "code:kw:switch code:sym:lparen code:var:x code:sym:rparen code:sym:lbrace code:kw:case code:literal:1 code:sym:=> code:kw:return code:literal:1 code:sym:, code:kw:else code:sym:=> code:kw:return code:literal:0 code:sym:rbrace",
]


# All patterns combined
ALL_LANGUAGE_PATTERNS = {
    "python": PYTHON_PATTERNS,
    "rust": RUST_PATTERNS,
    "javascript": JAVASCRIPT_PATTERNS,
    "go": GO_PATTERNS,
    "ruby": RUBY_PATTERNS,
    "cobol": COBOL_PATTERNS,
    "zig": ZIG_PATTERNS,
}

ALL_LOGIC_PATTERNS = {
    "python": PYTHON_LOGIC,
    "rust": RUST_LOGIC,
    "javascript": JAVASCRIPT_LOGIC,
    "go": GO_LOGIC,
    "ruby": RUBY_LOGIC,
    "cobol": COBOL_LOGIC,
    "zig": ZIG_LOGIC,
}


def train_language(lnn, language: str, repetitions: int = 60) -> dict:
    """Train grammar for a specific language."""
    stats = {"patterns": 0, "tokens": 0, "springs": 0, "ngrams": 0, "promoted": 0}
    
    patterns = ALL_LANGUAGE_PATTERNS.get(language, [])
    logic_patterns = ALL_LOGIC_PATTERNS.get(language, [])
    
    for _ in range(repetitions):
        for pattern in patterns + logic_patterns:
            result = add_code_file(lnn, pattern)
            stats["patterns"] += 1
            stats["tokens"] += result["tokens"]
            stats["springs"] += result["springs"]
            stats["ngrams"] += result["ngrams"]
    
    stats["promoted"] = promote_code_springs(lnn, threshold=50)
    return stats


def train_all_languages(lnn, repetitions_per_lang: int = 40) -> dict:
    """Train grammar for all 7 languages."""
    stats = {"languages": 0, "patterns": 0, "tokens": 0, "springs": 0, "ngrams": 0, "promoted": 0}
    
    for lang in ALL_LANGUAGE_PATTERNS.keys():
        lang_stats = train_language(lnn, lang, repetitions=repetitions_per_lang)
        stats["languages"] += 1
        stats["patterns"] += lang_stats["patterns"]
        stats["tokens"] += lang_stats["tokens"]
        stats["springs"] += lang_stats["springs"]
        stats["ngrams"] += lang_stats["ngrams"]
        stats["promoted"] += lang_stats["promoted"]
    
    return stats


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
