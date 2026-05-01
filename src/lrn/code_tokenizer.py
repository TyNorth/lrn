"""
Multi-language code tokenization for LRN.
Supports Python, Rust, JavaScript, Go, Ruby, COBOL, Zig.
"""
import re


# ============================================================
# KEYWORDS PER LANGUAGE
# ============================================================

LANGUAGE_KEYWORDS = {
    "python": {
        "def", "return", "if", "else", "elif", "for", "while", "class",
        "import", "from", "as", "with", "try", "except", "finally", "raise",
        "pass", "break", "continue", "and", "or", "not", "is", "in",
        "True", "False", "None", "self", "lambda", "yield", "assert",
        "global", "nonlocal", "del", "async", "await", "match", "case",
    },
    "rust": {
        "fn", "pub", "mut", "let", "const", "static", "if", "else", "match",
        "loop", "while", "for", "in", "break", "continue", "return", "struct",
        "enum", "impl", "trait", "type", "where", "use", "mod", "crate",
        "self", "Self", "super", "unsafe", "async", "await", "dyn", "ref",
        "move", "box", "true", "false",
    },
    "javascript": {
        "function", "var", "let", "const", "if", "else", "for", "while", "do",
        "switch", "case", "break", "continue", "return", "throw", "try", "catch",
        "finally", "class", "extends", "new", "this", "super", "import", "export",
        "default", "async", "await", "yield", "typeof", "instanceof", "in", "of",
        "true", "false", "null", "undefined",
    },
    "go": {
        "func", "package", "import", "var", "const", "type", "struct", "interface",
        "if", "else", "for", "range", "switch", "case", "default", "break", "continue",
        "return", "go", "chan", "select", "defer", "map", "make", "new", "true", "false",
    },
    "ruby": {
        "def", "end", "class", "module", "if", "elsif", "unless", "case", "when",
        "while", "until", "for", "do", "return", "break", "next", "redo", "retry",
        "yield", "begin", "rescue", "ensure", "raise", "throw", "catch",
        "true", "false", "nil", "self", "super", "and", "or", "not", "in",
    },
    "cobol": {
        # COBOL keywords (uppercase in source, handled case-insensitive)
        "IDENTIFICATION", "PROGRAM-ID", "ENVIRONMENT", "CONFIGURATION", "DATA", "WORKING-STORAGE",
        "PROCEDURE", "DIVISION", "FILE", "INPUT-OUTPUT", "LINKAGE", "LOCAL-STORAGE",
        "IF", "THEN", "ELSE", "END-IF", "EVALUATE", "WHEN", "END-EVALUATE",
        "MOVE", "TO", "ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "COMPUTE",
        "DISPLAY", "ACCEPT", "PERFORM", "UNTIL", "VARYING", "TIMES", "THRU", "THROUGH",
        "GOBACK", "STOP", "RUN", "OPEN", "CLOSE", "READ", "WRITE", "REWRITE",
        "VALUE", "VALUES", "PIC", "PICTURE", "OCCURS", "INDEXED", "BY", "REDEFINES",
        "INITIALIZE", "SET", "TRUE", "FALSE", "SPACE", "SPACES", "ZERO", "ZEROS",
        "GREATER", "LESS", "EQUAL", "NOT", "AND", "OR",
    },
    "zig": {
        "fn", "pub", "const", "var", "struct", "enum", "union", "type", "if", "else",
        "while", "for", "in", "break", "continue", "return", "switch", "case",
        "defer", "errdefer", "try", "catch", "unreachable", "asm", "export",
        "inline", "noinline", "callconv", "extern", "align", "section",
        "true", "false", "null", "undefined",
    },
}


# ============================================================
# TYPES PER LANGUAGE
# ============================================================

LANGUAGE_TYPES = {
    "python": {"str", "int", "float", "bool", "list", "dict", "set", "tuple", "bytes", "type", "object"},
    "rust": {"i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64", "u128", "usize", "f32", "f64", "bool", "char", "str", "String", "Vec", "Option", "Result", "Box", "Rc", "Arc"},
    "javascript": {"string", "number", "boolean", "object", "symbol", "bigint", "undefined"},
    "go": {"int", "int8", "int16", "int32", "int64", "uint", "uint8", "uint16", "uint32", "uint64", "float32", "float64", "complex64", "complex128", "byte", "rune", "string", "bool", "error"},
    "ruby": {"String", "Integer", "Float", "Array", "Hash", "TrueClass", "FalseClass", "NilClass", "Object", "Class", "Module"},
    "cobol": {"PIC", "PICTURE", "X", "9", "A", "S", "V", "Z", "$", "9V9"},
    "zig": {"i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64", "u128", "usize", "f16", "f32", "f64", "f128", "bool", "void", "noreturn", "type", "error", "anyerror"},
}


# ============================================================
# BUILT-IN FUNCTIONS PER LANGUAGE
# ============================================================

LANGUAGE_BUILTINS = {
    "python": {"print", "len", "range", "str", "int", "float", "bool", "list", "dict", "set", "tuple", "type", "isinstance", "enumerate", "zip", "map", "filter", "sorted", "sum", "min", "max", "abs"},
    "rust": {"println", "print", "eprintln", "eprint", "format", "panic", "assert", "assert_eq", "assert_ne", "vec", "some", "none", "ok", "err"},
    "javascript": {"console", "log", "parseInt", "parseFloat", "isNaN", "isFinite", "JSON", "parse", "stringify", "Array", "Object", "Math", "Date"},
    "go": {"fmt", "Println", "Print", "Printf", "make", "new", "len", "cap", "append", "copy", "delete", "panic", "recover"},
    "ruby": {"puts", "print", "gets", "chomp", "to_s", "to_i", "to_f", "each", "map", "select", "reduce", "split", "join", "strip", "length", "size"},
    "cobol": {"DISPLAY", "ACCEPT", "MOVE", "STRING", "UNSTRING", "INSPECT", "SUBSTITUTE"},
    "zig": {"print", "panic", "sizeof", "@alignOf", "@typeInfo", "@fieldParentPtr", "@bitCast", "@panic"},
}


# ============================================================
# ADDITIONAL OPERATORS PER LANGUAGE
# ============================================================

LANGUAGE_OPS = {
    "python": {"==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "//=", "**=", "->", "//", "**"},
    "rust": {"==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "%=", "&&", "||", "->", "=>", "::", "..", "..=", "|>", "<<", ">>"},
    "javascript": {"==", "!=", "===", "!==", "<=", ">=", "+=", "-=", "*=", "/=", "&&", "||", "??", "?.", "?.", "++", "--", "**"},
    "go": {"==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "&&", "||", ":=", "<<", ">>", "&=", "|=", "^="},
    "ruby": {"==", "!=", "<=", ">=", "=~", "!~", "+=", "-=", "*=", "/=", "&&", "||", "and", "or", "not", "..", "..."},
    "cobol": {"=", "+", "-", "*", "/", "**", "GT", "GE", "LT", "LE", "EQ", "NE", "GREATER", "LESS", "EQUAL", "NOT", "AND", "OR"},
    "zig": {"==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "%=", "&&", "||", "<<", ">>", "->", "=>", "|>", "??", ".*", ".?"},
}


# ============================================================
# SYMBOL MAPPING
# ============================================================

SYMBOL_MAP = {
    "(": "lparen", ")": "rparen",
    "[": "lbracket", "]": "rbracket",
    "{": "lbrace", "}": "rbrace",
    ":": "colon", ",": "comma", ".": "dot",
    "=": "assign", "+": "plus", "-": "minus",
    "*": "star", "/": "slash", "%": "percent",
    "==": "eq", "!=": "neq", "<": "lt", ">": "gt",
    "<=": "lte", ">=": "gte", "->": "arrow",
    "@": "at", ";": "semicolon", "#": "hash",
    "<>": "angled", "|": "pipe", "&": "ampersand",
}


# All operators across all languages
ALL_OPERATORS = set()
for ops in LANGUAGE_OPS.values():
    ALL_OPERATORS.update(ops)


# ============================================================
# TOKENIZER
# ============================================================

# Tokenizer regex: matches identifiers, numbers, strings, operators, symbols
# Combined with string concatenation to avoid regex syntax issues
_TOKEN_PATTERN = (
    r'"[^"]*"|'                    # double-quoted strings
    r"'[^']*'|"                    # single-quoted strings
    r'//|==|!=|<=|>=|'             # multi-char operators
    r'\*\*|->|=>|'                  # more multi-char
    r'\?\?|\?\.|<<|>>|'            # JS/Go/Zig operators
    r'::|:=|'                      # Rust/Go operators
    r'[+\-*/%<>=!().,:;@#\[\]{}]|' # single-char symbols
    r'\d+\.?\d*|'                  # numbers
    r'[A-Za-z_]\w*'               # identifiers
)
TOKEN_RE = re.compile(_TOKEN_PATTERN)


def classify_token(token: str, language: str = "python") -> tuple:
    """
    Classify a token for a specific language.
    """
    keywords = LANGUAGE_KEYWORDS.get(language, LANGUAGE_KEYWORDS["python"])
    types = LANGUAGE_TYPES.get(language, LANGUAGE_TYPES["python"])
    builtins = LANGUAGE_BUILTINS.get(language, LANGUAGE_BUILTINS["python"])
    ops = LANGUAGE_OPS.get(language, LANGUAGE_OPS["python"])
    
    # Normalize for COBOL (uppercase)
    token_upper = token.upper()
    
    # Handle COBOL special case - keywords are uppercase
    if language == "cobol" and token_upper in keywords:
        return (f"code:kw:{token_upper}", "kw")
    
    # Check operators first
    if token in ops or token in ALL_OPERATORS:
        op_name = SYMBOL_MAP.get(token, token)
        return (f"code:op:{op_name}", "op")
    
    # Check symbols
    if token in SYMBOL_MAP:
        return (f"code:sym:{SYMBOL_MAP[token]}", "sym")
    
    # Check keywords (case-insensitive for most)
    if token in keywords or token_upper in keywords:
        kw = token if token in keywords else token_upper
        return (f"code:kw:{kw}", "kw")
    
    # Check types
    if token in types:
        return (f"code:type:{token}", "type")
    
    # Check builtins
    if token in builtins:
        return (f"code:func:{token}", "func")
    
    # Literals
    if token.startswith('"') or token.startswith("'"):
        return ("code:lit:str", "lit")
    
    try:
        int(token)
        return ("code:lit:int", "lit")
    except ValueError:
        pass
    
    try:
        float(token)
        return ("code:lit:float", "lit")
    except ValueError:
        pass
    
    if token.lower() in ("true", "false"):
        return ("code:lit:bool", "lit")
    
    if token.lower() in ("none", "null", "nil"):
        return ("code:lit:none", "lit")
    
    # Default: variable/identifier
    return (f"code:var:{token}", "var")


def tokenize_code_line(code: str, language: str = "python") -> list:
    """
    Tokenize a line of code for a specific language.
    """
    supported = {"python", "rust", "javascript", "go", "ruby", "cobol", "zig"}
    if language not in supported:
        raise ValueError(f"Language '{language}' not supported. Use: {supported}")
    
    # Strip comments based on language
    if language == "python" and "#" in code:
        code = code[:code.index("#")]
    elif language in ("rust", "javascript", "go", "zig") and "//" in code:
        code = code[:code.index("//")]
    elif language == "ruby" and "#" in code:
        code = code[:code.index("#")]
    
    code = code.strip()
    if not code:
        return []
    
    raw_tokens = TOKEN_RE.findall(code)
    result = []
    
    for tok in raw_tokens:
        tok = tok.strip()
        if not tok:
            continue
        
        node_name, category = classify_token(tok, language)
        result.append(node_name)
    
    return result


# Keep alias for backwards compatibility
def assign_code_roles(tokens: list) -> list:
    """Assign code-specific roles to tokens."""
    if not tokens:
        return []
    
    roles = []
    n = len(tokens)
    
    for i, tok in enumerate(tokens):
        # Closing symbols
        if tok.startswith("code:sym:") and tok in (
            "code:sym:rparen", "code:sym:rbracket",
            "code:sym:rbrace", "code:sym:colon"
        ):
            roles.append(4)  # CLOSER
        # Keywords that start definitions
        elif tok.startswith("code:kw:"):
            if tok in ("code:kw:return", "code:kw:EXIT"):
                roles.append(9)  # RETURN
            else:
                roles.append(5)  # DECLARATION
        # Operators
        elif tok.startswith("code:op:"):
            roles.append(8)  # EXPRESSION
        # First token
        elif i == 0:
            roles.append(0)  # STARTER
        # Variables
        elif tok.startswith("code:var:"):
            roles.append(1)  # ACTOR
        else:
            roles.append(2)  # LINKER
    
    return roles


# Export supported languages
SUPPORTED_LANGUAGES = list(LANGUAGE_KEYWORDS.keys())