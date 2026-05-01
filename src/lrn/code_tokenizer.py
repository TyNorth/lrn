"""
Code-aware tokenization for LRN.
Maps Python source tokens to LRN node names with proper prefixes.
"""
import re

# Python keyword set
PYTHON_KEYWORDS = {
    "def", "return", "if", "else", "elif", "for", "while", "class",
    "import", "from", "as", "with", "try", "except", "finally", "raise",
    "pass", "break", "continue", "and", "or", "not", "is", "in",
    "True", "False", "None", "self", "lambda", "yield", "assert",
    "global", "nonlocal", "del",
}

PYTHON_TYPES = {
    "str", "int", "float", "bool", "list", "dict", "set", "tuple",
    "bytes", "range", "type", "object", "List", "Dict", "Set", "Tuple",
    "Optional", "Union", "Any",
}

BUILTIN_FUNCS = {
    "print", "len", "range", "str", "int", "float", "bool", "list",
    "dict", "set", "tuple", "type", "isinstance", "enumerate", "zip",
    "map", "filter", "sorted", "reversed", "sum", "min", "max", "abs",
    "input", "open", "repr", "hash", "id", "super", "property",
}

# Symbol mapping
SYMBOL_MAP = {
    "(": "lparen",
    ")": "rparen",
    "[": "lbracket",
    "]": "rbracket",
    "{": "lbrace",
    "}": "rbrace",
    ":": "colon",
    ",": "comma",
    ".": "dot",
    "=": "assign",
    "+": "plus",
    "-": "minus",
    "*": "star",
    "/": "slash",
    "//": "fslash",
    "%": "percent",
    "**": "dstar",
    "==": "eq",
    "!=": "neq",
    "<": "lt",
    ">": "gt",
    "<=": "lte",
    ">=": "gte",
    "->": "arrow",
    "@": "at",
}


OPERATORS = {"+", "-", "*", "/", "//", "%", "**", "==", "!=", "<", ">", "<=", ">=", "->"}


def classify_token(token: str) -> tuple:
    """
    Classify a token and return (node_name, category).
    Categories: kw, type, func, var, op, sym, lit
    """
    if token in OPERATORS:
        op_name = SYMBOL_MAP.get(token, token)
        return (f"code:op:{op_name}", "op")

    if token in SYMBOL_MAP:
        return (f"code:sym:{SYMBOL_MAP[token]}", "sym")

    if token in PYTHON_KEYWORDS:
        return (f"code:kw:{token}", "kw")

    if token in PYTHON_TYPES:
        return (f"code:type:{token}", "type")

    if token in BUILTIN_FUNCS:
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

    if token in ("True", "False"):
        return ("code:lit:bool", "lit")

    if token == "None":
        return ("code:lit:none", "lit")

    # Default: variable/identifier
    return (f"code:var:{token}", "var")


# Tokenizer regex: matches identifiers, numbers, strings, operators, symbols
TOKEN_RE = re.compile(
    r'"[^"]*"|'           # double-quoted strings
    r"'[^']*'|"           # single-quoted strings
    r'//|==|!=|<=|>=|'    # multi-char operators (checked before single-char)
    r'\*\*|->|'           # more multi-char
    r'[+\-*/%<>=!().,:;@\[\]{}]|'  # single-char symbols
    r'\d+\.?\d*|'         # numbers
    r'[A-Za-z_]\w*'       # identifiers
)


def tokenize_code_line(code: str, language: str = "python") -> list:
    """
    Tokenize a line of code into LRN node names.

    Example:
        "def foo(x): return x + 1"
        → ["code:kw:def", "code:var:foo", "code:sym:lparen",
           "code:var:x", "code:sym:rparen", "code:sym:colon",
           "code:kw:return", "code:var:x", "code:op:plus",
           "code:lit:int"]
    """
    if language != "python":
        raise ValueError(f"Language '{language}' not supported yet")

    # Strip comments
    if "#" in code:
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

        node_name, category = classify_token(tok)
        result.append(node_name)

    return result


def assign_code_roles(tokens: list) -> list:
    """
    Assign code-specific roles to tokens.

    Roles:
        0 STARTER  - first token of line/block
        1 ACTOR    - subject of operation (function names, variable names)
        2 LINKER   - connectors (operators, keywords like def/if/for)
        3 SETTLER  - result/target position
        4 CLOSER   - closing tokens (), }, :, etc.
        5 DECLARATION - variable/function definition start
        6 INVOCATION  - function call context
        7 ASSIGNMENT  - right-hand side of assignment
        8 EXPRESSION  - arithmetic/logic expression context
        9 RETURN      - return statement context
    """
    if not tokens:
        return []

    roles = []
    n = len(tokens)

    for i, tok in enumerate(tokens):
        role = _assign_single_code_role(tok, i, n, tokens)
        roles.append(role)

    return roles


def _assign_single_code_role(tok: str, idx: int, total: int, all_tokens: list) -> int:
    """Assign a single code role based on token type and context."""
    # Closing symbols are CLOSER
    if tok.startswith("code:sym:") and tok in (
        "code:sym:rparen", "code:sym:rbracket",
        "code:sym:rbrace", "code:sym:colon"
    ):
        return 4  # CLOSER

    # Keywords that start definitions
    if tok in ("code:kw:def", "code:kw:class", "code:kw:if",
               "code:kw:for", "code:kw:while", "code:kw:return"):
        if tok == "code:kw:return":
            return 9  # RETURN
        return 5  # DECLARATION

    # Operators are LINKER
    if tok.startswith("code:op:"):
        return 8  # EXPRESSION

    # Function calls: if followed by lparen, this is INVOCATION
    if idx + 1 < total and all_tokens[idx + 1] == "code:sym:lparen":
        if tok.startswith("code:var:") or tok.startswith("code:func:"):
            return 6  # INVOCATION

    # First token of line
    if idx == 0:
        return 0  # STARTER

    # After assignment operator
    if idx > 0 and all_tokens[idx - 1] == "code:sym:assign":
        return 7  # ASSIGNMENT

    # Variable names in non-special positions
    if tok.startswith("code:var:"):
        return 1  # ACTOR

    # Default: LINKER
    return 2  # LINKER
