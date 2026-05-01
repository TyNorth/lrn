"""
Code Logic Module for LRN.
Boolean logic as spring equilibrium - parallel to Zero-Energy Arithmetic.
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN
from lrn.node import Node
from lrn.spring import Spring


# Constants
BOOLEAN_TRUE = "code:logic:true"
BOOLEAN_FALSE = "code:logic:false"

# Logic spring stiffness
LOGIC_COMPARISON_K = 80    # Comparison operator springs
LOGIC_AND_K = 60          # AND conjunction
LOGIC_OR_K = 40            # OR disjunction
LOGIC_NOT_K = -30          # NOT negation (repulsion)
LOGIC_BRANCH_K = 100       # Branch selection springs


def initialize_logic_nodes(lnn: LatticeNN) -> None:
    """Create boolean truth value nodes."""
    lnn.add_node(BOOLEAN_TRUE)
    lnn.nodes[BOOLEAN_TRUE].modality = "B"
    lnn.nodes[BOOLEAN_TRUE].tau = 1
    
    lnn.add_node(BOOLEAN_FALSE)
    lnn.nodes[BOOLEAN_FALSE].modality = "B"
    lnn.nodes[BOOLEAN_FALSE].tau = 1
    
    # True and False are antonyms (tensegrity)
    lnn.add_or_update_spring(
        BOOLEAN_TRUE, BOOLEAN_FALSE,
        stiffness=LOGIC_NOT_K,  # Negative = repulsion (negation)
        tau=2  # Causal - negation causes opposite
    )


def compare_to_springs(lnn: LatticeNN, var_node: str, op: str, value_node: str) -> str:
    """
    Create boolean constraint springs for comparison.
    
    Returns boolean constraint node name.
    """
    # Map operator to spring behavior
    COMPARISON_OPS = {
        ">": ("greater_than", LOGIC_COMPARISON_K),
        "<": ("less_than", LOGIC_COMPARISON_K),
        ">=": ("greater_equal", LOGIC_COMPARISON_K),
        "<=": ("less_equal", LOGIC_COMPARISON_K),
        "==": ("equal", LOGIC_COMPARISON_K),
        "!=": ("not_equal", -LOGIC_COMPARISON_K),  # Negative = repulsion
    }
    
    if op not in COMPARISON_OPS:
        return None
    
    comp_name, k = COMPARISON_OPS[op]
    bool_node = f"boolean:{var_node}_{comp_name}_{value_node}"
    
    lnn.add_node(bool_node)
    lnn.nodes[bool_node].modality = "B"
    lnn.nodes[bool_node].tau = 2  # Causal - comparison causes boolean
    
    # Variable connects to boolean node
    lnn.add_or_update_spring(var_node, bool_node, stiffness=k, tau=2)
    
    # Value connects to boolean node  
    lnn.add_or_update_spring(value_node, bool_node, stiffness=k, tau=2)
    
    return bool_node


def add_boolean_condition(lnn: LatticeNN, tokens: list) -> list:
    """
    Add boolean condition from token sequence.
    Handles: comparisons, and/or, not, truth values.
    
    Returns list of boolean constraint nodes created.
    """
    boolean_nodes = []
    
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        
        # Truth values
        if tok == "code:lit:bool":
            bool_node = BOOLEAN_TRUE if i+1 < len(tokens) and tokens[i+1] == "True" else BOOLEAN_FALSE
            boolean_nodes.append(bool_node)
            
        # Comparison operators - look for var op literal pattern
        if tok.startswith("code:op:"):
            op = tok.replace("code:op:", "")
            # Check for pattern: var op value
            if i >= 2 and i + 1 < len(tokens):
                var = tokens[i-1]
                val = tokens[i+1]
                if var.startswith("code:var:") and val.startswith("code:lit:"):
                    bool_node = compare_to_springs(lnn, var, op, val)
                    if bool_node:
                        boolean_nodes.append(bool_node)
        
        i += 1
    
    return boolean_nodes


def logic_residual(lnn: LatticeNN, condition_node: str, expected: bool, n_steps: int = 8) -> int:
    """
    Measure tension when evaluating a boolean condition.
    
    Lower residual = condition is more satisfied.
    expected=True: residual measures how TRUE the condition is
    expected=False: residual measures how FALSE the condition is
    
    Returns activation level of boolean node (0-100).
    """
    lnn.reset()
    
    if condition_node not in lnn.nodes:
        return 999  # Unknown condition = high residual
    
    # Pin the condition node
    lnn.nodes[condition_node].activation = 100
    lnn.nodes[condition_node].pinned = True
    
    # Propagate
    from lrn import propagate
    for _ in range(n_steps):
        propagate(lnn, n_steps=1)
    
    # Get residual as activation level
    bool_node = lnn.nodes.get(condition_node)
    return 100 - bool_node.activation if bool_node else 999


def solve_condition(lnn: LatticeNN, condition_node: str) -> bool:
    """
    Solve boolean condition - check which boolean pole has higher activation.
    """
    if condition_node not in lnn.nodes:
        return False  # Unknown = False
    
    # For simple boolean nodes (true/false), just check activation
    if condition_node == BOOLEAN_TRUE:
        # Pin true, propagate, check which has higher activation
        lnn.reset()
        lnn.nodes[BOOLEAN_TRUE].activation = 80
        lnn.nodes[BOOLEAN_TRUE].pinned = True
        from lrn import propagate
        propagate(lnn, n_steps=3)
        true_act = lnn.nodes[BOOLEAN_TRUE].activation
        false_act = lnn.nodes.get(BOOLEAN_FALSE, lnn.nodes.get("code:logic:false", lnn.nodes.get("logic:false", lnn.get_or_create(BOOLEAN_FALSE)))).activation
        return true_act > false_act
    
    if condition_node == BOOLEAN_FALSE:
        lnn.reset()
        lnn.nodes[BOOLEAN_FALSE].activation = 80
        lnn.nodes[BOOLEAN_FALSE].pinned = True
        from lrn import propagate
        propagate(lnn, n_steps=3)
        true_act = lnn.nodes.get(BOOLEAN_TRUE).activation
        false_act = lnn.nodes[BOOLEAN_FALSE].activation
        return true_act > false_act
    
    # For other boolean conditions, measure residual
    true_residual = logic_residual(lnn, condition_node, True)
    false_residual = logic_residual(lnn, condition_node, False)
    
    # Lower residual = more likely true
    return true_residual < false_residual


def add_logic_pattern(lnn: LatticeNN, pattern: str) -> dict:
    """
    Add a logic pattern to the lattice.
    Handles both code tokens and logic expressions.
    """
    from lrn.code_training import add_code_file
    
    stats = add_code_file(lnn, pattern)
    
    # Extract boolean conditions from pattern and add logic springs
    tokens = pattern.split()
    bool_nodes = add_boolean_condition(lnn, tokens)
    stats["boolean_nodes"] = len(bool_nodes)
    
    return stats


# --- Logic Training Patterns ---
# Patterns for learning boolean logic through springs

LOGIC_PATTERNS = [
    # Simple truth values
    "code:kw:if code:logic:true code:sym:colon",
    "code:kw:if code:logic:false code:sym:colon",
    
    # Comparisons (trained via code patterns, logic springs added separately)
    "code:kw:if code:var:x code:op:gt code:lit:int code:sym:colon",
    "code:kw:if code:var:x code:op:lt code:lit:int code:sym:colon",
    "code:kw:if code:var:x code:op:eq code:lit:int code:sym:colon",
    "code:kw:if code:var:x code:op:neq code:lit:int code:sym:colon",
    
    # Return on condition
    "code:kw:if code:var:x code:op:gt code:lit:int code:sym:colon code:kw:return code:lit:int",
    "code:kw:if code:var:x code:op:lt code:lit:int code:sym:colon code:kw:return code:lit:str",
    
    # If-else structure
    "code:kw:if code:var:condition code:sym:colon code:kw:return code:lit:int code:kw:else code:kw:return code:lit:str",
]


def train_logic_patterns(lnn: LatticeNN, repetitions: int = 40) -> dict:
    """
    Train boolean logic patterns.
    """
    stats = {"patterns": 0, "springs": 0, "boolean_nodes": 0}
    
    for _ in range(repetitions):
        for pattern in LOGIC_PATTERNS:
            result = add_logic_pattern(lnn, pattern)
            stats["patterns"] += 1
            stats["springs"] += result.get("springs", 0)
            stats["boolean_nodes"] += result.get("boolean_nodes", 0)
    
    return stats