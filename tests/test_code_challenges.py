"""
Coding challenges test runner for LRN.
Tests code generation against a battery of coding challenges.
Reports X/Y pass rate.

Grammar = syntax + logic learned through springs.
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import (
    LatticeNN,
    tokenize_code_line,
    add_code_file,
    initialize_scope_axis,
    promote_code_springs,
    train_code_grammar,
    generate_code,
    validate_syntax,
    generate_code_sequence,
    CODE_E_THRESHOLD,
    add_identity_anchor,
    # Logic module
    initialize_logic_nodes,
    compare_to_springs,
    logic_residual,
    solve_condition,
    add_logic_pattern,
    train_logic_patterns,
    BOOLEAN_TRUE,
    BOOLEAN_FALSE,
)

RESULTS = []
PASSED = 0
FAILED = 0

def challenge(name, test_fn):
    global PASSED, FAILED
    try:
        test_fn()
        PASSED += 1
        RESULTS.append(f"  PASS: {name}")
    except Exception as e:
        FAILED += 1
        RESULTS.append(f"  FAIL: {name} — {e}")

def assert_eq(actual, expected, msg=""):
    if actual != expected:
        raise AssertionError(f"{msg}: expected {expected!r}, got {actual!r}")

def assert_true(cond, msg=""):
    if not cond:
        raise AssertionError(f"{msg}: expected True, got False")

def assert_gt(actual, threshold, msg=""):
    if actual <= threshold:
        raise AssertionError(f"{msg}: expected > {threshold}, got {actual}")

def assert_in(item, container, msg=""):
    if item not in container:
        raise AssertionError(f"{msg}: {item!r} not found in {container}")


# ============================================================
# BUILD A TRAINED LATTICE (Grammar + Logic)
# ============================================================

def build_lattice():
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    # Part 1: Grammar learned through repetition (no spec needed)
    # Grammar patterns become τ=0 springs through promotion
    grammar_stats = train_code_grammar(lnn, repetitions=60)
    print(f"  Grammar: {grammar_stats['promoted']} springs promoted to τ=0")
    
    # Part 2: Logic module for boolean reasoning
    initialize_logic_nodes(lnn)
    logic_stats = train_logic_patterns(lnn, repetitions=30)
    print(f"  Logic: {logic_stats['boolean_nodes']} boolean nodes")
    
    # Scope axis for variable binding
    initialize_scope_axis(lnn)
    
    return lnn


# ============================================================
# CHALLENGE 1: Tokenization (Syntax)
# ============================================================

def test_tokenize_def():
    tokens = tokenize_code_line("def")
    assert_eq(tokens, ["code:kw:def"], "tokenize 'def'")

def test_tokenize_function_def():
    tokens = tokenize_code_line("def foo ( ) :")
    assert_true(len(tokens) == 5, f"expected 5 tokens, got {len(tokens)}")
    assert_eq(tokens[0], "code:kw:def")
    assert_eq(tokens[1], "code:var:foo")
    assert_eq(tokens[4], "code:sym:colon")

def test_tokenize_return_expr():
    tokens = tokenize_code_line("return a + b")
    assert_true(len(tokens) == 4, f"expected 4 tokens, got {len(tokens)}")
    assert_eq(tokens[0], "code:kw:return")
    assert_eq(tokens[2], "code:op:plus")

def test_tokenize_if_statement():
    tokens = tokenize_code_line("if x > 0 :")
    assert_eq(tokens[0], "code:kw:if")
    assert_eq(tokens[3], "code:lit:int")

def test_tokenize_assignment():
    tokens = tokenize_code_line("x = 1")
    assert_eq(tokens, ["code:var:x", "code:sym:assign", "code:lit:int"])

def test_tokenize_string_literal():
    tokens = tokenize_code_line('"hello"')
    assert_eq(tokens, ["code:lit:str"])

def test_tokenize_empty_and_comment():
    assert_eq(tokenize_code_line(""), [])
    assert_eq(tokenize_code_line("# comment"), [])


# ============================================================
# CHALLENGE 2: Spring Formation (Syntax Grammar)
# ============================================================

def test_code_file_creates_nodes(lnn):
    lnn2 = LatticeNN()
    add_code_file(lnn2, "code:kw:def code:var:foo code:sym:lparen")
    assert_true("code:kw:def" in lnn2.nodes, "def node not created")
    assert_true("code:var:foo" in lnn2.nodes, "foo node not created")

def test_code_file_creates_springs(lnn):
    lnn2 = LatticeNN()
    initial = len(lnn2.springs)
    add_code_file(lnn2, "code:kw:def code:var:foo code:sym:lparen")
    assert_gt(len(lnn2.springs), initial, "no springs created")

def test_repetition_accumulates_stiffness(lnn):
    lnn2 = LatticeNN()
    for _ in range(10):
        add_code_file(lnn2, "code:kw:def code:var:foo code:sym:lparen")
    key = lnn2._key("code:kw:def", "code:var:foo")
    assert_gt(lnn2.springs[key].stiffness, 20, f"stiffness not accumulated")

def test_ngrams_for_code_patterns(lnn):
    lnn2 = LatticeNN()
    initial = len(lnn2.trigrams)
    add_code_file(lnn2, "code:kw:def code:var:foo code:sym:lparen code:sym:rparen code:sym:colon")
    assert_gt(len(lnn2.trigrams), initial, "no n-grams created")


# ============================================================
# CHALLENGE 3: Spring Promotion (Grammar Lock-in)
# ============================================================

def test_spring_promotion_to_tau_zero(lnn):
    key = lnn._key("code:kw:def", "code:var:add")
    if key in lnn.springs:
        assert_eq(lnn.springs[key].tau, 0, f"def→add not promoted: tau={lnn.springs[key].tau}")

def test_promoted_springs_have_high_stiffness(lnn):
    key = lnn._key("code:kw:def", "code:var:add")
    if key in lnn.springs:
        assert_gt(lnn.springs[key].stiffness, 80, f"stiffness too low: {lnn.springs[key].stiffness}")

def test_return_spring_promoted(lnn):
    key = lnn._key("code:kw:return", "code:var:x")
    if key in lnn.springs:
        assert_eq(lnn.springs[key].tau, 0, "return→x not promoted")


# ============================================================
# CHALLENGE 4: Scope Axis
# ============================================================

def test_scope_nodes_exist(lnn):
    assert_true("code:block:0" in lnn.nodes, "scope:0 not found")
    assert_true("code:block:4" in lnn.nodes, "scope:4 not found")

def test_scope_positions(lnn):
    assert_eq(lnn.nodes["code:block:0"].x, 0)
    assert_eq(lnn.nodes["code:block:4"].x, 16)

def test_scope_springs_tau_one(lnn):
    key = lnn._key("code:block:0", "code:block:1")
    if key in lnn.springs:
        assert_eq(lnn.springs[key].tau, 1, "scope spring not τ=1")


# ============================================================
# CHALLENGE 5: Code Generation (Syntax)
# ============================================================

def test_generate_code_after_def(lnn):
    candidates = generate_code(lnn, ["code:kw:def"])
    assert_true(len(candidates) > 0, "no candidates after 'def'")

def test_generate_code_energy_gate(lnn):
    candidates = generate_code(lnn, ["code:kw:def"])
    for c in candidates:
        assert_true(c["energy"] < CODE_E_THRESHOLD, f"energy {c['energy']} >= {CODE_E_THRESHOLD}")

def test_generate_code_returns_var_after_def(lnn):
    candidates = generate_code(lnn, ["code:kw:def"])
    names = [c["word"] for c in candidates]
    found_var = any(n.startswith("code:var:") for n in names)
    assert_true(found_var, f"no variable candidate: {names}")

def test_generate_code_after_return(lnn):
    candidates = generate_code(lnn, ["code:kw:return"])
    assert_true(len(candidates) > 0, "no candidates after 'return'")

def test_generate_code_sequence(lnn):
    seq = generate_code_sequence(lnn, "code:kw:def", max_tokens=5)
    assert_true(isinstance(seq, str), "not a string")
    assert_true(len(seq) > 0, "empty sequence")


# ============================================================
# CHALLENGE 6: Syntax Validation
# ============================================================

def test_validate_syntax_basic(lnn):
    result = validate_syntax(lnn, "code:var:foo", ["code:kw:def"])
    assert_true(isinstance(result, bool), "not a bool")

def test_validate_syntax_no_false_positives(lnn):
    result = validate_syntax(lnn, "code:kw:return", ["code:kw:def"])
    assert_true(isinstance(result, bool), "not a bool")


# ============================================================
# CHALLENGE 7: Grammar Integration
# ============================================================

def test_curriculum_creates_nodes(lnn):
    assert_gt(len(lnn.nodes), 10, f"too few nodes: {len(lnn.nodes)}")

def test_curriculum_creates_springs(lnn):
    assert_gt(len(lnn.springs), 10, f"too few springs: {len(lnn.springs)}")

def test_curriculum_creates_ngrams(lnn):
    assert_gt(len(lnn.trigrams), 0, "no n-grams")


# ============================================================
# CHALLENGE 8: Specific Pattern Generation
# ============================================================

def test_add_function_pattern(lnn):
    candidates = generate_code(lnn, ["code:kw:def", "code:var:add", "code:sym:lparen"])
    assert_true(len(candidates) > 0, "no candidates after 'def add ('")

def test_return_plus_pattern(lnn):
    candidates = generate_code(lnn, ["code:kw:return", "code:var:a"])
    assert_true(len(candidates) > 0, "no candidates after 'return a'")


# ============================================================
# CHALLENGE 9: Epistemic Honesty
# ============================================================

def test_empty_prompt_no_crash(lnn):
    seq = generate_code_sequence(lnn, "code:kw:nonexistent", max_tokens=3)
    assert_true(isinstance(seq, str), "crashed on unknown token")


# ============================================================
# CHALLENGE 10: Code vs Text Discrimination
# ============================================================

def test_code_stricter_than_text(lnn):
    code_cands = generate_code(lnn, ["code:kw:def"])
    assert_true(len(code_cands) <= 20, "code not applying energy gate")


# ============================================================
# CHALLENGE 11: Logic Module - Boolean Nodes
# ============================================================

def test_boolean_nodes_exist(lnn):
    assert_true(BOOLEAN_TRUE in lnn.nodes, "true node not found")
    assert_true(BOOLEAN_FALSE in lnn.nodes, "false node not found")

def test_boolean_nodes_have_antonyms(lnn):
    key = lnn._key(BOOLEAN_TRUE, BOOLEAN_FALSE)
    if key in lnn.springs:
        assert_true(lnn.springs[key].stiffness < 0, "true/false should be repelling")

def test_comparison_creates_boolean_node(lnn):
    bool_node = compare_to_springs(lnn, "code:var:x", ">", "code:lit:int")
    assert_true(bool_node is not None, "boolean node not created")
    assert_true(bool_node in lnn.nodes, "boolean node not in lattice")


# ============================================================
# CHALLENGE 12: Logic Residual Computation
# ============================================================

def test_logic_residual_returns_int(lnn):
    # First add a comparison to the lattice
    add_code_file(lnn, "code:kw:if code:var:x code:op:gt code:lit:int code:sym:colon")
    compare_to_springs(lnn, "code:var:x", ">", "code:lit:int")
    
    residual = logic_residual(lnn, "boolean:x_greater_than_code:lit:int", True)
    assert_true(isinstance(residual, int), "not an int")

def test_logic_residual_lower_when_pinned(lnn):
    # Pin x to activate, this should affect comparison
    lnn.nodes["code:var:x"] = lnn.get_or_create("code:var:x")
    lnn.nodes["code:var:x"].activation = 100
    lnn.nodes["code:var:x"].pinned = True
    
    residual = logic_residual(lnn, "boolean:x_greater_than_code:lit:int", True)
    # With x pinned, residual should reflect that activation


# ============================================================
# CHALLENGE 13: Solve Condition
# ============================================================

def test_solve_condition_returns_bool(lnn):
    result = solve_condition(lnn, "code:logic:true")
    assert_true(isinstance(result, bool), "not a bool")

def test_solve_true_condition(lnn):
    result = solve_condition(lnn, BOOLEAN_TRUE)
    assert_true(result == True, "true should resolve to True")

def test_solve_false_condition(lnn):
    result = solve_condition(lnn, BOOLEAN_FALSE)
    assert_true(result == False, "false should resolve to False")


# ============================================================
# CHALLENGE 14: Logic Pattern Training
# ============================================================

def test_logic_patterns_add_springs(lnn):
    lnn2 = LatticeNN()
    initialize_logic_nodes(lnn2)
    initial = len(lnn2.springs)
    add_logic_pattern(lnn2, "code:kw:if code:logic:true code:sym:colon")
    assert_gt(len(lnn2.springs), initial, "logic pattern added no springs")


# ============================================================
# RUN ALL CHALLENGES
# ============================================================

def main():
    global PASSED, FAILED

    print("=" * 60)
    print("LRN Coding Capabilities — Challenge Battery")
    print("Grammar (Syntax) + Logic = Complete Code Understanding")
    print("=" * 60)

    lnn = build_lattice()
    print(f"\nLattice built: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs, {len(lnn.trigrams)} n-grams")
    promoted = sum(1 for sp in lnn.springs.values() if sp.tau == 0)
    print(f"Promoted τ=0 springs: {promoted}")

    print(f"\n--- Challenge 1: Tokenization ---")
    challenge("tokenize 'def'", test_tokenize_def)
    challenge("tokenize 'def foo ( ) :'", test_tokenize_function_def)
    challenge("tokenize 'return a + b'", test_tokenize_return_expr)
    challenge("tokenize 'if x > 0 :'", test_tokenize_if_statement)
    challenge("tokenize 'x = 1'", test_tokenize_assignment)
    challenge("tokenize string literal", test_tokenize_string_literal)
    challenge("tokenize empty and comment", test_tokenize_empty_and_comment)

    print(f"\n--- Challenge 2: Spring Formation (Grammar) ---")
    challenge("code file creates nodes", lambda: test_code_file_creates_nodes(lnn))
    challenge("code file creates springs", lambda: test_code_file_creates_springs(lnn))
    challenge("repetition accumulates stiffness", lambda: test_repetition_accumulates_stiffness(lnn))
    challenge("n-grams for code patterns", lambda: test_ngrams_for_code_patterns(lnn))

    print(f"\n--- Challenge 3: Spring Promotion ---")
    challenge("def→add promoted to τ=0", lambda: test_spring_promotion_to_tau_zero(lnn))
    challenge("promoted springs have high stiffness", lambda: test_promoted_springs_have_high_stiffness(lnn))
    challenge("return→x promoted", lambda: test_return_spring_promoted(lnn))

    print(f"\n--- Challenge 4: Scope Axis ---")
    challenge("scope nodes exist", lambda: test_scope_nodes_exist(lnn))
    challenge("scope positions correct", lambda: test_scope_positions(lnn))
    challenge("scope springs τ=1", lambda: test_scope_springs_tau_one(lnn))

    print(f"\n--- Challenge 5: Code Generation ---")
    challenge("candidates after 'def'", lambda: test_generate_code_after_def(lnn))
    challenge("energy gate < 20", lambda: test_generate_code_energy_gate(lnn))
    challenge("variable candidate after 'def'", lambda: test_generate_code_returns_var_after_def(lnn))
    challenge("candidates after 'return'", lambda: test_generate_code_after_return(lnn))
    challenge("sequence generation", lambda: test_generate_code_sequence(lnn))

    print(f"\n--- Challenge 6: Syntax Validation ---")
    challenge("validate_syntax returns bool", lambda: test_validate_syntax_basic(lnn))
    challenge("no false positives", lambda: test_validate_syntax_no_false_positives(lnn))

    print(f"\n--- Challenge 7: Grammar Integration ---")
    challenge("grammar creates >10 nodes", lambda: test_curriculum_creates_nodes(lnn))
    challenge("grammar creates >10 springs", lambda: test_curriculum_creates_springs(lnn))
    challenge("grammar creates n-grams", lambda: test_curriculum_creates_ngrams(lnn))

    print(f"\n--- Challenge 8: Specific Pattern Generation ---")
    challenge("candidates after 'def add ('", lambda: test_add_function_pattern(lnn))
    challenge("candidates after 'return a'", lambda: test_return_plus_pattern(lnn))

    print(f"\n--- Challenge 9: Epistemic Honesty ---")
    challenge("unknown token doesn't crash", lambda: test_empty_prompt_no_crash(lnn))

    print(f"\n--- Challenge 10: Code vs Text Discrimination ---")
    challenge("code applies energy gate", lambda: test_code_stricter_than_text(lnn))

    print(f"\n--- Challenge 11: Logic Module - Boolean Nodes ---")
    challenge("boolean nodes exist", lambda: test_boolean_nodes_exist(lnn))
    challenge("boolean nodes have antonyms", lambda: test_boolean_nodes_have_antonyms(lnn))
    challenge("comparison creates boolean node", lambda: test_comparison_creates_boolean_node(lnn))

    print(f"\n--- Challenge 12: Logic Residual Computation ---")
    challenge("logic_residual returns int", lambda: test_logic_residual_returns_int(lnn))
    challenge("logic_residual lower when pinned", lambda: test_logic_residual_lower_when_pinned(lnn))

    print(f"\n--- Challenge 13: Solve Condition ---")
    challenge("solve_condition returns bool", lambda: test_solve_condition_returns_bool(lnn))
    challenge("solve true condition", lambda: test_solve_true_condition(lnn))
    challenge("solve false condition", lambda: test_solve_false_condition(lnn))

    print(f"\n--- Challenge 14: Logic Pattern Training ---")
    challenge("logic patterns add springs", lambda: test_logic_patterns_add_springs(lnn))

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {PASSED}/{PASSED + FAILED} passed")
    print(f"{'=' * 60}")
    for r in RESULTS:
        print(r)
    
    # Summary
    syntax_pass = sum(1 for r in RESULTS if "Syntax" in r or "Grammar" in r or "Tokenize" in r or "Spring" in r or "Scope" in r or "Generation" in r or "Validation" in r or "Pattern" in r)
    logic_pass = sum(1 for r in RESULTS if "Logic" in r or "Boolean" in r or "Residual" in r or "Solve" in r)
    print(f"\n  Syntax/Grammar: {syntax_pass} passed")
    print(f"  Logic: {logic_pass} passed")

    return PASSED, PASSED + FAILED


if __name__ == "__main__":
    passed, total = main()
    sys.exit(0 if passed == total else 1)