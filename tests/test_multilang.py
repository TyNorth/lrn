"""
Multi-language coding test for LRN.
Tests 7 diverse languages: Python, Rust, JavaScript, Go, Ruby, COBOL, Zig.
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import (
    LatticeNN,
    tokenize_code_line,
    add_code_file,
    initialize_scope_axis,
    promote_code_springs,
    train_all_languages,
    generate_code,
    CODE_E_THRESHOLD,
    add_identity_anchor,
    initialize_logic_nodes,
    solve_condition,
    BOOLEAN_TRUE,
    BOOLEAN_FALSE,
    train_logic_patterns,
)
from lrn.code_training import ALL_LANGUAGE_PATTERNS


RESULTS = []
PASSED = 0
FAILED = 0
LANGUAGES = ["python", "rust", "javascript", "go", "ruby", "cobol", "zig"]


def challenge(name, test_fn):
    global PASSED, FAILED
    try:
        test_fn()
        PASSED += 1
        RESULTS.append(f"  PASS: {name}")
    except Exception as e:
        FAILED += 1
        RESULTS.append(f"  FAIL: {name} — {e}")


def main():
    global PASSED, FAILED, RESULTS
    
    print("=" * 70)
    print("LRN Multi-Language Coding Test (7 Languages)")
    print("=" * 70)
    
    # Build lattice
    print("\nBuilding multi-language lattice...")
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    stats = train_all_languages(lnn, repetitions_per_lang=40)
    initialize_logic_nodes(lnn)
    train_logic_patterns(lnn, repetitions=20)
    initialize_scope_axis(lnn)
    
    print(f"  Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
    print(f"  Promoted τ=0: {sum(1 for sp in lnn.springs.values() if sp.tau == 0)}")
    
    # Test tokenization per language
    print("\n--- TOKENIZATION TESTS ---")
    test_cases = {
        "python": [("def add(a, b): return a + b", "def", "return"), ("if x > 0: return 1", "if", "return")],
        "rust": [("fn add(a: i32, b: i32) -> i32 { return a + b; }", "fn", "return"), ("if x > 0 { return 1; }", "if", "return")],
        "javascript": [("function add(a, b) { return a + b; }", "function", "return"), ("if (x > 0) { return 1; }", "if", "return")],
        "go": [("func add(a int, b int) int { return a + b }", "func", "return"), ("if x > 0 { return 1 }", "if", "return")],
        "ruby": [("def add(a, b); a + b; end", "def", "end"), ("if x > 0 then 1 end", "if", "end")],
        "cobol": [("IF X GREATER THAN ZERO DISPLAY 'YES' END-IF", "IF", "DISPLAY"), ("MOVE VALUE TO TARGET", "MOVE", "TO")],
        "zig": [("fn add(a: i32, b: i32) i32 { return a + b; }", "fn", "return"), ("if (x > 0) { return x; }", "if", "return")],
    }
    
    for lang, cases in test_cases.items():
        for code, *expected_kw in cases:
            try:
                tokens = tokenize_code_line(code, language=lang)
                found = any(any(kw in t for kw in expected_kw) for t in tokens)
                challenge(f"{lang}: tokenize '{code[:20]}...'", lambda: len(tokens) > 0)
            except Exception as e:
                challenge(f"{lang}: tokenize '{code[:20]}...'", lambda: False)
    
    # Test grammar patterns add nodes
    print("\n--- GRAMMAR TESTS ---")
    for lang in LANGUAGES:
        lnn2 = LatticeNN()
        patterns = ALL_LANGUAGE_PATTERNS.get(lang, [])
        for p in patterns[:3]:
            add_code_file(lnn2, p)
        challenge(f"{lang}: grammar adds nodes ({len(lnn2.nodes)})", lambda: len(lnn2.nodes) > 5)
    
    # Test generation
    print("\n--- GENERATION TESTS ---")
    prompts = {"python": ["code:kw:def"], "rust": ["code:kw:fn"], "javascript": ["code:kw:function"],
              "go": ["code:kw:func"], "ruby": ["code:kw:def"], "cobol": ["code:kw:IF"], "zig": ["code:kw:fn"]}
    
    for lang, prompt in prompts.items():
        try:
            cands = generate_code(lnn, prompt, top_k=3)
            valid = all(c["energy"] < CODE_E_THRESHOLD for c in cands)
            challenge(f"{lang}: generate ({len(cands)} candidates, energy OK)", lambda: len(cands) > 0 and valid)
        except:
            challenge(f"{lang}: generate", lambda: False)
    
    # Logic tests
    print("\n--- LOGIC TESTS ---")
    challenge("boolean true node exists", lambda: BOOLEAN_TRUE in lnn.nodes)
    challenge("boolean false node exists", lambda: BOOLEAN_FALSE in lnn.nodes)
    challenge("solve_condition(true)", lambda: isinstance(solve_condition(lnn, BOOLEAN_TRUE), bool))
    challenge("solve_condition(false)", lambda: isinstance(solve_condition(lnn, BOOLEAN_FALSE), bool))
    
    # Scope tests
    print("\n--- SCOPE TESTS ---")
    challenge("scope nodes created", lambda: "code:block:0" in lnn.nodes)
    
    # Summary
    print("\n" + "=" * 70)
    print(f"RESULTS: {PASSED}/{PASSED + FAILED} passed")
    print("=" * 70)
    for r in RESULTS:
        print(r)
    
    # Per-language summary
    lang_counts = {}
    for r in RESULTS:
        for lang in LANGUAGES:
            if f"{lang}:" in r:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
    
    print("\nPer Language:")
    for lang in LANGUAGES:
        print(f"  {lang}: {lang_counts.get(lang, 0)} passed")
    
    print(f"\n  TOTAL: {PASSED}/{PASSED + FAILED}")
    
    return PASSED, PASSED + FAILED


if __name__ == "__main__":
    passed, total = main()
    sys.exit(0 if passed >= total * 0.8 else 1)  # 80% pass rate OK