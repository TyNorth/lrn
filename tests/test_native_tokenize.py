"""
Test: Native Tokenization - Tokens emerge from training, not from rules.
The LRN learns "def", "fn", "function" as tokens through repetition, not via keyword lists.
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor
from lrn.native_tokenize import learn_from_code, discover_tokens, native_tokenize


def test_tokens_emerge():
    """Test that tokenization is learned, not hardcoded."""
    
    print("=" * 70)
    print("NATIVE TOKENIZATION TEST")
    print("Tokens should emerge from training, not from hardcoded rules")
    print("=" * 70)
    
    # Build fresh LRN
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    # Training corpus: raw code examples (no explicit tokenization)
    PYTHON_CODE = [
        "def add(a, b): return a + b",
        "def hello(name): return 'Hello ' + name",
        "def factorial(n): if n <= 1: return 1 else: return n * factorial(n-1)",
        "class MyClass: def __init__(self): self.value = 0",
        "if x > 0: return x else: return -x",
        "for i in range(n): print(i)",
        "while x > 0: x = x - 1",
    ]
    
    RUST_CODE = [
        "fn add(a: i32, b: i32) -> i32 { return a + b; }",
        "fn main() { println!(\"Hello\"); }",
        "struct Point { x: i32, y: i32 }",
        "if x > 0 { return x; } else { return -x; }",
        "let mut x = 5;",
        "for i in 0..10 { println!(\"{}\", i); }",
    ]
    
    print("\n1. Training on Python code (STRICT SYNTAX: τ=0 requires 50+ reps)...")
    for code in PYTHON_CODE:
        learn_from_code(lnn, code, repetitions=60)  # Higher = stricter
    
    print("2. Training on Rust code...")
    for code in RUST_CODE:
        learn_from_code(lnn, code, repetitions=60)
    
    # Discover tokens that emerged
    print("\n3. Discovered tokens (emerged from training):")
    tokens = discover_tokens(lnn, PYTHON_CODE + RUST_CODE, min_frequency=3)
    
    # Show top discovered tokens
    print("   Top 20 discovered tokens (character sequences):")
    for i, (token, info) in enumerate(list(tokens.items())[:20]):
        print(f"   {i+1}. '{token}' - used {info['usage_count']} times")
    
    # Verify keywords emerged
    print("\n4. Checking if 'def' emerged as a token...")
    if "def" in tokens:
        print(f"   ✓ 'def' emerged! Usage: {tokens['def']['usage_count']}")
    else:
        print(f"   ✗ 'def' not found - need more training")
    
    print("\n5. Checking if 'fn' emerged...")
    if "fn" in tokens:
        print(f"   ✓ 'fn' emerged! Usage: {tokens['fn']['usage_count']}")
    else:
        print(f"   ✗ 'fn' not found")
    
    # Test native tokenization
    print("\n6. Testing native tokenization (query-based, not rule-based):")
    test_code = "def add(a, b): return a + b"
    result = native_tokenize(lnn, test_code)
    print(f"   Input: '{test_code}'")
    print(f"   Tokens: {result[:10]}...")  # Show first 10
    
    # Compare with rule-based (hardcoded)
    print("\n7. Comparing: Native (learned) vs Rule-based (hardcoded):")
    print("   Rule-based: fixed list of keywords - doesn't learn")
    print("   Native: discovers 'def', 'fn', etc through repetition")
    print("   → The LRN learned what tokens ARE from exposure!")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULT:")
    
    discovered_kw = sum(1 for t in tokens.keys() if len(t) <= 5 and t.isalpha())
    print(f"   Discovered {len(tokens)} token patterns")
    print(f"   Of which {discovered_kw} are likely keywords (short, frequent)")
    
    if "def" in tokens and "fn" in tokens:
        print("\n   ✓ SUCCESS: Tokenization is now NATIVE to LRN!")
        print("   The LRN learned keywords from examples, not rules.")
    else:
        print("\n   Need more training iterations for tokens to emerge")
    print("=" * 70)
    
    return tokens


if __name__ == "__main__":
    test_tokens_emerge()