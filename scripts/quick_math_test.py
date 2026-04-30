#!/usr/bin/env python3
"""Quick Math + Language Test"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn.math_lattice import MathLattice, NumberLine

print("=" * 50)
print("Quick Math Word Problems Test")
print("=" * 50)

ml = MathLattice()
ml.initialize_math()
nl = NumberLine(ml)

print(f"\n[Math Module] {len(ml.nodes)} nodes")

# Test basic math
print(f"\n[Basic Math]")
tests = [
    (1, 1, 2, "add"),
    (2, 2, 4, "add"),
    (3, 2, 6, "multiply"),
    (4, 3, 12, "multiply"),
]

passed = 0
for a, b, expected, op in tests:
    ml.reset()
    if op == "add":
        result = nl.step_forward(a, b, verbose=False)
    else:
        result = nl.multiply(a, b, verbose=False)
    
    status = "✓" if result == expected else "✗"
    print(f"  {status} {a} {op} {b} = {result} (expected {expected})")
    if result == expected:
        passed += 1

print(f"\n[Results] {passed}/{len(tests)}")

# Word problem style
print(f"\n[Word Problem Style]")
word_tests = [
    ("one plus one", 2),
    ("two plus two", 4),
    ("two times two", 4),
    ("three times three", 9),
]

for prompt, expected in word_tests:
    ml.reset()
    nums = prompt.lower().replace("plus", " + ").replace("times", " × ").split()
    try:
        if "+" in prompt:
            parts = prompt.replace("plus", "+").split("+")
            a = int(parts[0].strip().split()[-1])
            b = int(parts[1].strip())
            result = nl.step_forward(a, b, verbose=False)
        elif "times" in prompt:
            parts = prompt.replace("times", "×").split("×")
            a = int(parts[0].strip().split()[-1])
            b = int(parts[1].strip())
            result = nl.multiply(a, b, verbose=False)
        else:
            result = None
        
        if result == expected:
            print(f"  ✓ {prompt} = {result}")
            passed += 1
        else:
            print(f"  ✗ {prompt} = {result} (expected {expected})")
    except Exception as e:
        print(f"  ✗ {prompt} - error: {e}")

print(f"\n[Final] {passed}/{len(tests)+len(word_tests)}")