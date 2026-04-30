#!/usr/bin/env python3
"""
Full Math Test Suite - 25 tests
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import json


def main():
    print("=" * 60)
    print("Full Math Test Suite - 25 Tests")
    print("=" * 60)
    
    from lrn.math_lattice import create_math_lattice, NumberLine
    
    ml = create_math_lattice()
    nl = NumberLine(ml)
    
    tests_passed = 0
    tests_total = 0
    
    # === ADDITION TESTS (8) ===
    print("\n[ADDITION TESTS]")
    
    # Forward
    for start, steps, expected in [(0, 3, 3), (5, 2, 7), (10, 5, 15), (0, 10, 10)]:
        ml.reset()
        result = nl.step_forward(start, steps, verbose=False)
        passed = result == expected
        tests_passed += passed
        tests_total += 1
        print(f"    {'✓' if passed else '✗'} {start}+{steps}={result}")
    
    # Edge cases
    ml.reset()
    result = nl.step_forward(-5, 7)  # -5 + 7 = 2
    passed = result == 2
    tests_passed += passed
    tests_total += 1
    print(f"    {'✓' if passed else '✗'} -5+7={result}")
    
    ml.reset()
    result = nl.step_forward(99, 1)  # Edge of range
    passed = result == 100
    tests_passed += passed
    tests_total += 1
    print(f"    {'✓' if passed else '✗'} 99+1={result}")
    
    # === SUBTRACTION TESTS (6) ===
    print("\n[SUBTRACTION TESTS]")
    
    for start, steps, expected in [(5, 2, 3), (3, 3, 0), (3, 5, -2), (0, 5, -5)]:
        ml.reset()
        result = nl.step_backward(start, steps, verbose=False)
        passed = result == expected
        tests_passed += passed
        tests_total += 1
        print(f"    {'✓' if passed else '✗'} {start}-{steps}={result}")
    
    ml.reset()
    result = nl.step_backward(8, 10)  # Into negatives
    passed = result == -2
    tests_passed += passed
    tests_total += 1
    print(f"    {'✓' if passed else '✗'} 8-10={result}")
    
    ml.reset()
    result = nl.step_backward(100, 100)  # Edge
    passed = result == 0
    tests_passed += passed
    tests_total += 1
    print(f"    {'✓' if passed else '✗'} 100-100={result}")
    
    # === MULTIPLICATION TESTS (5) ===
    print("\n[MULTIPLICATION TESTS]")
    
    for groups, size, expected in [(2, 3, 6), (3, 4, 12), (4, 4, 16), (0, 5, 0), (1, 7, 7)]:
        ml.reset()
        result = nl.multiply(groups, size, verbose=False)
        passed = result == expected
        tests_passed += passed
        tests_total += 1
        print(f"    {'✓' if passed else '✗'} {groups}×{size}={result}")
    
    # === DIVISION TESTS (4) ===
    print("\n[DIVISION TESTS]")
    
    for dividend, divisor, expected in [(6, 2, (3, 0)), (12, 4, (3, 0)), (10, 3, (3, 1)), (8, 2, (4, 0))]:
        ml.reset()
        result = nl.divide(dividend, divisor, verbose=False)
        passed = result == expected
        tests_passed += passed
        tests_total += 1
        print(f"    {'✓' if passed else '✗'} {dividend}÷{divisor}={result}")
    
    # === NUMBER LINE PROPERTIES (2) ===
    print("\n[NUMBER LINE PROPERTIES]")
    
    # Verify range
    passed = ml.number_line_min == -100 and ml.number_line_max == 100
    tests_passed += passed
    tests_total += 1
    print(f"    {'✓' if passed else '✗'} Range: {ml.number_line_min} to {ml.number_line_max}")
    
    # Verify number nodes exist
    nodes_exist = all(f"sensor:count:{i}" in ml.nodes for i in range(-100, 101, 10))
    passed = nodes_exist
    tests_passed += passed
    tests_total += 1
    print(f"    {'✓' if passed else '✗'} Number nodes present at intervals")
    
    # === BALANCE NODE TESTS ===
    print("\n[BALANCE NODES]")
    balance_nodes = len([n for n in ml.nodes if n.startswith("balance:")])
    passed = balance_nodes > 200
    tests_passed += passed
    tests_total += 1
    print(f"    {'✓' if passed else '✗'} {balance_nodes} balance nodes installed")
    
    print("\n" + "=" * 60)
    print(f"FINAL RESULTS: {tests_passed}/{tests_total} ({(tests_passed/tests_total*100):.0f}%)")
    print("=" * 60)
    
    # Save
    with open("/Users/tyarc/github/lrn/sys_test/math_full_results.json", "w") as f:
        json.dump({
            "passed": tests_passed,
            "total": tests_total,
            "percentage": round(tests_passed/tests_total*100, 1)
        }, f, indent=2)
    
    return tests_passed >= tests_total


if __name__ == "__main__":
    main()