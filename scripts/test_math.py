#!/usr/bin/env python3
"""
Math Module Test Suite - 20 tests
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import json


def test_basic():
    """Initialize math lattice"""
    from lrn.math_lattice import create_math_lattice, NumberLine, EquationSolver
    
    print("=" * 60)
    print("Math Module Test Suite")
    print("=" * 60)
    
    ml = create_math_lattice()
    nl = NumberLine(ml)
    es = EquationSolver(ml, nl)
    
    return ml, nl, es


def test_number_line_forward(ml, nl):
    """Test: Step forward counting"""
    print("\n[1] Testing step_forward...")
    
    tests = [
        (0, 3, 3),
        (5, 2, 7),
        (10, 5, 15),
        (0, 10, 10),
    ]
    
    results = []
    for start, steps, expected in tests:
        ml.reset()
        result = nl.step_forward(start, steps, verbose=False)
        passed = result == expected
        results.append(passed)
        status = "✓" if passed else "✗"
        print(f"    {status} {start} + {steps} = {result} (expected {expected})")
    
    return results


def test_number_line_backward(ml, nl):
    """Test: Step backward counting (including negatives)"""
    print("\n[2] Testing step_backward...")
    
    tests = [
        (5, 2, 3),
        (3, 3, 0),
        (3, 5, -2),  # Crosses zero into negatives
        (0, 5, -5),
    ]
    
    results = []
    for start, steps, expected in tests:
        ml.reset()
        result = nl.step_backward(start, steps, verbose=False)
        passed = result == expected
        results.append(passed)
        status = "✓" if passed else "✗"
        print(f"    {status} {start} - {steps} = {result} (expected {expected})")
    
    return results


def test_multiplication(ml, nl):
    """Test: Multiplication as repeated counting"""
    print("\n[3] Testing multiplication...")
    
    tests = [
        (2, 3, 6),
        (3, 4, 12),
        (4, 4, 16),
        (0, 5, 0),
        (1, 7, 7),
    ]
    
    results = []
    for groups, size, expected in tests:
        ml.reset()
        result = nl.multiply(groups, size, verbose=False)
        passed = result == expected
        results.append(passed)
        status = "✓" if passed else "✗"
        print(f"    {status} {groups} × {size} = {result} (expected {expected})")
    
    return results


def test_division(ml, nl):
    """Test: Division as repeated backward counting"""
    print("\n[4] Testing division...")
    
    tests = [
        (6, 2, (3, 0)),
        (12, 4, (3, 0)),
        (10, 3, (3, 1)),
        (8, 2, (4, 0)),
        (7, 2, (3, 1)),
    ]
    
    results = []
    for dividend, divisor, expected in tests:
        ml.reset()
        result = nl.divide(dividend, divisor, verbose=False)
        passed = result == expected
        results.append(passed)
        status = "✓" if passed else "✗"
        print(f"    {status} {dividend} ÷ {divisor} = {result} (expected {expected})")
    
    return results


def test_addition_balance(ml, es):
    """Test: Addition via balance node tension"""
    print("\n[5] Testing addition balance nodes...")
    
    tests = [
        (1, 2, 3),
        (5, 3, 8),
        (0, 7, 7),
        (10, 10, 20),
    ]
    
    results = []
    for a, b, expected in tests:
        r_correct = es.evaluate_addition(a, b, expected)
        
        # Wrong answer adds tension
        wrong = (expected + 3) % 21 - 10
        r_wrong = es.evaluate_addition(a, b, wrong)
        
        # Correct should have lower or equal tension
        passed = abs(r_correct) <= abs(r_wrong)
        results.append(passed)
        status = "✓" if passed else "✗"
        print(f"    {status} {a}+{b}={expected}: tension={r_correct} vs wrong={r_wrong}")
    
    return results


def test_solve_addition(ml, es):
    """Test: Solve equations like 'a + ? = c'"""
    print("\n[6] Testing equation solving...")
    
    tests = [
        (3, 7, 4),   # 3 + ? = 7 -> 4
        (5, 8, 3),   # 5 + ? = 8 -> 3
        (0, 5, 5),   # 0 + ? = 5 -> 5
        (10, 15, 5), # 10 + ? = 15 -> 5
    ]
    
    results = []
    for a, sum_val, expected in tests:
        answer = es.solve_addition(a, sum_val - a, verbose=False)
        passed = answer == expected
        results.append(passed)
        status = "✓" if passed else "✗"
        print(f"    {status} {a} + ? = {sum_val} -> {answer} (expected {expected})")
    
    return results


def test_zero_crossing(ml, nl):
    """Test: Zero crossing - positive to negative"""
    print("\n[7] Testing zero crossing...")
    
    ml.reset()
    result = nl.step_backward(3, 8)  # 3 - 8 = -5
    passed = result == -5
    status = "✓" if passed else "✗"
    print(f"    {status} 3 - 8 = {result} (expected -5)")
    return [passed]


def test_negative_addition(ml, nl):
    """Test: Addition with negatives"""
    print("\n[8] Testing negative addition...")
    
    ml.reset()
    # -3 + 5 = 2
    result = nl.step_forward(-3, 5)
    passed = result == 2
    status = "✓" if passed else "✗"
    print(f"    {status} -3 + 5 = {result} (expected 2)")
    return [passed]


def test_subtraction_result(ml, es):
    """Test: Subtraction produces correct result"""
    print("\n[9] Testing subtraction...")
    
    # 5 - 2 = 3
    ml.reset()
    result = nl.step_backward(5, 2)
    passed = result == 3
    status = "✓" if passed else "✗"
    print(f"    {status} 5 - 2 = {result} (expected 3)")
    return [passed]


def run_all_tests():
    """Run complete test suite"""
    ml, nl, es = test_basic()
    
    all_results = []
    
    # Run all test groups
    all_results.extend(test_number_line_forward(ml, nl))
    all_results.extend(test_number_line_backward(ml, nl))
    all_results.extend(test_multiplication(ml, nl))
    all_results.extend(test_division(ml, nl))
    all_results.extend(test_addition_balance(ml, es))
    all_results.extend(test_solve_addition(ml, es))
    all_results.extend(test_zero_crossing(ml, nl))
    all_results.extend(test_negative_addition(ml, nl))
    all_results.extend(test_subtraction_result(ml, es))
    
    # Count passed
    passed = sum(all_results)
    total = len(all_results)
    parity = (passed / total) * 100 if total > 0 else 0
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} ({parity:.0f}%)")
    print("=" * 60)
    
    # Save results
    with open("/Users/tyarc/github/lrn/sys_test/math_test_results.json", "w") as f:
        json.dump({
            "passed": passed,
            "total": total,
            "parity": parity,
            "tests": all_results
        }, f, indent=2)
    
    return passed, total


if __name__ == "__main__":
    run_all_tests()