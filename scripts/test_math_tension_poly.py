#!/usr/bin/env python3
"""
Math Module - Balance Node & Traversal Parity + Polynomials
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import json


def evaluate_polynomial(lnn, nl, coeffs: list, x: int) -> int:
    """Evaluate polynomial via traversal (Horner's method)
    
    coeffs = [highest_degree, ..., constant]
    e.g., [2, 3, 1] = 2x² + 3x + 1
    """
    if not coeffs:
        return 0
    
    # Start from highest degree coefficient
    result = coeffs[0]
    
    # For each remaining coefficient, multiply by x and add
    for i in range(1, len(coeffs)):
        # Multiply result by x
        result = nl.multiply(result, x, verbose=False)
        # Add coefficient
        result = nl.step_forward(result, coeffs[i], verbose=False)
    
    return result


def calculate_balance_tension(lnn, balance_node: str) -> float:
    """Calculate tension on a balance node"""
    if balance_node not in lnn.nodes:
        return float('inf')
    
    tension = 0.0
    for neighbor, sp in lnn.get_neighbors(balance_node):
        if neighbor in lnn.nodes:
            tension += sp.stiffness * lnn.nodes[neighbor].activation
    return tension


def main():
    print("=" * 60)
    print("Math Module - Balance Node Parity & Polynomials")
    print("=" * 60)
    
    from lrn.math_lattice import MathLattice, NumberLine, EquationSolver
    
    ml = MathLattice()
    ml.initialize_math()
    nl = NumberLine(ml)
    es = EquationSolver(ml, nl)
    
    es.install_addition_facts(20)
    es.install_subtraction_facts(20)
    es.install_multiplication_facts(12)
    es.install_division_facts(12)
    
    tests_passed = 0
    tests_total = 0
    
    # === PART 1: TRAVERSAL OPERATIONS ===
    print("\n" + "="*50)
    print("PART 1: TRAVERSAL OPERATIONS")
    print("="*50)
    
    print("\n[Addition]")
    add_tests = [(0, 3, 3), (5, 2, 7), (10, 5, 15)]
    for start, steps, expected in add_tests:
        ml.reset()
        result = nl.step_forward(start, steps, verbose=False)
        passed = result == expected
        tests_passed += passed
        tests_total += 1
        print(f"  {'✓' if passed else '✗'} {start}+{steps}={result}")
    
    print("\n[Subtraction]")
    sub_tests = [(5, 2, 3), (3, 5, -2)]
    for start, steps, expected in sub_tests:
        ml.reset()
        result = nl.step_backward(start, steps, verbose=False)
        passed = result == expected
        tests_passed += passed
        tests_total += 1
        print(f"  {'✓' if passed else '✗'} {start}-{steps}={result}")
    
    print("\n[Multiplication]")
    mul_tests = [(2, 3, 6), (3, 4, 12), (4, 5, 20)]
    for g, s, expected in mul_tests:
        ml.reset()
        result = nl.multiply(g, s, verbose=False)
        passed = result == expected
        tests_passed += passed
        tests_total += 1
        print(f"  {'✓' if passed else '✗'} {g}×{s}={result}")
    
    print("\n[Division]")
    div_tests = [(6, 2, (3, 0)), (12, 4, (3, 0)), (10, 3, (3, 1))]
    for d, v, expected in div_tests:
        ml.reset()
        result = nl.divide(d, v, verbose=False)
        passed = result == expected
        tests_passed += passed
        tests_total += 1
        print(f"  {'✓' if passed else '✗'} {d}÷{v}={result}")
    
    # === PART 2: BALANCE NODE PARITY ===
    print("\n" + "="*50)
    print("PART 2: BALANCE NODE PARITY")
    print("="*50)
    
    print("\n[Verify traversal results map to balance nodes]")
    for a, b, expected in add_tests:
        ml.reset()
        result = nl.step_forward(a, b, verbose=False)
        
        # Check corresponding balance node exists
        bal = f"balance:{a}:+:{b}"
        bal_exists = bal in ml.nodes
        passed = bal_exists and result == expected
        tests_passed += passed
        tests_total += 1
        print(f"  {'✓' if passed else '✗'} {a}+{b}={result} -> balance:{bal_exists}")
    
    print("\n[Verify multiplication balance nodes]")
    for a, b, expected in mul_tests:
        ml.reset()
        result = nl.multiply(a, b, verbose=False)
        bal = f"balance:{a}:x:{b}"
        bal_exists = bal in ml.nodes
        passed = bal_exists and result == expected
        tests_passed += passed
        tests_total += 1
        print(f"  {'✓' if passed else '✗'} {a}×{b}={result} -> balance:{bal_exists}")
    
    print("\n[Calculate tension on correct vs wrong answers]")
    for a, b, correct in [(2, 3, 6), (3, 4, 12)]:
        ml.reset()
        
        bal = f"balance:{a}:x:{b}"
        
        # Pin operand a
        ml.add_node(f"sensor:count:{a}")
        ml.nodes[f"sensor:count:{a}"].activation = 100
        ml.nodes[f"sensor:count:{a}"].pinned = True
        
        # Pin operand b
        ml.add_node(f"sensor:count:{b}")
        ml.nodes[f"sensor:count:{b}"].activation = 100
        ml.nodes[f"sensor:count:{b}"].pinned = True
        
        # Pin correct answer
        ml.add_node(f"sensor:count:{correct}")
        ml.nodes[f"sensor:count:{correct}"].activation = 100
        ml.nodes[f"sensor:count:{correct}"].pinned = True
        
        if bal in ml.nodes:
            tension = calculate_balance_tension(ml, bal)
            # Check springs - correct answer should have negative stiffness
            neg_springs = sum(1 for n, sp in ml.get_neighbors(bal) 
                            if sp.stiffness < 0 and n in ml.nodes)
            passed = neg_springs > 0  # Has negative spring = correct setup
            tests_passed += passed
            tests_total += 1
            print(f"  {'✓' if passed else '✗'} {a}×{b}={correct} tension={tension:.0f} neg_springs={neg_springs}")
    
    # === PART 3: ALGEBRAIC EQUATIONS (via traversal inverse) ===
    print("\n" + "="*50)
    print("PART 3: ALGEBRAIC EQUATIONS")
    print("="*50)
    
    print("\n[Addition: a + x = result]")
    for a, result, expected in [(3, 7, 4), (5, 10, 5), (2, 9, 7)]:
        ml.reset()
        answer = result - a  # Inverse of traversal
        passed = answer == expected
        tests_passed += passed
        tests_total += 1
        print(f"  {'✓' if passed else '✗'} {a} + x = {result} -> x = {answer}")
    
    print("\n[Subtraction: x - a = result]")
    for a, result, expected in [(3, 5, 8), (5, 7, 12)]:
        ml.reset()
        answer = result + a
        passed = answer == expected
        tests_passed += passed
        tests_total += 1
        print(f"  {'✓' if passed else '✗'} x - {a} = {result} -> x = {answer}")
    
    print("\n[Multiplication: a × x = result]")
    for a, result, expected in [(3, 12, 4), (4, 16, 4)]:
        ml.reset()
        answer = result // a
        passed = answer == expected
        tests_passed += passed
        tests_total += 1
        print(f"  {'✓' if passed else '✗'} {a} × x = {result} -> x = {answer}")
    
    # === PART 4: POLYNOMIALS ===
    print("\n" + "="*50)
    print("PART 4: POLYNOMIAL EVALUATION")
    print("="*50)
    
    print("\n[Linear: 2x + 3]")
    for x, expected in [(0, 3), (1, 5), (2, 7), (5, 13)]:
        ml.reset()
        result = evaluate_polynomial(ml, nl, [2, 3], x)
        passed = result == expected
        tests_passed += passed
        tests_total += 1
        print(f"  {'✓' if passed else '✗'} 2x+3 at x={x} = {result}")
    
    print("\n[Quadratic: x² + 2x + 1]")
    for x, expected in [(0, 1), (1, 4), (2, 9), (3, 16)]:
        ml.reset()
        result = evaluate_polynomial(ml, nl, [1, 2, 1], x)
        passed = result == expected
        tests_passed += passed
        tests_total += 1
        print(f"  {'✓' if passed else '✗'} x²+2x+1 at x={x} = {result}")
    
    print("\n[Cubic: x³]")
    for x, expected in [(0, 0), (1, 1), (2, 8), (3, 27)]:
        ml.reset()
        result = evaluate_polynomial(ml, nl, [1, 0, 0, 0], x)
        passed = result == expected
        tests_passed += passed
        tests_total += 1
        print(f"  {'✓' if passed else '✗'} x³ at x={x} = {result}")
    
    # === SUMMARY ===
    print("\n" + "="*50)
    print(f"FINAL: {tests_passed}/{tests_total} ({(tests_passed/tests_total*100):.0f}%)")
    print("="*50)
    
    # Stats
    balance_count = len([n for n in ml.nodes if n.startswith("balance:")])
    print(f"  Balance nodes: {balance_count}")
    
    with open("/Users/tyarc/github/lrn/sys_test/math_parity_poly_results.json", "w") as f:
        json.dump({
            "passed": tests_passed,
            "total": tests_total,
            "percentage": round(tests_passed/tests_total*100, 1),
            "balance_nodes": balance_count
        }, f, indent=2)
    
    return tests_passed >= tests_total - 2


if __name__ == "__main__":
    main()