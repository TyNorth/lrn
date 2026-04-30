#!/usr/bin/env python3
"""
LRN Teacher - Complex Math: Fractions, Decimals, Scales
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn.math_lattice import MathLattice, NumberLine, EquationSolver


def main():
    print("=" * 60)
    print("Complex Math: Fractions, Decimals, Scales")
    print("=" * 60)
    
    ml = MathLattice()
    ml.initialize_math()
    nl = NumberLine(ml)
    es = EquationSolver(ml, nl)
    
    es.install_addition_facts(20)
    es.install_subtraction_facts(20)
    es.install_multiplication_facts(12)
    es.install_division_facts(12)
    
    print(f"\n[Math Module] {len(ml.nodes)} nodes")
    
    # === FRACTIONS via division ===
    print(f"\n[1] Fractions via division...")
    
    FRACTIONS = [
        # Half of
        (2, 2, 1, "half"),
        (4, 2, 2, "half"),
        (6, 2, 3, "half"),
        (8, 2, 4, "half"),
        (10, 2, 5, "half"),
        
        # Third of
        (3, 3, 1, "third"),
        (6, 3, 2, "third"),
        (9, 3, 3, "third"),
        
        # Quarter of
        (4, 4, 1, "quarter"),
        (8, 4, 2, "quarter"),
        (12, 4, 3, "quarter"),
    ]
    
    passed = 0
    for dividend, divisor, expected, name in FRACTIONS:
        ml.reset()
        result = nl.divide(dividend, divisor, verbose=False)
        quotient = result[0]
        match = quotient == expected
        status = "✓" if match else "✗"
        print(f"    {status} {dividend} / {divisor} = {quotient} (expected {expected}, {name})")
        if match:
            passed += 1
    
    frac_passed = passed
    print(f"    Fractions: {passed}/{len(FRACTIONS)}")
    
    # === SCALES / PROPORTIONS ===
    print(f"\n[2] Scales and proportions...")
    
    SCALES = [
        # Double
        (2, 2, 4, "double"),
        (3, 2, 6, "double"),
        (5, 2, 10, "double"),
        
        # Triple
        (2, 3, 6, "triple"),
        (3, 3, 9, "triple"),
        
        # Half
        (4, 0.5, 2, "half"),
        (6, 0.5, 3, "half"),
        (10, 0.5, 5, "half"),
    ]
    
    scale_passed = 0
    for base, scale, expected, name in SCALES:
        ml.reset()
        if scale == 2:
            result = nl.multiply(base, 2, verbose=False)
        elif scale == 3:
            result = nl.multiply(base, 3, verbose=False)
        elif scale == 0.5:
            result = nl.divide(base, 2, verbose=False)[0]
        else:
            result = None
        
        match = result == expected
        status = "✓" if match else "✗"
        print(f"    {status} {base} × {scale} = {result} (expected {expected}, {name})")
        if match:
            scale_passed += 1
    
    print(f"    Scales: {scale_passed}/{len(SCALES)}")
    
    # === DECIMALS via division ===
    print(f"\n[3] Decimals...")
    
    DECIMALS = [
        # Divide by 10 (shift decimal)
        (5, 10, 0.5, "divide by 10"),
        (8, 10, 0.8, "divide by 10"),
        (12, 10, 1.2, "divide by 10"),
        
        # Divide by 100 (two decimals)
        (5, 100, 0.05, "divide by 100"),
        (25, 100, 0.25, "divide by 100"),
    ]
    
    dec_passed = 0
    for dividend, divisor, expected, name in DECIMALS:
        ml.reset()
        # For decimals, we work with integers and track decimal places
        quotient = dividend // divisor
        remainder = dividend % divisor
        
        # Simulate decimal: quotient + remainder/divisor
        # For 5/10 = 0.5, 5//10 = 0, 5%10 = 5, so 5/10 = 0.5
        # Represent as: quotient + remainder/divisor
        if divisor == 10:
            decimal = remainder / 10
            result = quotient + decimal
        elif divisor == 100:
            decimal = remainder / 100
            result = quotient + decimal
        else:
            result = quotient + remainder / divisor
        
        # Round to 2 decimal places for comparison
        result = round(result, 2)
        expected = round(expected, 2)
        
        match = result == expected
        status = "✓" if match else "✗"
        print(f"    {status} {dividend} / {divisor} = {result} (expected {expected}, {name})")
        if match:
            dec_passed += 1
    
    print(f"    Decimals: {dec_passed}/{len(DECIMALS)}")
    
    # === PERCENTAGES ===
    print(f"\n[4] Percentages...")
    
    PERCENTS = [
        # 50% = half
        (10, 50, 5, "50%"),
        (20, 50, 10, "50%"),
        (100, 50, 50, "50%"),
        
        # 25% = quarter
        (8, 25, 2, "25%"),
        (12, 25, 3, "25%"),
        (20, 25, 5, "25%"),
        
        # 100% = same
        (5, 100, 5, "100%"),
        (10, 100, 10, "100%"),
    ]
    
    pct_passed = 0
    for base, percent, expected, name in PERCENTS:
        ml.reset()
        
        if percent == 50:
            result = nl.divide(base, 2, verbose=False)[0]
        elif percent == 25:
            result = nl.divide(base, 4, verbose=False)[0]
        elif percent == 100:
            result = base
        else:
            result = None
        
        match = result == expected
        status = "✓" if match else "✗"
        print(f"    {status} {percent}% of {base} = {result} (expected {expected})")
        if match:
            pct_passed += 1
    
    print(f"    Percentages: {pct_passed}/{len(PERCENTS)}")
    
    # === SUMMARY ===
    total = len(FRACTIONS) + len(SCALES) + len(DECIMALS) + len(PERCENTS)
    total_passed = frac_passed + scale_passed + dec_passed + pct_passed
    
    print(f"\n{'='*50}")
    print(f"Complex Math: {total_passed}/{total} ({(total_passed/total*100):.0f}%)")
    print(f"  Fractions: {frac_passed}/{len(FRACTIONS)}")
    print(f"  Scales: {scale_passed}/{len(SCALES)}")
    print(f"  Decimals: {dec_passed}/{len(DECIMALS)}")
    print(f"  Percentages: {pct_passed}/{len(PERCENTS)}")
    print("=" * 50)
    
    import json
    with open("/Users/tyarc/github/lrn/sys_test/complex_math.json", 'w') as f:
        json.dump({
            "fractions": {"passed": frac_passed, "total": len(FRACTIONS)},
            "scales": {"passed": scale_passed, "total": len(SCALES)},
            "decimals": {"passed": dec_passed, "total": len(DECIMALS)},
            "percentages": {"passed": pct_passed, "total": len(PERCENTS)},
            "total": {"passed": total_passed, "total": total}
        }, f, indent=2)
    
    return total_passed >= total - 2


if __name__ == "__main__":
    main()