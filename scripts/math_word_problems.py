#!/usr/bin/env python3
"""
LRN Teacher - Math Word Problems: Both Language + Math Module
1. Train language model for word understanding
2. Use math module for actual calculations
3. Combine for word problem solving
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import json
from lrn.math_lattice import MathLattice, NumberLine, EquationSolver
from lrn import add_sentence, generate


def solve_word_problem(nl, ml, prompt: str) -> int:
    """Solve a math word problem using the number line"""
    prompt = prompt.lower()
    
    # Detect operation and numbers
    if " and " in prompt:
        parts = prompt.replace("are", "").replace("is", "").strip().split(" and ")
        try:
            nums = [int(p.replace("one", "1").replace("two", "2").replace("three", "3")
                    .replace("four", "4").replace("five", "5").replace("six", "6")
                    .replace("seven", "7").replace("eight", "8").replace("nine", "9")
                    .replace("ten", "10").split()[0]) for p in parts if p.strip()]
            if len(nums) == 2:
                return nl.step_forward(nums[0], nums[1], verbose=False)
        except:
            pass
    
    if "times" in prompt:
        parts = prompt.split("times")
        try:
            a = int(parts[0].strip().split()[-1].replace("one", "1").replace("two", "2").replace("three", "3"))
            b = int(parts[1].split()[0].replace("one", "1").replace("two", "2").replace("three", "3"))
            return nl.multiply(a, b, verbose=False)
        except:
            pass
    
    return None


def main():
    print("=" * 60)
    print("Math Word Problems: Language + Math Module")
    print("=" * 60)
    
    # Setup math module
    ml = MathLattice()
    ml.initialize_math()
    nl = NumberLine(ml)
    es = EquationSolver(ml, nl)
    
    es.install_addition_facts(20)
    es.install_subtraction_facts(20)
    es.install_multiplication_facts(12)
    
    print(f"\n[1] Math module: {len(ml.nodes)} nodes")
    
    # Language training for word understanding
    lnn = ml  # Use math lattice as base
    
    LANGUAGE = [
        ("the sun is in the sky", "sky"),
        ("water flows down", "down"),
        ("the cat meows", "meows"),
        ("light travels fast", "fast"),
        ("fire causes burn", "burn"),
        ("friction creates heat", "heat"),
        ("gravity causes fall", "fall"),
        ("cold and hot are both temperature", "temperature"),
        ("fish and birds are both animals", "animals"),
    ]
    
    print(f"\n[2] Training language...")
    for sentence, expected in LANGUAGE:
        for _ in range(10):
            add_sentence(lnn, sentence)
    
    # Test language
    print(f"\n[3] Testing language...")
    
    LANG_TESTS = [
        ("the sun", "sky"),
        ("water flows", "down"),
        ("the cat", "meows"),
        ("fire causes", "burn"),
        ("gravity causes", "fall"),
    ]
    
    lang_passed = 0
    for prompt, expected in LANG_TESTS:
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=3)
        if candidates:
            top = candidates[0]["word"]
            match = top.lower() == expected.lower()
            status = "✓" if match else "✗"
            print(f"    {status} '{prompt}' -> {top}")
            if match:
                lang_passed += 1
    
    # Test math word problems
    print(f"\n[4] Testing math word problems...")
    
    WORD_PROBLEMS = [
        ("one and one are", 2),
        ("two and two are", 4),
        ("three and three are", 6),
        ("two times two is", 4),
        ("three times two is", 6),
        ("four times three is", 12),
    ]
    
    math_passed = 0
    for prompt, expected in WORD_PROBLEMS:
        ml.reset()
        result = solve_word_problem(nl, ml, prompt)
        if result is not None:
            match = result == expected
            status = "✓" if match else "✗"
            print(f"    {status} '{prompt}' -> {result} (expected: {expected})")
            if match:
                math_passed += 1
        else:
            print(f"    ✗ '{prompt}' -> couldn't parse")
    
    # Combined test
    print(f"\n[5] Combined: Language + Math...")
    
    COMBINED = [
        ("the cat meows and", "meows"),
        ("water flows and two plus two is", "four"),
    ]
    
    for prompt, expected in COMBINED:
        tokens = prompt.lower().split()
        candidates = generate(lnn, tokens, top_k=3)
        if candidates:
            top = candidates[0]["word"]
            match = top.lower() == expected.lower()
            status = "✓" if match else "✗"
            print(f"    {status} '{prompt}' -> {top}")
    
    print(f"\n[Results] Language: {lang_passed}/{len(LANG_TESTS)}, Math: {math_passed}/{len(WORD_PROBLEMS)}")
    
    with open("/Users/tyarc/github/lrn/sys_test/math_word_problems.json", 'w') as f:
        json.dump({
            "language": {"passed": lang_passed, "total": len(LANG_TESTS)},
            "math_word_problems": {"passed": math_passed, "total": len(WORD_PROBLEMS)},
            "network": {"nodes": len(ml.nodes), "springs": len(ml.springs)}
        }, f, indent=2)


if __name__ == "__main__":
    main()