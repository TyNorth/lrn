#!/usr/bin/env python3
"""Math + Language Combined with Word-to-Number Bridge"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_sentence, generate
from lrn.math_lattice import MathLattice, NumberLine


WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20
}


def word_to_num(word: str) -> int:
    """Convert word to number"""
    return WORD_TO_NUM.get(word.lower(), None)


def solve_word_math(prompt: str, nl, ml) -> tuple:
    """Solve math word problem with number line"""
    prompt = prompt.lower()
    
    # Addition: "one and one" or "one plus one"
    if " and " in prompt or " plus " in prompt:
        words = prompt.replace(" and ", " ").replace(" plus ", " ").split()
        nums = [word_to_num(w) for w in words if word_to_num(w)]
        if len(nums) >= 2:
            ml.reset()
            result = nl.step_forward(nums[0], nums[1], verbose=False)
            return result, f"{nums[0]} + {nums[1]} = {result}"
    
    # Multiplication: "two times two"
    if " times " in prompt:
        words = prompt.split(" times ")
        a = word_to_num(words[0].split()[-1])
        b = word_to_num(words[1].split()[0])
        if a and b:
            ml.reset()
            result = nl.multiply(a, b, verbose=False)
            return result, f"{a} × {b} = {result}"
    
    return None, "not a math problem"


def main():
    print("=" * 60)
    print("Math + Language Combined")
    print("=" * 60)
    
    # Language model
    lnn = LatticeNN()
    
    LANGUAGE = [
        ("the sun is in the sky", "sky"),
        ("water flows down", "down"),
        ("the cat meows", "meows"),
        ("fire causes burn", "burn"),
        ("friction creates heat", "heat"),
        ("gravity causes fall", "fall"),
        ("cold and hot are both temperature", "temperature"),
    ]
    
    print(f"\n[1] Training language...")
    for sentence, expected in LANGUAGE:
        for _ in range(10):
            add_sentence(lnn, sentence)
    
    print(f"    Language: {len(lnn.nodes)} nodes")
    
    # Math module
    ml = MathLattice()
    ml.initialize_math()
    nl = NumberLine(ml)
    
    print(f"    Math: {len(ml.nodes)} nodes")
    
    # Test language
    print(f"\n[2] Language tests...")
    
    LANG_TESTS = [
        ("the sun", "sky"),
        ("water flows", "down"),
        ("the cat", "meows"),
        ("fire causes", "burn"),
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
    
    # Test math
    print(f"\n[3] Math word problems...")
    
    MATH_TESTS = [
        "one and one",
        "two and two", 
        "two times two",
        "three times three",
        "four and three",
    ]
    
    math_passed = 0
    expected_results = [2, 4, 4, 9, 7]
    
    for prompt, expected in zip(MATH_TESTS, expected_results):
        result, msg = solve_word_math(prompt, nl, ml)
        if result == expected:
            print(f"    ✓ {prompt} = {result}")
            math_passed += 1
        else:
            print(f"    ✗ {prompt} = {result} (expected {expected})")
    
    print(f"\n[Results] Language: {lang_passed}/{len(LANG_TESTS)}, Math: {math_passed}/{len(MATH_TESTS)}")
    print(f"          Combined: {lang_passed + math_passed}/{len(LANG_TESTS) + len(MATH_TESTS)}")
    
    import json
    with open("/Users/tyarc/github/lrn/sys_test/math_language_combined.json", 'w') as f:
        json.dump({
            "language": {"passed": lang_passed, "total": len(LANG_TESTS)},
            "math": {"passed": math_passed, "total": len(MATH_TESTS)},
            "combined": lang_passed + math_passed,
            "total": len(LANG_TESTS) + len(MATH_TESTS)
        }, f, indent=2)


if __name__ == "__main__":
    main()