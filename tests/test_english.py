"""
Focus: English Language - Fluent Polyglot Phase 1
Test English vocabulary growth with incremental targets.
Grades: 250, 500, 1000, 2000, 3500, 5000
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, discover_words, get_language_samples, get_grade
from lrn.natural_tokenize import learn_from_text


GRADE_TARGETS = [
    (250, "grade1"),
    (500, "grade2"),
    (1000, "grade3"),
    (2000, "grade4"),
    (3500, "grade5"),
    (5000, "grade6"),
]


def train_english(target_words: int, max_reps: int = 500) -> dict:
    """Train English until target words reached."""
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    samples = get_language_samples("english")
    min_freq = 2
    
    words = 0
    reps = 0
    
    # Increase batch size for faster training
    batch_size = 200
    
    while words < target_words and reps < max_reps:
        # Train a batch
        batch = samples[:batch_size]
        for _ in range(10):
            for text in batch:
                learn_from_text(lnn, text, repetitions=1)
        
        reps += 10
        
        # Check word count
        words = len(discover_words(lnn, batch, min_frequency=min_freq))
        
        if reps % 50 == 0:
            print(f"  {reps} reps: {words} words")
    
    return {"words": words, "reps": reps, "grade": get_grade(words)}


def main():
    print("=" * 60)
    print("ENGLISH LANGUAGE - INCREMENTAL TEST")
    print("=" * 60)
    
    results = {}
    
    for target, grade_name in GRADE_TARGETS:
        print(f"\n--- Target: {target} words ({grade_name}) ---")
        result = train_english(target)
        results[grade_name] = result
        
        status = "PASS" if result["words"] >= target else "FAIL"
        print(f"Result: {result['words']}/{target} words ({result['grade']}) - {status}")
        
        if result["words"] < target:
            print(f"Stopped at {result['reps']} reps - not enough")
            break
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for grade, result in results.items():
        print(f"  {grade}: {result['words']} words ({result['reps']} reps)")
    
    # Final test
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    print("\n--- FINAL COMPREHENSIVE TEST ---")
    samples = get_language_samples("english")
    
    # More intensive training
    for _ in range(100):
        for text in samples[:500]:
            learn_from_text(lnn, text, repetitions=1)
    
    words = discover_words(lnn, samples[:500], min_frequency=2)
    final_grade = get_grade(len(words))
    print(f"Final: {len(words)} words - {final_grade}")
    
    return len(words), final_grade


if __name__ == "__main__":
    words, grade = main()
    print(f"\nFINAL: {words} words, grade: {grade}")