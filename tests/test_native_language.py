"""
Quick Test: Native Language Training - Fluent Polyglot
Fast verification test - checks if language learning works at basic level.
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import (
    LatticeNN, add_identity_anchor,
    train_language_native, train_all_languages_native,
    discover_words, get_language_samples, get_mixed_samples,
    get_grade,
)


def test_quick_language() -> dict:
    """Quick test - train all 3 languages + mixed."""
    print("=" * 70)
    print("NATIVE LANGUAGE QUICK TEST")
    print("=" * 70)
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    # Use fewer repetitions for quick test
    stats = train_all_languages_native(lnn, repetitions_per_lang=30)
    
    print(f"\nLanguages trained: {stats['languages']}")
    print(f"Total samples: {stats['samples']}")
    print(f"Total springs: {stats['springs']}")
    
    print("\n--- VOCABULARY RESULTS ---")
    for lang in ["english", "spanish", "mandarin"]:
        words = stats.get("vocabulary", {}).get(lang, [])
        grade = stats.get("grades", {}).get(lang, "grade0")
        print(f"  {lang}: {len(words)} words - {grade}")
    
    print("\n--- MIXED LANGUAGE ---")
    mixed_vocab = stats.get("mixed_vocabulary", {})
    for mix_type, words in mixed_vocab.items():
        print(f"  {mix_type}: {len(words)} words - {words[:5]}")
    
    # Test criterion: At least 50 words per language to pass
    print("\n--- GRADE CHECK ---")
    passed = 0
    for lang in ["english", "spanish", "mandarin"]:
        words = stats.get("vocabulary", {}).get(lang, [])
        word_count = len(words)
        
        if word_count >= 50:
            print(f"  {lang}: {word_count}/50 - PASS")
            passed += 1
        else:
            print(f"  {lang}: {word_count}/50 - FAIL")
    
    print(f"\nResult: {passed}/3 languages passed")
    
    return stats


def test_language_increment() -> dict:
    """Test incremental training - show progress at different levels."""
    print("\n" + "=" * 70)
    print("INCREMENTAL LANGUAGE TEST")
    print("=" * 70)
    
    results = {}
    
    # Test each language with increasing repetitions
    for lang in ["english", "spanish", "mandarin"]:
        print(f"\n--- {lang.upper()} ---")
        
        for reps in [20, 40, 60]:
            lnn = LatticeNN()
            add_identity_anchor(lnn)
            
            samples = get_language_samples(lang)
            sample_size = min(len(samples), 500)
            
            from lrn.natural_tokenize import learn_from_text
            for _ in range(reps):
                for text in samples[:sample_size]:
                    learn_from_text(lnn, text, repetitions=1)
            
            words = discover_words(lnn, samples[:sample_size], min_frequency=2)
            word_count = len(words)
            grade = get_grade(word_count)
            
            print(f"  {reps} reps: {word_count} words ({grade})")
            
            if reps == 60:
                results[lang] = {"words": word_count, "grade": grade}
    
    return results


def test_mixed_quick() -> dict:
    """Quick test for mixed languages."""
    print("\n" + "=" * 70)
    print("MIXED LANGUAGE TEST (Spanglish, Chinglish)")
    print("=" * 70)
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    from lrn.natural_tokenize import learn_from_text
    
    mixed_samples = get_mixed_samples()
    total_samples = 0
    
    for mix_type, texts in mixed_samples.items():
        for _ in range(20):
            for text in texts:
                learn_from_text(lnn, text, repetitions=1)
                total_samples += 1
    
    print(f"Trained on {total_samples} mixed language samples")
    
    # Check vocabulary for each type
    print("\nVocabulary discovered:")
    for mix_type, texts in mixed_samples.items():
        words = discover_words(lnn, texts, min_frequency=2)
        print(f"  {mix_type}: {len(words)} words - {list(words.keys())[:5]}")
    
    return {"samples": total_samples}


def main():
    """Run quick tests."""
    # Test 1: Full polyglot
    polyglot = test_quick_language()
    
    # Test 2: Incremental training
    incremental = test_language_increment()
    
    # Test 3: Mixed language
    mixed = test_mixed_quick()
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
    return 1


if __name__ == "__main__":
    main()