"""
Grammar Assessment Module
Tests grammar mastery based on grammar workbook lessons.
Can be used standalone or integrated into grade assessors.

Usage:
    from lrn.grammar_assessor import GrammarAssessor, assess_grammar
    results = assess_grammar(lnn, grade=2)
"""

import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn.grammar_workbook import GrammarWorkbook


def _has_spring(lnn, a, b):
    if a not in lnn.nodes or b not in lnn.nodes:
        return False
    key = lnn._key(a, b)
    return key in lnn.springs


def _node_exists(lnn, word):
    return f"word:{word}" in lnn.nodes


def _connection_count(lnn, word):
    node = f"word:{word}"
    if node not in lnn.nodes:
        return 0
    return len(lnn.get_neighbors(node))


def _test_concept(lnn, concept_name, words):
    """Test if a grammar concept has been learned."""
    items = []
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 2 else "PARTIAL", f"{count} connections"))
        else:
            items.append((word, "FAIL", "no node"))
    return items


def _make_result(items, domain, skill):
    score = sum(1 for _, status, _ in items if status == "PASS")
    partial = sum(1 for _, status, _ in items if status == "PARTIAL")
    possible = len(items)
    return {
        "score": score + partial * 0.5,
        "possible": possible,
        "pct": int((score + partial * 0.5) * 100 // possible) if possible > 0 else 0,
        "items": items,
        "domain": domain,
        "skill": skill,
    }


def test_nouns(lnn):
    words = ["boy", "girl", "teacher", "dog", "cat", "park", "school", "book", "table", "city"]
    items = _test_concept(lnn, "nouns", words)
    return _make_result(items, "Grammar", "Nouns (People, Places, Things, Animals)")


def test_verbs(lnn):
    words = ["runs", "walks", "jumps", "eats", "sleeps", "reads", "writes", "plays", "sings", "dances"]
    items = _test_concept(lnn, "verbs", words)
    return _make_result(items, "Grammar", "Verbs (Action Words)")


def test_adjectives(lnn):
    words = ["big", "small", "red", "blue", "happy", "sad", "tall", "short", "soft", "hard"]
    items = _test_concept(lnn, "adjectives", words)
    return _make_result(items, "Grammar", "Adjectives (Describing Words)")


def test_adverbs(lnn):
    words = ["quickly", "slowly", "happily", "sadly", "carefully", "loudly", "softly", "fast", "well", "brightly"]
    items = _test_concept(lnn, "adverbs", words)
    return _make_result(items, "Grammar", "Adverbs (How, When, Where)")


def test_pronouns(lnn):
    words = ["he", "she", "it", "they", "we", "you", "i", "him", "her", "them", "his", "her", "its", "their", "our", "your", "my"]
    items = _test_concept(lnn, "pronouns", words)
    return _make_result(items, "Grammar", "Pronouns")


def test_prepositions(lnn):
    words = ["in", "on", "at", "to", "from", "under", "over", "between", "behind", "in front of"]
    items = _test_concept(lnn, "prepositions", words)
    return _make_result(items, "Grammar", "Prepositions")


def test_conjunctions(lnn):
    words = ["and", "but", "or", "because", "although", "when", "if", "unless", "since", "until"]
    items = _test_concept(lnn, "conjunctions", words)
    return _make_result(items, "Grammar", "Conjunctions")


def test_sentence_structure(lnn):
    """Test basic sentence structure words."""
    words = ["subject", "predicate", "noun", "verb", "complete", "sentence", "fragment", "subject"]
    items = _test_concept(lnn, "sentence structure", words)
    return _make_result(items, "Grammar", "Sentence Structure")


def test_capitalization(lnn):
    """Test capitalization concepts."""
    words = ["capitalize", "uppercase", "begin", "sentence", "proper", "name", "monday", "january", "america"]
    items = _test_concept(lnn, "capitalization", words)
    return _make_result(items, "Grammar", "Capitalization")


def test_punctuation(lnn):
    """Test punctuation concepts."""
    words = ["period", "question", "exclamation", "comma", "apostrophe", "quotation", "mark", "end"]
    items = _test_concept(lnn, "punctuation", words)
    return _make_result(items, "Grammar", "Punctuation")


def test_contractions(lnn):
    """Test contraction concepts."""
    words = ["isn't", "aren't", "don't", "doesn't", "wasn't", "weren't", "can't", "won't", "i'm", "you're", "he's", "she's", "it's", "we're", "they're", "i've", "we've", "they've", "i'll", "we'll", "they'll"]
    items = []
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 1 else "PARTIAL", f"{count}"))
        else:
            clean = word.replace("'", "").replace("'", "")
            if _node_exists(lnn, clean):
                count = _connection_count(lnn, clean)
                items.append((word, "PARTIAL", f"exists as '{clean}'"))
            else:
                items.append((word, "FAIL", "no node"))
    return _make_result(items, "Grammar", "Contractions")


def test_subject_verb_agreement(lnn):
    """Test subject-verb agreement concepts."""
    words = ["subject", "verb", "agreement", "singular", "plural", "walks", "walk", "runs", "run", "eats", "eat"]
    items = _test_concept(lnn, "subject-verb agreement", words)
    return _make_result(items, "Grammar", "Subject-Verb Agreement")


def test_verb_tenses(lnn):
    """Test verb tense concepts."""
    words = ["past", "present", "future", "walked", "walks", "will walk", "ran", "runs", "will run", "ate", "eats", "will eat"]
    items = []
    for word in words:
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 1 else "PARTIAL", f"{count}"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Grammar", "Verb Tenses")


def test_proper_nouns(lnn):
    """Test proper noun concepts."""
    words = ["monday", "tuesday", "wednesday", "january", "america", "boston", "mary", "john", "smith", "lincoln"]
    items = _test_concept(lnn, "proper nouns", words)
    return _make_result(items, "Grammar", "Proper Nouns")


def test_plural_nouns(lnn):
    """Test plural noun concepts."""
    words = ["dogs", "cats", "children", "books", "mice", "feet", "teeth", "people", "boys", "girls", "apples", "boxes"]
    items = _test_concept(lnn, "plural nouns", words)
    return _make_result(items, "Grammar", "Plural Nouns")


def test_possession(lnn):
    """Test possessive concepts."""
    words = ["boy's", "girl's", "dog's", "teacher's", "children's", "mary's", "dad's", "friend's", " book's"]
    items = []
    for word in words:
        clean = word.replace("'s", "").replace("'", "")
        if _node_exists(lnn, word):
            count = _connection_count(lnn, word)
            items.append((word, "PASS" if count >= 1 else "PARTIAL", f"{count}"))
        elif _node_exists(lnn, clean):
            count = _connection_count(lnn, clean)
            items.append((word, "PARTIAL", f"exists as '{clean}'"))
        else:
            items.append((word, "FAIL", "no node"))
    return _make_result(items, "Grammar", "Possession (Apostrophes)")


def assess_grammar(lnn, grade=2):
    """
    Run grammar assessment appropriate for the grade level.
    Returns dict of results by grammar skill.
    """
    results = {}

    results["nouns"] = test_nouns(lnn)
    results["verbs"] = test_verbs(lnn)
    results["adjectives"] = test_adjectives(lnn)

    if grade >= 2:
        results["pronouns"] = test_pronouns(lnn)
        results["prepositions"] = test_prepositions(lnn)
        results["conjunctions"] = test_conjunctions(lnn)
        results["punctuation"] = test_punctuation(lnn)
        results["proper_nouns"] = test_proper_nouns(lnn)
        results["plural_nouns"] = test_plural_nouns(lnn)

    if grade >= 3:
        results["adverbs"] = test_adverbs(lnn)
        results["capitalization"] = test_capitalization(lnn)
        results["contractions"] = test_contractions(lnn)
        results["subject_verb_agreement"] = test_subject_verb_agreement(lnn)
        results["verb_tenses"] = test_verb_tenses(lnn)
        results["possession"] = test_possession(lnn)

    return results


def assess_grammar_1st(lnn):
    return assess_grammar(lnn, grade=1)


def assess_grammar_2nd(lnn):
    return assess_grammar(lnn, grade=2)


def assess_grammar_3rd(lnn):
    return assess_grammar(lnn, grade=3)


def print_grammar_assessment(results):
    """Print grammar assessment results."""
    print("\n  GRAMMAR ASSESSMENT")
    print("  " + "-" * 55)
    total_score = 0
    total_possible = 0
    for skill, result in results.items():
        score = result.get("score", 0)
        possible = result.get("possible", 0)
        pct = result.get("pct", 0)
        total_score += score
        total_possible += possible
        bar = "#" * (pct // 5)
        print(f"  {result.get('skill', skill):40s} {score:.0f}/{possible:.0f} {pct:3d}% {bar}")
    mastery_pct = int((total_score * 100) // total_possible) if total_possible > 0 else 0
    print("  " + "-" * 55)
    print(f"  {'GRAMMAR TOTAL':40s} {total_score:.0f}/{total_possible:.0f} {mastery_pct:3d}%")
    return mastery_pct, total_score, total_possible


if __name__ == "__main__":
    import pickle
    import json

    CHECKPOINT_DIR = '/Users/tyarc/github/lrn/checkpoints/iterative'

    for grade in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"GRAMMAR ASSESSMENT: GRADE {grade}")
        print(f"{'='*60}")

        path = f'{CHECKPOINT_DIR}/lattice_college.pkl'
        with open(path, 'rb') as f:
            data = pickle.load(f)
        from lrn import LatticeNN
        lnn = LatticeNN()
        lnn.nodes = data['nodes']
        lnn.springs = data['springs']
        lnn.trigrams = data['trigrams']

        results = assess_grammar(lnn, grade=grade)
        mastery_pct, total_score, total_possible = print_grammar_assessment(results)

        print(f"\n  Mastery: {mastery_pct}%")