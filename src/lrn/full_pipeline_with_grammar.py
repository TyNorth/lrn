"""
Full Grade Pipeline with Grammar Spiral - From Pre-K to 3rd Grade
Trains the full pipeline and tests grammar generation at each level.
"""
import sys
import os
import pickle
import time

sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate, assign_roles, optimal_rem, prune_springs
from lrn.fast_trainer import train_fast
from lrn.grammar_lessons import get_grammar_lessons
from lrn.grammar_lessons_2nd import get_grammar_lessons as get_2nd_lessons
from lrn.grammar_lessons_3rd import get_grammar_lessons as get_3rd_lessons


CHECKPOINT_DIR = '/Users/tyarc/github/lrn/checkpoints/iterative'
REPORT_DIR = '/Users/tyarc/github/lrn/reports/iterative'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def load_pretrained(prefix="pre_k"):
    """Load pre-trained lattice from disk."""
    path = f"{CHECKPOINT_DIR}/lattice_{prefix}.pkl"
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        lnn = LatticeNN()
        lnn.nodes = data['nodes']
        lnn.springs = data['springs']
        lnn.trigrams = data.get('trigrams', {})
        print(f"Loaded {prefix}: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
        return lnn
    return None


def save_checkpoint(lnn, prefix):
    """Save lattice checkpoint."""
    path = f"{CHECKPOINT_DIR}/lattice_{prefix}.pkl"
    with open(path, 'wb') as f:
        pickle.dump({
            'nodes': lnn.nodes,
            'springs': lnn.springs,
            'trigrams': lnn.trigrams,
        }, f)
    print(f"Saved: {path}")


def train_grammar_lessons(lnn, grade, full=False):
    """Train on grammar lesson sentences."""
    if grade == 1:
        lessons = get_grammar_lessons(grade, full)
    elif grade == 2:
        lessons = get_2nd_lessons(grade, full)
    elif grade == 3:
        lessons = get_3rd_lessons(grade, full)
    else:
        return 0

    sentences = []
    for lesson in lessons:
        sentences.extend(lesson.get("sentences", []))

    print(f"  Training {len(sentences)} grammar sentences (grade {grade})")
    train_fast(lnn, sentences, reps=15, verbose=False)
    return len(sentences)


def generate_from_grammar(lnn, prompt, grade, num_tokens=8):
    """Generate using grammar-trained lattice."""
    from lrn.generate import generate_sequence

    tokens = generate_sequence(lnn, prompt, n_tokens=num_tokens, temperature=0.8)
    return " ".join(tokens)


def test_grammar_generation(lnn, grade):
    """Test grammar generation at a grade level."""
    print(f"\n  === GRADE {grade} GRAMMAR GENERATION TEST ===")

    if grade == 1:
        lessons = get_grammar_lessons(grade, False)
        test_prompts = [
            "the dog",
            "a cat",
            "i see",
            "she runs",
            "the sun",
        ]
    elif grade == 2:
        lessons = get_2nd_lessons(grade, False)
        test_prompts = [
            "the girl runs",
            "the bird builds",
            "he walked",
            "she eats",
            "they play",
        ]
    else:
        lessons = get_3rd_lessons(grade, False)
        test_prompts = [
            "he walks",
            "she is walking",
            "they were reading",
            "the dogs run",
            "it's barking",
        ]

    print(f"  Testing {len(test_prompts)} prompts:")
    for prompt in test_prompts:
        result = generate_from_grammar(lnn, prompt, grade, num_tokens=5)
        print(f"    '{prompt}' -> '{result}'")

    baseline_count = min(10, len(lessons))
    baseline_test = lessons[0] if lessons else None
    if baseline_test:
        concept = baseline_test.get("concept", "")
        sentences = baseline_test.get("sentences", [])
        print(f"\n  Lesson: {baseline_test['title']}")
        print(f"  Concept: {concept}")
        print(f"  Sentences: {sentences[:3]}...")

    return True


def print_assessment(results, title):
    """Print assessment results."""
    print(f"\n  {title}")
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
        skill_name = result.get('skill', skill)
        print(f"  {skill_name:40s} {score:.0f}/{possible:.0f} {pct:3d}% {bar}")
    mastery_pct = int((total_score * 100) // total_possible) if total_possible > 0 else 0
    print("  " + "-" * 55)
    print(f"  {'TOTAL':40s} {total_score:.0f}/{total_possible:.0f} {mastery_pct:3d}%")
    return mastery_pct


def run_grade_assessment(lnn, grade):
    """Run assessment for a grade."""
    print(f"\n  === GRADE {grade} ASSESSMENT ===")

    if grade == 1:
        from lrn.assessor_1st import assess_first_grade
        results = assess_first_grade(lnn)
    elif grade == 2:
        from lrn.assessor_2nd import assess_second_grade
        results = assess_second_grade(lnn)
    elif grade == 3:
        from lrn.assessor_3rd import assess_third_grade
        results = assess_third_grade(lnn)
    else:
        return None

    mastery = print_assessment(results, "ASSESSMENT RESULTS")
    return mastery


def main():
    print("=" * 70)
    print("FULL GRADE PIPELINE WITH GRAMMAR SPIRAL - Pre-K to 3rd Grade")
    print("=" * 70)

    start_time = time.time()

    # === PRE-K LATTICE (already trained) ===
    print("\n[1] LOADING PRE-K LATTICE...")
    lnn = load_pretrained("pre_k")
    if lnn is None:
        print("ERROR: Pre-K lattice not found. Run pre_k_pipeline.py first.")
        return

    print(f"  Pre-K: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")

    # === KINDERGARTEN ===
    print("\n[2] TRAINING KINDERGARTEN...")
    k_sentences = [
        "the boy runs fast", "the girl jumps high", "a cat sleeps", "a dog barks",
        "i see a bird", "the sun shines", "the moon glows", "stars twinkle",
        "water is wet", "fire is hot", "trees grow tall", "flowers bloom bright",
        "mom cooks dinner", "dad works hard", "baby cries loud", "kids play games",
        "red ball bounces", "blue sky stretches", "green grass grows", "yellow sun shines",
        "big dog runs", "small cat sits", "fast car drives", "slow turtle walks",
        "one apple", "two books", "three cars", "four dogs", "five fish",
        "a is for apple", "b is for ball", "c is for cat", "d is for dog",
        "i can run", "i can jump", "i can read", "i can write",
        "the cat sits on the mat", "the dog runs in the park", "the bird flies in the sky",
    ]
    train_fast(lnn, k_sentences, reps=20, verbose=True)
    save_checkpoint(lnn, "k")
    print(f"  K total: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")

    # === 1ST GRADE (with grammar) ===
    print("\n[3] TRAINING 1ST GRADE + GRAMMAR...")
    first_sentences = [
        "the dog barks loud", "the cat catches mouse", "the bird builds nest",
        "a boy throws ball", "a girl reads book", "friends play together",
        "we go to school", "they walk to park", "she runs fast", "he jumps high",
        "big dog runs fast", "small cat sleeps sound", "red ball bounces high",
        "the sun shines bright", "the rain falls down", "snow falls soft",
        "i walk to school", "you run fast", "we play games", "they smile big",
        "mom cooks dinner", "dad fixes car", "teacher helps students", "doctor checks patients",
        "bird flies", "fish swim", "frog jumps", "bear walks", "bee flies",
        "book sits on desk", "cup holds water", "clock ticks loud", "pen writes smooth",
        "paper folds easy", "door opens wide", "window lets light", "table holds plates",
        "the ball bounces", "the chair has legs", "the pen writes", "the book reads",
    ]
    train_fast(lnn, first_sentences, reps=15, verbose=True)
    train_grammar_lessons(lnn, grade=1, full=False)
    save_checkpoint(lnn, "1st")
    print(f"  1st total: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    test_grammar_generation(lnn, grade=1)

    # === 2ND GRADE (with grammar) ===
    print("\n[4] TRAINING 2ND GRADE + GRAMMAR...")
    second_sentences = [
        "the girl runs fast", "the dog catches ball", "the bird sings beautiful",
        "the cat purrs soft", "the dog wags tail", "the bear catches fish",
        "the frog jumps high", "the fish swims deep", "the bee collects pollen",
        "the snake sheds skin", "the boy walks to school", "the girl plays soccer",
        "city is general but new york city is proper", "state is general but california is proper",
        "river is general but amazon river is proper", "country is general but brazil is proper",
        "ocean is general but pacific ocean is proper", "teacher is common but mrs smith is proper",
        "friend is common but emily is proper", "school is common but lincoln elementary is proper",
        "walk becomes walked", "play becomes played", "jump becomes jumped",
        "talk becomes talked", "learn becomes learned", "help becomes helped",
        "go becomes went", "see becomes saw", "come becomes came", "eat becomes ate",
        "i walk to school", "she walks to school", "they play in the park", "he plays soccer",
        "it barks loud", "the bird sings", "she catches the ball", "he watches the game",
        "i will go tomorrow", "she will come later", "they will play outside",
    ]
    train_fast(lnn, second_sentences, reps=12, verbose=True)
    train_grammar_lessons(lnn, grade=2, full=False)
    save_checkpoint(lnn, "2nd")
    print(f"  2nd total: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    test_grammar_generation(lnn, grade=2)

    # === 3RD GRADE (with grammar) ===
    print("\n[5] TRAINING 3RD GRADE + GRAMMAR...")
    third_sentences = [
        "he walks to school every day", "she walks to school every day", "they walk to school every day",
        "we walk to school every day", "the dog walks on a leash", "the dogs walk in the park",
        "she reads books every week", "they read books every week", "it barks loudly", "they bark loudly",
        "he walked to school yesterday", "she walked to school yesterday", "they walked to school yesterday",
        "we walked to school yesterday", "the dog walked on a leash", "the dogs walked in the park",
        "she read books last week", "they read books last week", "it barked loudly", "they barked loudly",
        "i was walking to school", "he was walking to school", "she was walking to school",
        "you were walking to school", "we were walking to school", "they were walking to school",
        "he is walking to school", "she is walking to school", "it is running fast", "they are walking to school",
        "we are walking to school", "you are walking to school", "the dog is barking loud",
        "the birds are singing", "the children are playing",
    ]
    train_fast(lnn, third_sentences, reps=12, verbose=True)
    train_grammar_lessons(lnn, grade=3, full=False)
    save_checkpoint(lnn, "3rd")
    print(f"  3rd total: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    test_grammar_generation(lnn, grade=3)

    # === FINAL SUMMARY ===
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Final: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs, {len(lnn.trigrams)} trigrams")
    print(f"\n  Checkpoints saved in: {CHECKPOINT_DIR}")
    print("\n  Testing generation quality...")

    from lrn.generate import generate_sequence

    test_prompts = [
        "quantum mechanics",
        "photosynthesis in plants",
        "democracy in ancient greece",
    ]
    print("\n  === GENERATION QUALITY TEST ===")
    for prompt in test_prompts:
        tokens = generate_sequence(lnn, prompt, n_tokens=6, temperature=0.7)
        print(f"    '{prompt}' -> '{' '.join(tokens)}'")


if __name__ == "__main__":
    main()