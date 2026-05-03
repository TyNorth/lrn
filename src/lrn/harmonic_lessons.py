"""
Harmonic Video Lessons per curriculum level.

Every level gets harmonic video labeling — the content changes but the
modality is always present. This is how children learn: they watch
educational content throughout their lives.

Each lesson attaches WORDS to concepts the lattice already knows through
sensory grounding, using 4 phase-aligned modalities:
- visual: word appears on screen
- audio: word is spoken
- emotional: feeling matches concept
- rhythm: syllable timing
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn.harmonic_video import make_label_lesson, HarmonicLesson, ModalityStream, harmonic_convergence


def get_harmonic_lessons(level):
    """Return harmonic video lessons appropriate for this curriculum level."""
    if level == "prek":
        return get_prek_lessons()
    elif level == "kindergarten":
        return get_kindergarten_lessons()
    elif level == "first_grade":
        return get_first_grade_lessons()
    return []


def get_prek_lessons():
    """Pre-K harmonic video lessons — basic concept labeling."""
    return [
        make_label_lesson(
            ["hot", "warm", "cool", "cold", "freezing", "boiling"],
            "Temperature Words"
        ),
        make_label_lesson(
            ["heavy", "light", "big", "small", "tiny"],
            "Weight & Size Words"
        ),
        make_label_lesson(
            ["fast", "slow", "quick", "crawl", "run", "walk"],
            "Speed Words"
        ),
        make_label_lesson(
            ["near", "far", "close", "away", "here", "there"],
            "Distance Words"
        ),
        make_label_lesson(
            ["bright", "dim", "dark", "light", "shine", "glow", "shadow"],
            "Brightness Words"
        ),
        make_label_lesson(
            ["loud", "quiet", "high", "low", "noisy", "whisper", "music"],
            "Sound Words"
        ),
        make_label_lesson(
            ["soft", "hard", "smooth", "rough", "sticky", "wet", "fuzzy"],
            "Texture Words"
        ),
        make_label_lesson(
            ["sweet", "sour", "salty", "bitter", "savory", "yummy", "yucky"],
            "Taste Words"
        ),
        make_label_lesson(
            ["fragrant", "stinky", "fresh", "rotten", "smoky", "spicy"],
            "Smell Words"
        ),
        make_label_lesson(
            ["happy", "sad", "angry", "scared", "surprised", "tired", "excited", "calm"],
            "Emotion Words"
        ),
    ]


def get_kindergarten_lessons():
    """Kindergarten harmonic video lessons — academic concept labeling.
    
    The lattice already knows:
    - numbers as quantity waves (from Pre-K)
    - letters as visual geometry (from Pre-K)
    - colors as EM spectrum (from Pre-K)
    - basic concepts (hot/cold, big/small, etc.)
    
    Now we attach academic labels:
    - Math operations: plus, minus, equals, add, subtract
    - Reading: read, write, spell, word, sentence, story
    - Science: grow, seed, plant, habitat, cycle
    - Geography: map, north, south, east, west
    - Time: morning, afternoon, night, today, tomorrow
    - Social: citizen, vote, rule, community, recycle
    """
    return [
        make_label_lesson(
            ["plus", "minus", "equals", "add", "subtract", "sum", "difference"],
            "Math Operation Words"
        ),
        make_label_lesson(
            ["read", "write", "spell", "word", "sentence", "story", "book", "page"],
            "Reading & Writing Words"
        ),
        make_label_lesson(
            ["grow", "seed", "plant", "root", "stem", "leaf", "flower", "tree"],
            "Plant Growth Words"
        ),
        make_label_lesson(
            ["habitat", "forest", "ocean", "desert", "arctic", "jungle", "shelter"],
            "Habitat Words"
        ),
        make_label_lesson(
            ["cycle", "begin", "middle", "end", "first", "next", "last", "repeat"],
            "Sequence & Cycle Words"
        ),
        make_label_lesson(
            ["map", "globe", "north", "south", "east", "west", "continent", "ocean"],
            "Geography Words"
        ),
        make_label_lesson(
            ["citizen", "vote", "rule", "community", "recycle", "save", "leader", "flag"],
            "Citizenship Words"
        ),
        make_label_lesson(
            ["morning", "afternoon", "night", "today", "tomorrow", "yesterday", "week", "month"],
            "Time Words"
        ),
        make_label_lesson(
            ["enormous", "tiny", "rapid", "ancient", "furious", "delighted", "exhausted", "frightened"],
            "Vocabulary Expansion Words"
        ),
        make_label_lesson(
            ["synonym", "antonym", "same", "opposite", "meaning", "define", "describe"],
            "Language Concept Words"
        ),
        make_label_lesson(
            ["graph", "tally", "survey", "count", "data", "bar", "chart", "compare"],
            "Data & Graph Words"
        ),
        make_label_lesson(
            ["cube", "sphere", "cylinder", "cone", "face", "edge", "corner", "round"],
            "3D Shape Words"
        ),
    ]


def get_first_grade_lessons():
    """1st Grade harmonic video lessons — advanced concept labeling.
    
    Builds on K foundation with:
    - Advanced math: dozen, half, quarter, fraction, measure
    - Advanced reading: chapter, paragraph, fiction, nonfiction, author
    - Science: experiment, observe, predict, result, conclusion
    - Social studies: history, past, present, future, tradition, culture
    - Health: exercise, nutrition, hygiene, germ, medicine
    """
    return [
        make_label_lesson(
            ["dozen", "half", "quarter", "fraction", "measure", "inch", "foot", "pound"],
            "Advanced Math Words"
        ),
        make_label_lesson(
            ["chapter", "paragraph", "fiction", "nonfiction", "author", "illustrator", "title"],
            "Advanced Reading Words"
        ),
        make_label_lesson(
            ["experiment", "observe", "predict", "result", "conclusion", "hypothesis", "test"],
            "Science Method Words"
        ),
        make_label_lesson(
            ["history", "past", "present", "future", "tradition", "culture", "generation", "ancestor"],
            "History & Culture Words"
        ),
        make_label_lesson(
            ["exercise", "nutrition", "hygiene", "germ", "medicine", "healthy", "sick", "doctor"],
            "Health Words"
        ),
        make_label_lesson(
            ["because", "therefore", "however", "although", "since", "unless", "while", "until"],
            "Complex Connector Words"
        ),
        make_label_lesson(
            ["compare", "contrast", "similar", "different", "alike", "unique", "pattern", "sequence"],
            "Analysis Words"
        ),
        make_label_lesson(
            ["responsibility", "respect", "honesty", "courage", "kindness", "patience", "perseverance"],
            "Character Trait Words"
        ),
    ]


def harmonic_video_training(lnn, level, verbose=True):
    """Run harmonic video labeling for a curriculum level."""
    lessons = get_harmonic_lessons(level)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"HARMONIC VIDEO LABELING: {level.upper()}")
        print(f"{'='*60}")
        print(f"  Lessons: {len(lessons)}")
        print()
    
    for i, lesson in enumerate(lessons):
        if verbose:
            print(f"  Lesson {i+1}/{len(lessons)}: {lesson.name}")
        harmonic_convergence(lnn, lesson, verbose=False)
    
    from lrn import propagate
    propagate(lnn, n_steps=5)
    
    if verbose:
        print(f"\n  After harmonic video: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
        print()
    
    return lnn
