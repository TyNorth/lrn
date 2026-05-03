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
    elif level == "second_grade":
        return get_second_grade_lessons()
    elif level == "third_grade":
        return get_third_grade_lessons()
    elif level == "fourth_grade":
        return get_fourth_grade_lessons()
    elif level == "fifth_grade":
        return get_fifth_grade_lessons()
    elif level == "sixth_grade":
        return get_sixth_grade_lessons()
    elif level == "seventh_grade":
        return get_seventh_grade_lessons()
    elif level == "eighth_grade":
        return get_eighth_grade_lessons()
    elif level == "ninth_grade":
        return get_ninth_grade_lessons()
    elif level == "tenth_grade":
        return get_tenth_grade_lessons()
    elif level == "eleventh_grade":
        return get_eleventh_grade_lessons()
    elif level == "twelfth_grade":
        return get_twelfth_grade_lessons()
    elif level == "college":
        return get_college_lessons()
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


def get_second_grade_lessons():
    """2nd Grade harmonic video lessons — complex concept labeling.
    
    Builds on 1st grade foundation with:
    - Advanced math: regrouping, estimate, perimeter, area
    - Advanced reading: context, infer, summarize, analyze, evaluate
    - Science: ecosystem, habitat, pollinator, decomposer, engineer
    - Social studies: constitution, amendment, citizen, economy, supply
    - SEL: self-control, cooperation, empathy, fairness, gratitude
    """
    return [
        make_label_lesson(
            ["regroup", "estimate", "perimeter", "area", "product", "quotient", "remainder"],
            "Advanced Math Words"
        ),
        make_label_lesson(
            ["context", "infer", "summarize", "analyze", "evaluate", "interpret", "compare"],
            "Advanced Reading Words"
        ),
        make_label_lesson(
            ["ecosystem", "habitat", "pollinator", "decomposer", "food chain", "camouflage", "migrate"],
            "Ecosystem Words"
        ),
        make_label_lesson(
            ["engineer", "design", "test", "improve", "solution", "material", "property"],
            "Engineering Words"
        ),
        make_label_lesson(
            ["constitution", "amendment", "citizen", "freedom", "right", "responsibility", "vote"],
            "Citizenship Words"
        ),
        make_label_lesson(
            ["economy", "supply", "demand", "producer", "consumer", "goods", "service"],
            "Economics Words"
        ),
        make_label_lesson(
            ["self-control", "cooperation", "empathy", "fairness", "gratitude", "perseverance"],
            "Advanced SEL Words"
        ),
        make_label_lesson(
            ["noun", "verb", "adjective", "adverb", "conjunction", "interjection", "pronoun"],
            "Grammar Words"
        ),
        # Gap-filling lessons for reading comprehension
        make_label_lesson(
            ["character", "setting", "plot", "story", "brave", "describe", "problem", "solution"],
            "Story Elements Words"
        ),
        # Gap-filling lessons for math operations
        make_label_lesson(
            ["regroup", "carry", "borrow", "check", "answer", "opposite", "operation", "verify"],
            "Math Operation Words"
        ),
        # Gap-filling lessons for earth science
        make_label_lesson(
            ["ocean", "map", "rock", "earth", "land", "water", "mountain", "river"],
            "Earth Science Words"
        ),
        # Gap-filling lessons for history
        make_label_lesson(
            ["event", "historian", "timeline", "source", "past", "history", "important", "change"],
            "History Words"
        ),
        # Final gap fillers
        make_label_lesson(
            ["lesson", "moral", "learn", "fantasy", "realistic", "cannot", "could", "happen"],
            "Story Type Words"
        ),
    ]


def get_third_grade_lessons():
    """3rd Grade harmonic video lessons — advanced academic concept labeling.
    
    Builds on 2nd grade foundation with:
    - Math: multiply, divide, times, fraction, numerator, denominator, area, perimeter
    - Reading: theme, compare, contrast, context, fluency, expression
    - Science: force, motion, friction, magnet, gravity, climate, organism, inherit
    - Social studies: government, legislative, executive, judicial, constitution
    - SEL: awareness, management, resilience, integrity, tolerance, leadership
    """
    return [
        make_label_lesson(
            ["multiply", "divide", "times", "product", "quotient", "factor", "multiple"],
            "Multiplication & Division Words"
        ),
        make_label_lesson(
            ["fraction", "numerator", "denominator", "half", "third", "fourth", "equal"],
            "Fraction Words"
        ),
        make_label_lesson(
            ["area", "perimeter", "square", "rectangle", "length", "width", "measure"],
            "Area & Perimeter Words"
        ),
        make_label_lesson(
            ["theme", "compare", "contrast", "context", "fluency", "expression", "chapter"],
            "Advanced Reading Words"
        ),
        make_label_lesson(
            ["force", "motion", "friction", "magnet", "gravity", "push", "pull", "balanced"],
            "Forces & Motion Words"
        ),
        make_label_lesson(
            ["climate", "organism", "inherit", "trait", "fossil", "environment", "survive"],
            "Ecosystem & Heredity Words"
        ),
        make_label_lesson(
            ["government", "legislative", "executive", "judicial", "constitution", "congress", "president"],
            "Government Words"
        ),
        make_label_lesson(
            ["resource", "specialization", "trade", "import", "export", "entrepreneur", "profit"],
            "Economics Words"
        ),
        make_label_lesson(
            ["continent", "hemisphere", "equator", "latitude", "longitude", "physical", "political"],
            "Geography Words"
        ),
        make_label_lesson(
            ["awareness", "management", "resilience", "integrity", "tolerance", "leadership", "teamwork"],
            "Advanced SEL Words"
        ),
    ]


def get_fourth_grade_lessons():
    """4th Grade harmonic video lessons."""
    return [
        make_label_lesson(
            ["multiply", "divide", "product", "quotient", "remainder", "algorithm", "digit"],
            "Multi-Digit Arithmetic Words"
        ),
        make_label_lesson(
            ["fraction", "equivalent", "numerator", "denominator", "decimal", "hundredth", "tenth"],
            "Fraction & Decimal Words"
        ),
        make_label_lesson(
            ["area", "perimeter", "rectangle", "formula", "measurement", "protractor", "angle"],
            "Geometry & Measurement Words"
        ),
        make_label_lesson(
            ["energy", "transfer", "collision", "motion", "speed", "sound", "light", "circuit"],
            "Energy Words"
        ),
        make_label_lesson(
            ["wave", "amplitude", "wavelength", "reflect", "absorb", "digital", "encode", "pattern"],
            "Waves & Technology Words"
        ),
        make_label_lesson(
            ["rainfall", "erosion", "canyon", "valley", "delta", "map", "landform", "weathering"],
            "Earth Systems Words"
        ),
        make_label_lesson(
            ["colony", "colonist", "freedom", "mayflower", "democracy", "tax", "resist", "britain"],
            "Colonial America Words"
        ),
        make_label_lesson(
            ["constitution", "amendment", "right", "responsibility", "citizen", "branch", "legislative", "judicial"],
            "Civics Words"
        ),
    ]


def get_fifth_grade_lessons():
    """5th Grade harmonic video lessons."""
    return [
        make_label_lesson(
            ["expression", "parentheses", "bracket", "evaluate", "sequence", "pattern", "rule"],
            "Operations & Algebraic Thinking Words"
        ),
        make_label_lesson(
            ["place value", "exponent", "power", "ten", "decimal", "thousandth", "round"],
            "Place Value & Decimals Words"
        ),
        make_label_lesson(
            ["multiply fractions", "divide fractions", "reciprocal", "unit fraction", "partition"],
            "Fraction Operations Words"
        ),
        make_label_lesson(
            ["volume", "cube", "unit cube", "cubic", "three dimensional", "length", "width", "height"],
            "Volume Words"
        ),
        make_label_lesson(
            ["coordinate plane", "axis", "origin", "ordered pair", "quadrant", "graph", "plot"],
            "Coordinate Geometry Words"
        ),
        make_label_lesson(
            ["matter", "particle", "conservation", "mixture", "solution", "dissolve", "property"],
            "Matter & Interactions Words"
        ),
        make_label_lesson(
            ["ecosystem", "food web", "producer", "consumer", "decomposer", "balance", "interdependent"],
            "Ecosystems Words"
        ),
        make_label_lesson(
            ["revolution", "independence", "declaration", "patriot", "loyalist", "battle", "treaty"],
            "American Revolution Words"
        ),
        make_label_lesson(
            ["analyze", "evaluate", "synthesize", "cite", "evidence", "perspective", "bias"],
            "Critical Thinking Words"
        ),
    ]


def get_sixth_grade_lessons():
    return [
        make_label_lesson(["ratio", "rate", "unit rate", "proportional", "percent", "quantity"], "Ratios & Proportional Relationships"),
        make_label_lesson(["quotient", "dividend", "decimal", "factor", "multiple", "absolute value", "integer"], "Number System"),
        make_label_lesson(["exponent", "expression", "variable", "equation", "inequality", "equivalent"], "Expressions & Equations"),
        make_label_lesson(["polygon", "surface area", "net", "three dimensional", "vertex", "coordinate"], "Geometry"),
        make_label_lesson(["statistical", "distribution", "median", "mean", "deviation", "histogram", "variability"], "Statistics"),
        make_label_lesson(["plate tectonics", "fossil", "earthquake", "hazard", "water cycle", "geoscience"], "Earth Science"),
        make_label_lesson(["cell", "organism", "cellular respiration", "ecosystem", "biodiversity", "population"], "Life Science"),
        make_label_lesson(["mesopotamia", "egypt", "pharaoh", "democracy", "athens", "roman", "republic"], "Ancient Civilizations"),
    ]


def get_seventh_grade_lessons():
    return [
        make_label_lesson(["proportional", "constant", "proportionality", "unit rate", "scale drawing"], "Proportional Relationships"),
        make_label_lesson(["rational number", "integer", "fraction", "decimal", "operation", "inverse"], "Rational Number Operations"),
        make_label_lesson(["expression", "equation", "inequality", "variable", "coefficient", "term"], "Expressions & Equations"),
        make_label_lesson(["scale drawing", "triangle", "angle", "circle", "area", "circumference", "pi"], "Geometry"),
        make_label_lesson(["probability", "compound event", "sample space", "random", "simulation"], "Probability"),
        make_label_lesson(["ecosystem", "energy flow", "photosynthesis", "cellular respiration", "adaptation"], "Life Science MS"),
        make_label_lesson(["chemical reaction", "atom", "molecule", "element", "compound", "conservation"], "Physical Science MS"),
        make_label_lesson(["middle ages", "feudalism", "renaissance", "exploration", "reformation"], "World History MS"),
    ]


def get_eighth_grade_lessons():
    return [
        make_label_lesson(["irrational number", "square root", "cube root", "scientific notation", "exponent"], "Number System 8th"),
        make_label_lesson(["linear equation", "slope", "intercept", "function", "graph", "rate of change"], "Linear Equations & Functions"),
        make_label_lesson(["system of equations", "simultaneous", "solution", "substitution", "elimination"], "Systems of Equations"),
        make_label_lesson(["transformation", "congruence", "similarity", "pythagorean", "theorem", "volume"], "Geometry 8th"),
        make_label_lesson(["scatter plot", "correlation", "trend line", "bivariate", "association"], "Statistics 8th"),
        make_label_lesson(["force", "newton", "motion", "energy", "wave", "electromagnetic", "kinetic"], "Physical Science 8th"),
        make_label_lesson(["constitution", "amendment", "federalism", "separation", "powers", "bill of rights"], "Civics 8th"),
        make_label_lesson(["revolution", "industrial", "enlightenment", "imperialism", "nationalism"], "Modern World History"),
    ]


def get_ninth_grade_lessons():
    return [
        make_label_lesson(["quadratic", "polynomial", "factoring", "vertex", "parabola", "coefficient"], "Algebra I"),
        make_label_lesson(["biology", "cell", "dna", "genetics", "evolution", "natural selection", "mutation"], "Biology"),
        make_label_lesson(["world war", "treaty", "alliance", "conflict", "diplomacy", "sovereignty"], "World History"),
        make_label_lesson(["essay", "thesis", "argument", "rhetoric", "analysis", "synthesis", "citation"], "English 9"),
        make_label_lesson(["geography", "region", "culture", "migration", "urbanization", "globalization"], "Geography"),
    ]


def get_tenth_grade_lessons():
    return [
        make_label_lesson(["geometry", "proof", "theorem", "congruent", "similar", "trigonometry", "sine"], "Geometry"),
        make_label_lesson(["chemistry", "atom", "bond", "reaction", "mole", "stoichiometry", "periodic"], "Chemistry"),
        make_label_lesson(["american history", "constitution", "civil war", "reconstruction", "progressive"], "US History"),
        make_label_lesson(["literature", "theme", "symbolism", "irony", "narrative", "perspective", "genre"], "English 10"),
        make_label_lesson(["economics", "supply", "demand", "market", "government", "policy", "trade"], "Economics"),
    ]


def get_eleventh_grade_lessons():
    return [
        make_label_lesson(["algebra", "logarithm", "exponential", "trigonometric", "sequence", "series"], "Algebra II"),
        make_label_lesson(["physics", "mechanics", "thermodynamics", "electricity", "magnetism", "optics"], "Physics"),
        make_label_lesson(["government", "political", "party", "election", "campaign", "policy", "judicial"], "US Government"),
        make_label_lesson(["american literature", "transcendentalism", "modernism", "harlem renaissance", "contemporary"], "English 11"),
        make_label_lesson(["psychology", "behavior", "cognition", "development", "social", "personality"], "Psychology"),
    ]


def get_twelfth_grade_lessons():
    return [
        make_label_lesson(["calculus", "derivative", "integral", "limit", "continuity", "differentiation"], "Calculus"),
        make_label_lesson(["statistics", "distribution", "hypothesis", "regression", "correlation", "inference"], "Statistics"),
        make_label_lesson(["environmental science", "sustainability", "biodiversity", "climate change", "ecosystem"], "Environmental Science"),
        make_label_lesson(["philosophy", "ethics", "logic", "epistemology", "metaphysics", "existentialism"], "Philosophy"),
        make_label_lesson(["comparative government", "international relations", "diplomacy", "human rights", "global"], "Comparative Government"),
    ]


def get_college_lessons():
    return [
        make_label_lesson(["linear algebra", "vector", "matrix", "eigenvalue", "transformation", "determinant"], "Linear Algebra"),
        make_label_lesson(["organic chemistry", "synthesis", "mechanism", "stereochemistry", "spectroscopy"], "Organic Chemistry"),
        make_label_lesson(["quantum mechanics", "relativity", "particle", "field", "wave function", "uncertainty"], "Quantum Physics"),
        make_label_lesson(["algorithm", "complexity", "data structure", "recursion", "optimization", "computation"], "Computer Science"),
        make_label_lesson(["macroeconomics", "microeconomics", "fiscal", "monetary", "equilibrium", "elasticity"], "Economics College"),
        make_label_lesson(["research", "methodology", "hypothesis", "empirical", "peer review", "publication"], "Research Methods"),
        make_label_lesson(["literary theory", "deconstruction", "feminism", "postcolonial", "structuralism"], "Literary Theory"),
        make_label_lesson(["cognitive science", "neuroscience", "artificial intelligence", "learning", "memory"], "Cognitive Science"),
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
