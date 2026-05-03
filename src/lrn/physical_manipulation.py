"""
Physical Manipulation Simulation — Counting Objects, Sorting, Measuring

Children learn math by physically manipulating objects: counting blocks,
combining groups, taking away. This simulation creates the sensory-motor
springs that text-only training cannot provide.

Mechanism:
1. Object nodes (block_1, block_2, etc.) are created
2. Counting creates springs between sequential quantities
3. Addition = combining groups → springs between (X, plus, Y, equals, Z)
4. Subtraction = removing objects → springs between (X, minus, Y, equals, Z)
5. Sorting creates springs between category labels and their members
6. Measuring creates springs between quantity words and scale positions
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn.node import Node
from lrn.spring import Spring, STIFFNESS_CEILINGS
from lrn.lattice import LatticeNN


def physical_manipulation(lnn, verbose=True):
    """Run physical manipulation simulation on the lattice.
    
    This simulates a child playing with blocks, counting fingers,
    sorting objects, and measuring things.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"PHYSICAL MANIPULATION SIMULATION")
        print(f"{'='*60}")
    
    # Phase 1: Counting objects (blocks, fingers, snacks)
    if verbose:
        print(f"\n  Phase 1: Counting objects...")
    _simulate_counting(lnn)
    
    # Phase 2: Addition by combining groups
    if verbose:
        print(f"\n  Phase 2: Addition (combining groups)...")
    _simulate_addition(lnn)
    
    # Phase 3: Subtraction by removing objects
    if verbose:
        print(f"\n  Phase 3: Subtraction (removing objects)...")
    _simulate_subtraction(lnn)
    
    # Phase 4: Sorting by category
    if verbose:
        print(f"\n  Phase 4: Sorting by category...")
    _simulate_sorting(lnn)
    
    # Phase 5: Measuring on scales
    if verbose:
        print(f"\n  Phase 5: Measuring on scales...")
    _simulate_measuring(lnn)
    
    # Propagate to spread activation
    from lrn import propagate
    propagate(lnn, n_steps=5)
    
    if verbose:
        print(f"\n  After physical manipulation: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    return lnn


def _ensure_node(lnn, label, role="word"):
    """Ensure a node exists, create if needed."""
    key = f"{role}:{label}"
    if key not in lnn.nodes:
        lnn.nodes[key] = Node(name=key)
    return lnn.nodes[key]


def _create_spring(lnn, a_key, b_key, stiffness, tau=2, is_constitutive=False):
    """Create or strengthen a spring between two nodes."""
    if a_key not in lnn.nodes or b_key not in lnn.nodes:
        return
    
    key = lnn._key(a_key, b_key)
    ceiling = STIFFNESS_CEILINGS.get(tau, 48)
    if key in lnn.springs:
        # Strengthen existing spring
        sp = lnn.springs[key]
        sp.stiffness = min(sp.stiffness + stiffness, ceiling)
        sp.saturation_count += 1
    else:
        # Create new spring
        lnn.springs[key] = Spring(
            stiffness=stiffness,
            tau=tau,
        )


def _simulate_counting(lnn):
    """Simulate counting objects: blocks, fingers, snacks.
    
    Creates springs between:
    - Sequential numbers (one↔two↔three...)
    - Number words and quantity concepts
    - Counting actions and number sequences
    """
    # Number words one through twenty
    numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight",
               "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
               "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
    
    # Ensure all number nodes exist
    for num in numbers:
        _ensure_node(lnn, num)
    
    # Sequential counting springs (strong, constitutive)
    # When you count, each number leads to the next
    for i in range(len(numbers) - 1):
        a = f"word:{numbers[i]}"
        b = f"word:{numbers[i+1]}"
        _create_spring(lnn, a, b, stiffness=120, tau=0, is_constitutive=True)
    
    # Counting objects: "one block, two blocks, three blocks..."
    # Creates springs between number words and the concept of counting
    counting_words = ["count", "counting", "blocks", "fingers", "objects"]
    for cw in counting_words:
        _ensure_node(lnn, cw)
        for num in numbers[:10]:  # Count to 10 with objects
            _create_spring(lnn, f"word:{cw}", f"word:{num}", stiffness=30, tau=2)
    
    # Quantity springs: number words connect to their quantity meaning
    # "one" connects to "single", "two" to "pair", "three" to "triple"
    quantity_pairs = [
        ("one", "single"), ("two", "pair"), ("three", "triple"),
        ("four", "quadruple"), ("five", "quintuple"),
    ]
    for num, qty in quantity_pairs:
        _ensure_node(lnn, qty)
        _create_spring(lnn, f"word:{num}", f"word:{qty}", stiffness=90, tau=1)
    
    # Skip counting (by 2s, 5s, 10s) - important math foundation
    # Count by 2s: two, four, six, eight, ten...
    for i in range(0, 18, 2):
        if i + 2 < len(numbers):
            _create_spring(lnn, f"word:{numbers[i]}", f"word:{numbers[i+2]}",
                          stiffness=60, tau=2)
    
    # Count by 5s: five, ten, fifteen, twenty
    for i in [4, 9, 14]:
        if i + 5 < len(numbers):
            _create_spring(lnn, f"word:{numbers[i]}", f"word:{numbers[i+5]}",
                          stiffness=60, tau=2)
    
    # Count by 10s: ten, twenty
    _create_spring(lnn, "word:ten", "word:twenty", stiffness=90, tau=2)


def _simulate_addition(lnn):
    """Simulate addition by combining groups of objects.
    
    Creates springs between:
    - "X plus Y equals Z" for all combinations within 20
    - Addition words (plus, add, sum, total, combine)
    - Result numbers and their component parts
    """
    numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight",
               "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
               "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
    
    # Ensure addition operation words exist
    add_words = ["plus", "add", "sum", "total", "combine", "together", "more", "equals"]
    for aw in add_words:
        _ensure_node(lnn, aw)
    
    # Create strong springs between addition operation words
    for i in range(len(add_words)):
        for j in range(i + 1, len(add_words)):
            _create_spring(lnn, f"word:{add_words[i]}", f"word:{add_words[j]}",
                          stiffness=90, tau=1)
    
    # Simulate physical addition: combining groups
    # "I have 3 blocks and get 2 more, now I have 5"
    for x in range(1, 11):
        for y in range(1, 11):
            result = x + y
            if result <= 20:
                # Create springs: X ↔ plus ↔ Y ↔ equals ↔ result
                x_node = f"word:{numbers[x-1]}"
                y_node = f"word:{numbers[y-1]}"
                r_node = f"word:{numbers[result-1]}"
                
                # X connects to plus (strong, constitutive for addition)
                _create_spring(lnn, x_node, "word:plus", stiffness=60, tau=0)
                
                # Plus connects to Y
                _create_spring(lnn, "word:plus", y_node, stiffness=60, tau=0)
                
                # Y connects to equals
                _create_spring(lnn, y_node, "word:equals", stiffness=60, tau=0)
                
                # Equals connects to result (very strong - this is the answer)
                _create_spring(lnn, "word:equals", r_node, stiffness=120, tau=0)
                
                # X also connects directly to result (commutative property)
                _create_spring(lnn, x_node, r_node, stiffness=30, tau=2)
                
                # Y also connects directly to result
                _create_spring(lnn, y_node, r_node, stiffness=30, tau=2)
    
    # Doubles are especially important for children: 1+1, 2+2, 3+3...
    for x in range(1, 11):
        result = x + x
        if result <= 20:
            x_node = f"word:{numbers[x-1]}"
            r_node = f"word:{numbers[result-1]}"
            # Doubles get extra reinforcement
            _create_spring(lnn, x_node, r_node, stiffness=180, tau=0)
            _create_spring(lnn, x_node, "word:double", stiffness=90, tau=1)
            _ensure_node(lnn, "double")
    
    # Near doubles: 3+4 is like 3+3+1
    for x in range(1, 10):
        y = x + 1
        result = x + y
        if result <= 20:
            x_node = f"word:{numbers[x-1]}"
            y_node = f"word:{numbers[y-1]}"
            r_node = f"word:{numbers[result-1]}"
            _create_spring(lnn, x_node, y_node, stiffness=60, tau=1)
            _create_spring(lnn, y_node, r_node, stiffness=60, tau=1)


def _simulate_subtraction(lnn):
    """Simulate subtraction by removing objects.
    
    Creates springs between:
    - "X minus Y equals Z" for valid subtractions within 20
    - Subtraction words (minus, subtract, less, take away, left)
    - Result numbers and what was removed
    """
    numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight",
               "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
               "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
    
    # Ensure subtraction operation words exist
    sub_words = ["minus", "subtract", "less", "take", "away", "left", "remain",
                 "difference", "equals"]
    for sw in sub_words:
        _ensure_node(lnn, sw)
    
    # Create strong springs between subtraction operation words
    for i in range(len(sub_words)):
        for j in range(i + 1, len(sub_words)):
            _create_spring(lnn, f"word:{sub_words[i]}", f"word:{sub_words[j]}",
                          stiffness=90, tau=1)
    
    # Simulate physical subtraction: removing objects
    # "I have 5 blocks and take away 2, now I have 3 left"
    for total in range(2, 21):
        for take in range(1, total):
            result = total - take
            if result >= 1:
                total_node = f"word:{numbers[total-1]}"
                take_node = f"word:{numbers[take-1]}"
                result_node = f"word:{numbers[result-1]}"
                
                # Total connects to minus
                _create_spring(lnn, total_node, "word:minus", stiffness=60, tau=0)
                
                # Minus connects to take amount
                _create_spring(lnn, "word:minus", take_node, stiffness=60, tau=0)
                
                # Take connects to equals
                _create_spring(lnn, take_node, "word:equals", stiffness=60, tau=0)
                
                # Equals connects to result
                _create_spring(lnn, "word:equals", result_node, stiffness=120, tau=0)
                
                # Total connects directly to result (inverse relationship)
                _create_spring(lnn, total_node, result_node, stiffness=30, tau=2)
                
                # Take connects to result (what's left)
                _create_spring(lnn, take_node, result_node, stiffness=30, tau=2)
    
    # Fact families: if 3+2=5, then 5-2=3 and 5-3=2
    # These get extra reinforcement because they connect addition and subtraction
    for x in range(1, 11):
        for y in range(1, 11):
            result = x + y
            if result <= 20:
                x_node = f"word:{numbers[x-1]}"
                y_node = f"word:{numbers[y-1]}"
                r_node = f"word:{numbers[result-1]}"
                
                # Connect addition result to subtraction operations
                _create_spring(lnn, r_node, "word:minus", stiffness=60, tau=1)
                _create_spring(lnn, r_node, "word:subtract", stiffness=60, tau=1)


def _simulate_sorting(lnn):
    """Simulate sorting objects by category.
    
    Creates springs between:
    - Category labels and their members
    - Similar categories (animals ↔ living things)
    - Sorting actions (sort, group, classify, category)
    """
    # Sorting vocabulary
    sort_words = ["sort", "group", "classify", "category", "same", "different",
                  "match", "pattern", "order", "arrange"]
    for sw in sort_words:
        _ensure_node(lnn, sw)
    
    # Connect sorting words together
    for i in range(len(sort_words)):
        for j in range(i + 1, len(sort_words)):
            _create_spring(lnn, f"word:{sort_words[i]}", f"word:{sort_words[j]}",
                          stiffness=60, tau=1)
    
    # Category: Animals
    animals = ["dog", "cat", "bird", "fish", "frog", "rabbit", "horse", "cow", "pig"]
    for animal in animals:
        _ensure_node(lnn, animal)
        _create_spring(lnn, "word:animal", f"word:{animal}", stiffness=60, tau=2)
    _ensure_node(lnn, "animal")
    
    # Category: Colors
    colors = ["red", "blue", "green", "yellow", "orange", "purple", "brown", "black", "white"]
    for color in colors:
        _ensure_node(lnn, color)
        _create_spring(lnn, "word:color", f"word:{color}", stiffness=60, tau=2)
    _ensure_node(lnn, "color")
    
    # Category: Shapes
    shapes = ["circle", "square", "triangle", "rectangle", "oval", "star", "heart"]
    for shape in shapes:
        _ensure_node(lnn, shape)
        _create_spring(lnn, "word:shape", f"word:{shape}", stiffness=60, tau=2)
    _ensure_node(lnn, "shape")
    
    # Category: Numbers
    numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    for num in numbers:
        _ensure_node(lnn, num)
        _create_spring(lnn, "word:number", f"word:{num}", stiffness=60, tau=2)
    _ensure_node(lnn, "number")
    
    # Category: Food
    foods = ["apple", "banana", "bread", "milk", "cheese", "egg", "carrot", "pizza"]
    for food in foods:
        _ensure_node(lnn, food)
        _create_spring(lnn, "word:food", f"word:{food}", stiffness=60, tau=2)
    _ensure_node(lnn, "food")
    
    # Category: Body parts
    body_parts = ["head", "hand", "foot", "eye", "ear", "nose", "mouth", "arm", "leg"]
    for bp in body_parts:
        _ensure_node(lnn, bp)
        _create_spring(lnn, "word:body", f"word:{bp}", stiffness=60, tau=2)
    _ensure_node(lnn, "body")
    
    # Cross-category springs: animals and living things
    _ensure_node(lnn, "living")
    _create_spring(lnn, "word:animal", "word:living", stiffness=90, tau=1)
    _create_spring(lnn, "word:plant", "word:living", stiffness=90, tau=1)
    _ensure_node(lnn, "plant")


def _simulate_measuring(lnn):
    """Simulate measuring on scales (length, weight, temperature, time).
    
    Creates springs between:
    - Measurement words and scale positions
    - Comparison words (longer, shorter, heavier, lighter)
    - Standard units (inch, foot, centimeter, pound)
    """
    # Measurement vocabulary
    measure_words = ["measure", "length", "width", "height", "weight", "size",
                     "long", "short", "heavy", "light", "tall", "wide", "narrow",
                     "thick", "thin", "deep", "shallow"]
    for mw in measure_words:
        _ensure_node(lnn, mw)
    
    # Connect measurement words together
    for i in range(len(measure_words)):
        for j in range(i + 1, len(measure_words)):
            _create_spring(lnn, f"word:{measure_words[i]}", f"word:{measure_words[j]}",
                          stiffness=30, tau=2)
    
    # Comparison pairs (opposites)
    opposites = [
        ("long", "short"), ("heavy", "light"), ("tall", "short"),
        ("wide", "narrow"), ("thick", "thin"), ("deep", "shallow"),
        ("big", "small"), ("more", "less"),
    ]
    for a, b in opposites:
        _ensure_node(lnn, a)
        _ensure_node(lnn, b)
        # Opposites connect through comparison concept
        _create_spring(lnn, f"word:{a}", f"word:compare", stiffness=60, tau=1)
        _create_spring(lnn, f"word:{b}", f"word:compare", stiffness=60, tau=1)
        _ensure_node(lnn, "compare")
    
    # Units of measurement
    length_units = ["inch", "inches", "foot", "feet", "yard", "centimeter", "meter"]
    for unit in length_units:
        _ensure_node(lnn, unit)
        _create_spring(lnn, "word:length", f"word:{unit}", stiffness=90, tau=1)
    
    weight_units = ["pound", "pounds", "ounce", "gram", "kilogram"]
    for unit in weight_units:
        _ensure_node(lnn, unit)
        _create_spring(lnn, "word:weight", f"word:{unit}", stiffness=90, tau=1)
    
    # Time measurement
    time_words = ["second", "minute", "hour", "day", "week", "month", "year",
                  "o'clock", "half", "past", "time", "clock", "watch"]
    for tw in time_words:
        _ensure_node(lnn, tw)
        _create_spring(lnn, "word:time", f"word:{tw}", stiffness=60, tau=2)
    _ensure_node(lnn, "time")
    
    # Temperature scale
    temp_words = ["hot", "warm", "cool", "cold", "freezing", "boiling", "temperature"]
    for i in range(len(temp_words) - 1):
        _ensure_node(lnn, temp_words[i])
        _ensure_node(lnn, temp_words[i + 1])
        _create_spring(lnn, f"word:{temp_words[i]}", f"word:{temp_words[i+1]}",
                      stiffness=60, tau=1)
    
    # Money measurement
    money_words = ["penny", "nickel", "dime", "quarter", "dollar", "cent", "cents"]
    for mw in money_words:
        _ensure_node(lnn, mw)
        _create_spring(lnn, "word:money", f"word:{mw}", stiffness=60, tau=2)
    _ensure_node(lnn, "money")
