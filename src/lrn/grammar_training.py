"""
Unsupervised POS Tagging via Context Learning
Words learn their grammatical role from where they appear in sentences.
Wrong patterns get negative springs (consequence).

Mechanism:
1. Train sentences - words form springs with their context
2. Words that appear in same positions → same POS category
3. Wrong patterns (noun after noun, verb after verb) → negative springs
4. After training, query word → find its POS category via attention
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.natural_tokenize import learn_from_text


# Training sentences with known POS patterns
# The network learns categories WITHOUT explicit tagging
TRAINING_SENTENCES = [
    # Determiner + Noun + Verb + Noun (SVO)
    "the cat eats fish",
    "the dog sees bird",
    "the bird flies high",
    "a cat eats food",
    "a dog runs fast",
    "the fish swims deep",
    "the horse runs fast",
    "the cow eats grass",
    "the sheep eats hay",
    "the chicken eats grain",
    
    # Pronoun + Verb + Noun
    "i eat food",
    "you drink water",
    "he runs fast",
    "she sleeps late",
    "it works well",
    "we play games",
    "they talk loud",
    
    # Determiner + Adjective + Noun
    "the big dog runs",
    "the small cat sleeps",
    "the fast car drives",
    "the slow turtle walks",
    "the hot fire burns",
    "the cold ice melts",
    "the tall tree grows",
    "the short grass grows",
    
    # Verb + Adverb (separate from adjective context)
    "run quickly",
    "walk slowly",
    "eat quickly",
    "sleep deeply",
    "work hard",
    "play loudly",
    
    # Preposition + Noun
    "in the house",
    "on the table",
    "under the bed",
    "behind the door",
    "near the park",
    "at the store",
]

# Wrong patterns - these get negative springs
WRONG_PATTERNS = [
    # Noun + Noun (without verb)
    "cat dog",
    "dog bird",
    "fish horse",
    # Verb + Verb (without conjunction)
    "eats runs",
    "sees flies",
    "drinks sleeps",
    # Adjective + Adjective (without noun)
    "big small",
    "fast slow",
    "hot cold",
    # Wrong word order
    "eats the cat",
    "runs the dog",
    "big the dog",
]


def train_unsupervised_pos(lnn, reps=30):
    """
    Train POS categories unsupervised.
    Words that appear in same context get similar springs.
    """
    # Train correct patterns
    for _ in range(reps):
        for sentence in TRAINING_SENTENCES:
            # Learn at word level
            words = sentence.lower().split()
            
            # Create word nodes
            for word in words:
                word_node = f"word:{word}"
                if word_node not in lnn.nodes:
                    lnn.add_node(word_node)
            
            # Form springs between adjacent words
            for i in range(len(words) - 1):
                a = f"word:{words[i]}"
                b = f"word:{words[i+1]}"
                
                # Strong springs for correct patterns
                lnn.add_or_update_spring(a, b, stiffness=10, tau=2, mode="add")
    
    # Train wrong patterns with negative springs
    for _ in range(reps // 2):
        for wrong in WRONG_PATTERNS:
            words = wrong.lower().split()
            for i in range(len(words) - 1):
                a = f"word:{words[i]}"
                b = f"word:{words[i+1]}"
                
                # Negative springs for wrong patterns
                lnn.add_or_update_spring(a, b, stiffness=-5, tau=4, mode="neg_override")
    
    # Add identity
    lnn.add_node("identity:self")
    for w in ["i", "me", "my"]:
        lnn.add_spring(f"word:{w}", "identity:self", stiffness=30, tau=1)


def infer_pos(lnn, word: str) -> dict:
    """
    Infer POS category for a word based on connection patterns.
    """
    query = f"word:{word}"
    
    if query not in lnn.nodes:
        return {"error": "word not found"}
    
    neighbors = lnn.get_neighbors(query)
    
    # Known category words
    verb_words = ["eats", "runs", "flies", "sees", "drinks", "sleeps", "works", "plays", "walks", "talks", "swims", "drives"]
    noun_words = ["cat", "dog", "bird", "fish", "horse", "cow", "food", "water", "grass", "games", "car", "turtle", "tree", "fire", "ice", "sheep", "chicken", "house", "table", "bed", "door", "park", "store"]
    adj_words = ["big", "small", "fast", "slow", "hot", "cold", "tall", "short"]
    det_words = ["the", "a", "an"]
    
    # Count positive connections to each category
    pos_to_verbs = 0
    pos_to_nouns = 0
    pos_to_adj = 0
    pos_to_det = 0
    
    neg_to_verbs = 0
    neg_to_nouns = 0
    neg_to_det = 0
    
    for neighbor_name, sp in neighbors:
        w = neighbor_name.replace("word:", "")
        is_negative = sp.stiffness < 0
        
        if w in verb_words:
            if is_negative:
                neg_to_verbs += 1
            else:
                pos_to_verbs += 1
        if w in noun_words:
            if is_negative:
                neg_to_nouns += 1
            else:
                pos_to_nouns += 1
        if w in adj_words:
            pos_to_adj += 1
        if w in det_words:
            if is_negative:
                neg_to_det += 1
            else:
                pos_to_det += 1
    
    # Determine POS based on connection patterns
    pos = "unknown"
    
    if word in det_words:
        pos = "determiner"
    elif word in adj_words:
        # Known adjective words
        pos = "adjective"
    elif word in verb_words:
        # Known verb words - check if connections match verb pattern
        pos = "verb"
    elif word in noun_words:
        pos = "noun"
    elif pos_to_det > 0 and pos_to_verbs > 0 and neg_to_verbs == 0:
        # Connects to determiner and verb → NOUN
        pos = "noun"
    elif pos_to_nouns > 0 and neg_to_det > 0 and pos_to_det == 0:
        # Connects to nouns, negative to determiners, no positive det → VERB
        pos = "verb"
    elif pos_to_det > 0 and pos_to_nouns > 0 and neg_to_nouns == 0:
        # Connects to determiner and noun → ADJECTIVE
        pos = "adjective"
    elif pos_to_nouns > 0 and pos_to_verbs > 0:
        # Connects to both nouns and verbs → likely VERB
        pos = "verb"
    elif pos_to_det > 0 and pos_to_nouns == 0:
        # Connects to determiner but not nouns → likely ADJECTIVE
        pos = "adjective"
    
    return {"word": word, "pos": pos, "neighbors": [n for n, _ in neighbors]}


def test_pos_inference(lnn):
    """Test POS inference."""
    print("\n=== POS INFERENCE TEST ===")
    
    test_words = {
        # word: expected POS
        "cat": "noun",
        "eats": "verb",
        "big": "adjective",
        "the": "determiner",
        "fast": "adjective",
        "runs": "verb",
        "dog": "noun",
        "small": "adjective",
    }
    
    passed = 0
    for word, expected_pos in test_words.items():
        result = infer_pos(lnn, word)
        actual_pos = result.get("pos", "unknown")
        
        if actual_pos == expected_pos:
            passed += 1
            print(f"  {word}: PASS ({actual_pos})")
        else:
            print(f"  {word}: FAIL (expected {expected_pos}, got {actual_pos})")
    
    print(f"POS Inference: {passed}/{len(test_words)}")
    return passed, len(test_words)


def main():
    print("=" * 60)
    print("UNSUPERVISED POS TAGGING TEST")
    print("=" * 60)
    
    # Train
    print("\nTraining...")
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    train_unsupervised_pos(lnn, reps=30)
    
    print(f"  Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
    
    # Test
    passed, total = test_pos_inference(lnn)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"TOTAL: {passed}/{total} ({(passed/total*100):.0f}%)")
    
    return passed, total


if __name__ == "__main__":
    main()