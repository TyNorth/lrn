"""
Chat with the 2nd Grade lattice - show what it knows
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn.sensory_grounding import sensory_grounding
from lrn.harmonic_lessons import harmonic_video_training
from lrn.physical_manipulation import physical_manipulation
from lrn.social_interaction import social_interaction
from lrn.trainer import train
from lrn.corpora import get_corpus
from lrn import propagate
from lrn.natural_tokenize import learn_from_text
from lrn.inference import add_word_nodes

print("Loading 2nd Grade lattice...")
lnn = sensory_grounding(verbose=False)
harmonic_video_training(lnn, 'second_grade', verbose=False)
physical_manipulation(lnn, verbose=False)
social_interaction(lnn, verbose=False)
corpus = get_corpus('second_grade')
train(lnn, corpus, reps=1, rem_interval='end', verbose=False)
print(f"Loaded: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs\n")


def query(lnn, question, steps=2):
    """Query the lattice and show what activates."""
    # Reset activations
    for n in lnn.nodes.values():
        n.activation = 0
    
    # Learn the question
    learn_from_text(lnn, question, repetitions=1, learn_type="language")
    add_word_nodes(lnn, [question])
    
    # Activate query words
    words = question.lower().split()
    for word in words:
        node = f"word:{word}"
        if node in lnn.nodes:
            lnn.nodes[node].activation = 100
    
    propagate(lnn, n_steps=steps)
    
    # Collect activated words
    activated = []
    for name, node in lnn.nodes.items():
        if name.startswith("word:") and node.activation > 10:
            w = name.replace("word:", "")
            if w not in words:
                activated.append((w, node.activation))
    
    activated.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate confidence from spring stiffness
    total_stiffness = 0
    count = 0
    for word in words:
        node = f"word:{word}"
        if node in lnn.nodes:
            for (a, b), sp in lnn.springs.items():
                if a == node or b == node:
                    total_stiffness += sp.stiffness
                    count += 1
    
    avg_stiffness = total_stiffness // max(1, count)
    confidence = min(100, avg_stiffness // 10)
    
    return activated, confidence


# Test queries covering all 2nd grade domains
queries = [
    # Literacy
    "what is a character",
    "what is the plot",
    "what is a theme",
    "what is a noun",
    "what is a conjunction",
    # Math
    "what is multiplication",
    "what is a fraction",
    "what is area",
    "what is a numerator",
    # Science
    "what is an ecosystem",
    "what is climate",
    "what is a force",
    "what is gravity",
    # Social Studies
    "what is government",
    "what is a constitution",
    "what is a continent",
    # SEL
    "what is empathy",
    "what is resilience",
    "what is integrity",
    "what is teamwork",
]

print("=" * 60)
print("CHAT: What does the 2nd Grade lattice know?")
print("=" * 60)

for q in queries:
    activated, confidence = query(lnn, q)
    
    # Build response from top activated words
    if activated:
        response_words = [w for w, _ in activated[:6]]
        response = " ".join(response_words)
    else:
        response = "(no connections)"
    
    if confidence >= 70:
        display = response
    elif confidence >= 30:
        display = f"{response} [MEDIUM]"
    else:
        display = f"{response} [LOW]"
    
    print(f"\n  Q: {q}")
    print(f"  A: {display}")
    
    if activated:
        top_details = ", ".join(f"{w}({a})" for w, a in activated[:4])
        print(f"     → {top_details}")
