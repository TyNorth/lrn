"""
Natural Sensory Training - Models real-world experience rhythm

The world doesn't feed you sentences at constant volume. It works like:
- NOISE: constant background sensory events (weak, τ=4 formation)
- SILENCE: gaps where the lattice settles and weak connections decay
- FOCUS: directed attention on specific concepts (τ=3→τ=2 promotion)
- ISOLATED: sudden singular events that create sharp impressions (τ=2→τ=1)

This rhythm is how the lattice should actually learn.
"""
import sys
import random
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.natural_tokenize import learn_from_text
from lrn.inference import add_word_nodes
from lrn.trainer import optimal_rem, prune_springs
from lrn.spring import STIFFNESS_CEILINGS
from lrn.charts import progress_bar


def noise_phase(lnn, noise_corpus, duration=20):
    """Background sensory input — weak, diffuse, forms τ=4 contextual springs."""
    sentences = random.sample(noise_corpus, min(duration, len(noise_corpus)))
    for s in sentences:
        learn_from_text(lnn, s, repetitions=1, learn_type="sensory")
        add_word_nodes(lnn, [s])


def silence_phase(lnn, steps=3):
    """No input — let the lattice settle. Weak springs decay via propagation."""
    propagate(lnn, n_steps=steps)


def focus_phase(lnn, focus_corpus, reps=3):
    """Directed attention — repeat focused input, strong consolidation."""
    for rep in range(reps):
        for s in focus_corpus:
            learn_from_text(lnn, s, repetitions=2, learn_type="sensory")
            add_word_nodes(lnn, [s])


def isolated_event(lnn, event_sentence):
    """Single strong input — creates sharp impression."""
    learn_from_text(lnn, event_sentence, repetitions=5, learn_type="sensory")
    add_word_nodes(lnn, [event_sentence])


def natural_train(lnn, config, verbose=True):
    """
    Train using natural sensory rhythm.
    
    Config format:
    {
        "noise": [list of background sensory sentences],
        "focus": [list of focused concept sentences],
        "isolated": [list of singular event sentences],
        "cycles": int (number of noise→silence→focus→silence cycles),
        "noise_duration": int (sentences per noise phase),
        "silence_steps": int (propagation steps during silence),
        "focus_reps": int (repetitions during focus phase),
    }
    """
    noise_corpus = config.get("noise", [])
    focus_corpus = config.get("focus", [])
    isolated_events = config.get("isolated", [])
    cycles = config.get("cycles", 10)
    noise_duration = config.get("noise_duration", 20)
    silence_steps = config.get("silence_steps", 3)
    focus_reps = config.get("focus_reps", 3)
    
    total_ops = cycles * 4 + len(isolated_events)  # noise + silence + focus + silence + isolated
    op_count = 0
    
    for cycle in range(cycles):
        if verbose:
            print(f"  Cycle {cycle+1}/{cycles}")
        
        # NOISE phase
        if noise_corpus:
            noise_phase(lnn, noise_corpus, duration=noise_duration)
            op_count += 1
            if verbose:
                print(progress_bar(op_count, total_ops), end="", flush=True)
        
        # SILENCE phase
        silence_phase(lnn, steps=silence_steps)
        op_count += 1
        
        # FOCUS phase
        if focus_corpus:
            focus_phase(lnn, focus_corpus, reps=focus_reps)
            op_count += 1
            if verbose:
                print(progress_bar(op_count, total_ops), end="", flush=True)
        
        # SILENCE phase
        silence_phase(lnn, steps=silence_steps)
        op_count += 1
        
        # ISOLATED event (random, ~30% chance per cycle)
        if isolated_events and random.random() < 0.3:
            event = random.choice(isolated_events)
            isolated_event(lnn, event)
            op_count += 1
            if verbose:
                print(progress_bar(op_count, total_ops), end="", flush=True)
    
    # Final REM and pruning
    wake_buffer = focus_corpus[:20] if focus_corpus else noise_corpus[:20]
    optimal_rem(lnn, wake_buffer)
    pruned = prune_springs(lnn)
    propagate(lnn, n_steps=5)
    
    if verbose:
        print(progress_bar(total_ops, total_ops))
        if pruned:
            print(f"  Pruned: {pruned} noise springs")
        print()


def create_lnn():
    """Create a new LatticeNN with identity anchor."""
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    return lnn
