"""
LRN Training Loop - Reusable across all levels
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.natural_tokenize import learn_from_text
from lrn.inference import add_word_nodes
from lrn.charts import progress_bar
from lrn.spring import STIFFNESS_CEILINGS, SATURATION_LEARNED_THRESHOLD, SATURATED_RETURN_RATE


def optimal_rem(lnn, wake_buffer):
    """REM synthesis - forms τ=3 categorical bridges between co-occurring words.
    
    Respects stiffness ceilings and saturation. Once a spring is learned
    (saturated 3+ times), it gets 10% reinforcement and is skipped for
    heavy consolidation, freeing capacity for weaker connections.
    """
    recent_words = set()
    for s in wake_buffer:
        for w in s.lower().split():
            recent_words.add(f"word:{w}")
    
    word_list = list(recent_words)
    for i in range(len(word_list)):
        for j in range(i+1, len(word_list)):
            a, b = word_list[i], word_list[j]
            key = lnn._key(a, b)
            ceiling = STIFFNESS_CEILINGS.get(3, 200)
            
            if key in lnn.springs:
                sp = lnn.springs[key]
                
                # Skip if already learned (saturated multiple times)
                if sp.saturated and sp.saturation_count >= SATURATION_LEARNED_THRESHOLD:
                    # Diminishing returns: 10% reinforcement
                    sp.stiffness = min(ceiling, sp.stiffness + 1)
                    continue
                
                # Check if hitting ceiling
                if sp.stiffness >= ceiling:
                    sp.saturated = True
                    sp.saturation_count += 1
                    continue
                
                # Normal reinforcement
                sp.stiffness = min(ceiling, max(sp.stiffness, 10))
                sp.tau = 3
            else:
                lnn.add_or_update_spring(a, b, stiffness=10, tau=3, mode="add")


def prune_springs(lnn, min_exposure=2):
    """Remove springs that never formed meaningful connections.
    
    Springs with exposure_count < min_exposure and τ=4 (contextual only)
    are pruned. This prevents noise from single co-occurrences.
    """
    to_remove = []
    for key, sp in lnn.springs.items():
        if sp.tau == 4 and sp.exposure_count < min_exposure and sp.stiffness < 5:
            to_remove.append(key)
    
    for key in to_remove:
        del lnn.springs[key]
    
    return len(to_remove)


def train(lnn, sentences, reps=50, learn_type="language", verbose=True, rem_interval="end"):
    """
    Train on sentences with REM synthesis.
    
    Args:
        lnn: LatticeNN instance
        sentences: list of training sentences
        reps: number of training repetitions
        learn_type: "sensory" or "language"
        verbose: print progress
        rem_interval: REM timing - "end" (default), 1 (every sentence), or N (every N sentences)
    """
    wake_buffer = []
    total_sentences = len(sentences) * reps
    
    for rep in range(reps):
        for idx, sentence in enumerate(sentences):
            learn_from_text(lnn, sentence, repetitions=1, learn_type=learn_type)
            add_word_nodes(lnn, [sentence])
            
            wake_buffer.append(sentence)
            if len(wake_buffer) > 20:
                wake_buffer = wake_buffer[-20:]
            
            # REM after each sentence if interval is 1
            if rem_interval == 1:
                optimal_rem(lnn, wake_buffer)
            
            # REM periodically (only if numeric and > 1)
            if isinstance(rem_interval, int) and rem_interval > 1:
                if (idx + 1) % rem_interval == 0:
                    optimal_rem(lnn, wake_buffer)
            
            if verbose and (idx + 1) % 50 == 0:
                current = rep * len(sentences) + idx + 1
                print(progress_bar(current, total_sentences), end="", flush=True)
        
        # REM after each full corpus pass (unless end-only or every-sentence)
        if rem_interval != "end" and rem_interval != 1:
            optimal_rem(lnn, wake_buffer)
        
        propagate(lnn, n_steps=3)
    
    # Final REM (always)
    optimal_rem(lnn, wake_buffer)
    
    # Prune noise springs
    pruned = prune_springs(lnn)
    
    propagate(lnn, n_steps=5)
    
    if verbose:
        print(progress_bar(total_sentences, total_sentences))
        if pruned:
            print(f"  Pruned: {pruned} noise springs")
        print()


def create_lnn():
    """Create a new LatticeNN with identity anchor."""
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    return lnn
