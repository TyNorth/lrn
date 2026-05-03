"""
ARCHITECTURAL ANALYSIS: Optimal Learning Conditions for LRN

What conditions maximize learning, retention, and access across all domains?

KEY FINDINGS:

1. TAU HIERARCHY AS LEARNING SCAFFOLD
   - τ=0 (Constitutive): Sensory foundation - MUST be stable
   - τ=1 (Definitional): Pattern recognition - needs repetition
   - τ=2 (Causal): Relationships - needs context
   - τ=3 (Categorical): Concept clusters - needs REM synthesis
   - τ=4 (Contextual): Pragmatics - needs social interaction

2. CRITICAL BOTTLENECKS IDENTIFIED:
   a) REM synthesis only forms τ=3, but τ=2 causal bridges are equally important
   b) No mechanism to strengthen frequently accessed paths (Hebbian learning)
   c) Wake buffer too small (5 sentences) - limits REM effectiveness
   d) No decay for unused connections - lattice becomes noisy
   e) No distinction between shallow/deep processing
   f) Propagation steps fixed - should adapt to complexity

3. OPTIMAL LEARNING CONDITIONS:
   
   A. PROGRESSIVE SCAFFOLDING
      - Start simple, increase complexity gradually
      - Each level must reach stability before advancing
      - Revisit earlier concepts at higher levels (spiral curriculum)
   
   B. SPACED REPETITION + INTERLEAVING
      - Don't train all reps at once - spread over time
      - Mix old and new content (interleaving)
      - Review previous stages when training new ones
   
   C. ADAPTIVE REM SYNTHESIS
      - Form τ=2 bridges for causal relationships
      - Form τ=3 bridges for categorical clusters
      - Strengthen bridges that are frequently activated
      - Weaken bridges that are never used
   
   D. HEBBIAN LEARNING RULE
      - "Neurons that fire together wire together"
      - When two nodes are co-activated, strengthen their spring
      - When activation decays without use, weaken the spring
   
   E. ATTENTION-DRIVEN CONSOLIDATION
      - After propagation, identify high-activation paths
      - Convert temporary activations to permanent springs
      - This is how "aha moments" become permanent knowledge
   
   F. ERROR-DRIVEN LEARNING
      - Wrong predictions create negative springs
      - Negative springs inhibit incorrect paths
      - This sharpens category boundaries

4. ARCHITECTURAL CHANGES NEEDED:

   a) Add Hebbian learning: strengthen co-activated springs
   b) Add spring decay: weaken unused springs over time
   c) Expand REM to form τ=2 AND τ=3 bridges
   d) Increase wake buffer to 10-20 sentences
   e) Add adaptive propagation: more steps for complex content
   f) Add consolidation phase: convert activations to springs
   g) Add error signal: negative reinforcement for wrong patterns

5. THE "FLUENCY" CONDITION:
   Fluency emerges when:
   - τ=3 categorical bridges form dense clusters
   - τ=2 causal bridges connect clusters meaningfully
   - High-stiffness springs create fast activation paths
   - Negative springs prevent incorrect activations
   - REM synthesis runs after every meaningful chunk
   - Hebbian learning strengthens frequently used paths
   - Decay removes noise from unused connections

6. METRICS FOR MASTERY:
   - Retention: Can the lattice recall after 100 propagation steps?
   - Access: Can any node in a category activate other category members?
   - Generalization: Can the lattice handle novel combinations?
   - Transfer: Can knowledge from one domain help another?
   - Fluency: Does activation spread quickly and accurately?

NEXT STEPS:
Implement these architectural changes and test across domains.
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.natural_tokenize import learn_from_text
from lrn.inference import add_word_nodes


def enhanced_rem(lnn, wake_buffer):
    """Enhanced REM - forms τ=2 AND τ=3 bridges, strengthens co-activated paths."""
    recent_words = set()
    for s in wake_buffer:
        for w in s.lower().split():
            recent_words.add(f"word:{w}")
    
    word_list = list(recent_words)
    for i in range(len(word_list)):
        for j in range(i+1, len(word_list)):
            a, b = word_list[i], word_list[j]
            key = lnn._key(a, b)
            
            if key in lnn.springs:
                sp = lnn.springs[key]
                # Strengthen frequently co-occurring words
                sp.stiffness = max(sp.stiffness, 5)
                # Promote to τ=3 if strong enough
                if sp.stiffness >= 15 and sp.tau > 2:
                    sp.tau = 3
            else:
                # Create new τ=2 causal bridge
                lnn.add_or_update_spring(a, b, stiffness=5, tau=2, mode="add")


def hebbian_update(lnn, activated_nodes):
    """Strengthen springs between co-activated nodes."""
    node_list = list(activated_nodes)
    for i in range(len(node_list)):
        for j in range(i+1, len(node_list)):
            a, b = node_list[i], node_list[j]
            key = lnn._key(a, b)
            if key in lnn.springs:
                sp = lnn.springs[key]
                sp.stiffness += 1  # Hebbian strengthening


def train_enhanced(lnn, sentences, reps=20, learn_type="language"):
    """Training with enhanced REM and Hebbian learning."""
    wake_buffer = []
    
    for rep in range(reps):
        for sentence in sentences:
            learn_from_text(lnn, sentence, repetitions=1, learn_type=learn_type)
            add_word_nodes(lnn, [sentence])
            wake_buffer.append(sentence)
            if len(wake_buffer) > 15:  # Larger wake buffer
                wake_buffer = wake_buffer[-15:]
        
        # Enhanced REM after each corpus pass
        enhanced_rem(lnn, wake_buffer)
        
        # Propagate and apply Hebbian learning
        propagate(lnn, n_steps=3)
        
        # Get activated nodes for Hebbian update
        activated = [n for n, node in lnn.nodes.items() if node.activation > 10]
        if activated:
            hebbian_update(lnn, activated)
    
    # Final REM and consolidation
    enhanced_rem(lnn, wake_buffer)
    propagate(lnn, n_steps=5)


def main():
    print("ARCHITECTURAL ANALYSIS COMPLETE")
    print("See docstring for full analysis.")
    print("\nKey insight: Fluency emerges from the interaction of:")
    print("  1. τ=3 categorical bridges (REM synthesis)")
    print("  2. τ=2 causal bridges (relationship learning)")
    print("  3. Hebbian strengthening (use it or lose it)")
    print("  4. Negative springs (error-driven learning)")
    print("  5. Adaptive propagation (complexity-matched processing)")
    print("\nNext: Implement enhanced training with these conditions.")


if __name__ == "__main__":
    main()
