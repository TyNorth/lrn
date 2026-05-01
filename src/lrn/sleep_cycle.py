"""
NREM Sleep Consolidation - τ=4 → τ=2 promotion

NREM Phase 1: Consolidation (d_eff × 0.16)
- Stabilize new τ=4 connections
- τ=4 → τ=2 if strong enough

NREM Phase 2: Integration (d_eff × 0.40)  
- Cross-domain bridging
- Force connections between distant concepts

NREM Phase 3: REM Dream Synthesis
- Creates τ=3 (categorical) bridges at high-interference intersections

NREM Phase 4: Knot Crystallization (post-REM)
- τ=3 → τ=2 promotion for strong novel bridges
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN


def nrem_consolidate(lnn: LatticeNN, efficiency: float = 0.5) -> dict:
    """
    NREM Phase 1: Consolidate τ=4 → τ=2
    d_eff × 0.16 strength multiplier
    """
    promoted = 0
    
    for key, sp in lnn.springs.items():
        if sp.tau == 4 and sp.stiffness >= 20:
            # Strong enough to consolidate
            if sp.stiffness * efficiency >= 10:
                sp.tau = 2  # Promote to causal
                promoted += 1
    
    return {"phase": "consolidation", "promoted": promoted}


def nrem_integrate(lnn: LatticeNN, k_base: int = 64) -> dict:
    """
    NREM Phase 2: Cross-domain integration
    d_eff × 0.40 - force bridging between distant concepts
    
    Looks for high-k clusters and creates τ=2 bridges between them.
    """
    # Find high-k clusters
    high_k_nodes = set()
    for key, sp in lnn.springs.items():
        if sp.stiffness >= k_base // 2:
            a, b = key
            high_k_nodes.add(a)
            high_k_nodes.add(b)
    
    # Create cross-domain bridges
    cross_bridges = 0
    high_k_list = list(high_k_nodes)
    
    for i, node_a in enumerate(high_k_list):
        for node_b in high_k_list[i+1:i+5]:  # Connect to nearby
            key = lnn._key(node_a, node_b)
            if key not in lnn.springs:
                lnn.add_or_update_spring(node_a, node_b, stiffness=k_base//4, tau=2, mode="add")
                cross_bridges += 1
    
    return {"phase": "integration", "cross_bridges": cross_bridges}


def knot_crystallize(lnn: LatticeNN) -> dict:
    """
    NREM Phase 4 (Post-REM): Knot Crystallization
    Promote strong τ=3 to τ=2 (causal)
    """
    promoted = 0
    
    for key, sp in lnn.springs.items():
        if sp.tau == 3 and sp.stiffness >= 50:
            # Strong categorical bridge → becomes causal knowledge
            sp.tau = 2
            promoted += 1
    
    return {"phase": "knot_crystallization", "promoted": promoted}


def full_sleep_cycle(lnn: LatticeNN, efficiency: float = 0.5, k_base: int = 64) -> dict:
    """
    Full sleep cycle: NREM + REM
    """
    results = {}
    
    # NREM Phase 1: Consolidation
    results["consolidation"] = nrem_consolidate(lnn, efficiency)
    
    # NREM Phase 2: Integration
    results["integration"] = nrem_integrate(lnn, k_base)
    
    # (REM would be called here - returns τ=3 bridges)
    
    # NREM Phase 4: Knot Crystallization
    results["crystallization"] = knot_crystallize(lnn)
    
    # Summary
    tau_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for sp in lnn.springs.values():
        tau_counts[sp.tau] = tau_counts.get(sp.tau, 0) + 1
    
    results["tau_summary"] = tau_counts
    
    return results


def test_sleep_cycle():
    """Test full sleep cycle."""
    print("=" * 60)
    print("FULL SLEEP CYCLE TEST")
    print("=" * 60)
    
    from lrn.natural_tokenize import learn_from_text
    from lrn.inference import add_word_nodes
    
    lnn = LatticeNN()
    
    # Train
    SENSORY = ["fire is hot", "fire burns wood", "water is wet", "ice is cold"]
    for text in SENSORY:
        learn_from_text(lnn, text, repetitions=10, learn_type="sensory")
    
    add_word_nodes(lnn, SENSORY)
    
    print(f"Before sleep: {len(lnn.springs)} springs")
    
    # Count before
    before_tau = {}
    for sp in lnn.springs.values():
        before_tau[sp.tau] = before_tau.get(sp.tau, 0) + 1
    print(f"  τ distribution: {before_tau}")
    
    # Full sleep cycle
    results = full_sleep_cycle(lnn)
    
    print(f"\n--- Sleep Results ---")
    print(f"Consolidation: {results['consolidation']}")
    print(f"Integration: {results['integration']}")
    print(f"Crystallization: {results['crystallization']}")
    
    print(f"\nAfter sleep: {len(lnn.springs)} springs")
    print(f"  τ distribution: {results['tau_summary']}")
    
    return results


if __name__ == "__main__":
    test_sleep_cycle()