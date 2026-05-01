"""
REM Sleep Synthesis - Novel Inference via Dream Traversal

Based on Sovereign Mind's REM mechanism:
- τ=0: Geometric (physical laws)
- τ=1: Definitive (protected facts)
- τ=2: Causal (cause→effect)
- τ=3: Categorical (REM-synthesized novel inference)
- τ=4: Contextual (temporary)

During REM sleep:
1. Seed from high-sigma events (surprise)
2. Outward walk: follow weak springs → high interference (novel intersections)
3. Form τ=3 bridges at intersections (categorical inference)
4. Inward walk: follow strong springs → ground to identity:self
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, FLAG_PINNED
from lrn.natural_tokenize import learn_from_text
from lrn.inference import add_word_nodes


# Tau purposes
TAU_PURPOSE = {
    0: "Geometric - physical laws, absolute",
    1: "Definitive - protected facts, words bound",
    2: "Causal - cause→effect learned from experience",
    3: "Categorical - REM synthesized novel inference", 
    4: "Contextual - temporary associations",
}


class REMSleep:
    """REM sleep synthesis for novel inference."""
    
    def __init__(self, lnn: LatticeNN):
        self.lnn = lnn
        self.sigma_accumulator = 0  # Accumulate surprise
        self.high_sigma_events = []  # Events tagged for REM
    
    def tag_event(self, event: str, surprise: int):
        """Tag high-surprise event for REM processing."""
        self.sigma_accumulator += surprise
        self.high_sigma_events.append((event, surprise))
        
        if len(self.high_sigma_events) > 5:
            # Keep only highest surprise
            self.high_sigma_events.sort(key=lambda x: -x[1])
            self.high_sigma_events = self.high_sigma_events[:5]
    
    def compute_rem_depth(self) -> int:
        """REM depth driven by accumulated sigma (surprise)."""
        # More surprise = deeper REM = more synthesis
        depth = min(10, max(1, self.sigma_accumulator // 100))
        return depth
    
    def run_rem_cycle(self, wake_context: dict) -> dict:
        """
        Run one REM synthesis cycle.
        
        Args:
            wake_context: {node_name: interference_score} from wake experience
        Returns:
            {"novel_bridges": list, "inference_count": int}
        """
        depth = self.compute_rem_depth()
        
        # Get seeds: high-sigma events or identity:self
        seeds = []
        if self.high_sigma_events:
            for event, _ in self.high_sigma_events[:3]:
                seeds.append(f"word:{event}")
        else:
            seeds.append("identity:self")
        
        novel_bridges = []
        
        # REM Outward: follow weak springs to high interference (novel)
        outward = []
        current = seeds[0] if seeds else "identity:self"
        
        # Initialize wake context with seeds
        ctx = {current: 1024}
        ctx.update(wake_context)
        
        # Outward walk (depth // 2 steps)
        for step in range(depth // 2 + 1):
            # Find neighbors with interference
            candidates = []
            
            neighbors = self.lnn.get_neighbors(current)
            for neighbor_name, sp in neighbors:
                # Weak springs = novel, untried paths
                if sp.stiffness < 20:
                    # Check interference (intersection count)
                    interference = ctx.get(neighbor_name, 0)
                    score = interference * (sp.stiffness + 1)
                    candidates.append((score, neighbor_name, sp))
            
            if not candidates:
                break
            
            # Sort by score (high interference first)
            candidates.sort(reverse=True)
            _, current, sp = candidates[0]
            outward.append(current)
            
            # HEYMANN WIRING: Form τ=3 bridge at intersection
            for prior_node, potential in ctx.items():
                if potential > 200:  # Threshold
                    key = self.lnn._key(prior_node, current)
                    
                    if key in self.lnn.springs:
                        existing = self.lnn.springs[key]
                        if existing.tau > 1:
                            existing.tau = 3  # Promote to categorical
                    else:
                        # Create new τ=3 categorical bridge
                        self.lnn.add_spring(prior_node, current, stiffness=10)
                        key = self.lnn._key(prior_node, current)
                        self.lnn.springs[key].tau = 3  # Categorical!
                        novel_bridges.append((prior_node, current))
        
        # REM Inward: follow strong springs back to identity:self (grounding)
        inward = []
        current = outward[-1] if outward else seeds[0]
        
        for step in range(depth // 2 + 1):
            neighbors = self.lnn.get_neighbors(current)
            
            # Strong springs = grounded, known
            candidates = []
            for neighbor_name, sp in neighbors:
                if sp.stiffness >= 30:  # Strong = grounded
                    candidates.append((sp.stiffness, neighbor_name))
            
            if not candidates:
                break
            
            candidates.sort(reverse=True)
            current = candidates[0][1]
            inward.append(current)
            
            if current == "identity:self":
                break
        
        # Compute NIR (Novel Inference Ratio)
        tau3_count = sum(1 for sp in self.lnn.springs.values() if sp.tau == 3)
        total_springs = len(self.lnn.springs)
        nir = (tau3_count * 10240) // max(1, total_springs)
        
        return {
            "novel_bridges": novel_bridges,
            "inference_count": len(novel_bridges),
            "rem_depth": depth,
            "outward_path": outward,
            "inward_path": inward,
            "tau3_count": tau3_count,
            "nir": nir,
        }


def test_rem_synthesis():
    """Test REM sleep synthesis."""
    print("=" * 60)
    print("REM SLEEP SYNTHESIS TEST")
    print("=" * 60)
    
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    
    # Train sensory + causation
    print("\n--- Training ---")
    SENSORY = [
        "fire is hot", "fire burns wood", "water is wet", "water cools fire",
        "ice is cold", "sun heats ground", "rock is hard", "soft pillow",
    ]
    CAUSATION = [
        "fire burns wood and wood becomes black",
        "water cools fire and fire becomes small",
        "sun heats ground and ground becomes warm",
    ]
    
    for text in SENSORY:
        learn_from_text(lnn, text, repetitions=10, learn_type="sensory")
    for text in CAUSATION:
        learn_from_text(lnn, text, repetitions=10, learn_type="causation")
    
    # Add word nodes
    add_word_nodes(lnn, SENSORY + CAUSATION)
    
    # Add identity:self
    lnn.add_node("identity:self")
    lnn.add_spring("word:fire", "identity:self", stiffness=50, tau=1)
    
    print(f"Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
    
    # Initialize REM
    rem = REMSleep(lnn)
    rem.tag_event("fire", surprise=100)
    rem.tag_event("water", surprise=80)
    
    # Wake context: what the agent experienced
    wake_context = {
        "word:fire": 500,
        "word:burns": 400,
        "word:water": 300,
        "word:hot": 250,
    }
    
    # Run REM cycle
    print("\n--- Running REM Cycle ---")
    result = rem.run_rem_cycle(wake_context)
    
    print(f"REM Depth: {result['rem_depth']}")
    print(f"Novel bridges formed: {result['inference_count']}")
    print(f"Outward path: {result['outward_path']}")
    print(f"Inward path: {result['inward_path']}")
    print(f"τ=3 (Categorical) bridges: {result['tau3_count']}")
    print(f"NIR (Novel Inference Ratio): {result['nir']}")
    
    # Check tau distribution
    print("\n--- Tau Distribution ---")
    tau_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for sp in lnn.springs.values():
        tau_counts[sp.tau] = tau_counts.get(sp.tau, 0) + 1
    
    for tau, count in tau_counts.items():
        print(f"  τ={tau}: {count} ({TAU_PURPOSE.get(tau, '')})")
    
    # Run multiple REM cycles
    print("\n--- Multiple REM Cycles ---")
    for i in range(5):
        result = rem.run_rem_cycle(wake_context)
        print(f"Cycle {i+1}: {result['inference_count']} new bridges, τ3={result['tau3_count']}, NIR={result['nir']}")
    
    return result


if __name__ == "__main__":
    test_rem_synthesis()