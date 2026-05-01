"""
Sequential English Training Pipeline

Trains the LRN through developmental stages:
1. Sensory/Letters → 2. Babbling → 3. Phonics → 4. Morphology →
5. Sight Words → 6. Vocabulary → 7. Grammar → 8. Syntax →
9. Sentences → 10. Pragmatics

Each stage trains until its developmental gate passes.
If a gate stalls, more training data is added.
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.natural_tokenize import learn_from_text, TAU_BY_TYPE
from lrn.inference import add_word_nodes
from lrn.english_corpus import ALL_STAGES
from lrn.english_gates import DevelopmentalGate


# Tau assignment per stage
STAGE_TAU = {
    "sensory": 0,      # Letters - rigid
    "babbling": 1,     # Sound exploration
    "phonics": 1,      # Letter→sound mapping
    "morphology": 1,   # Word parts
    "sight_words": 2,  # Instant recognition
    "vocabulary": 2,   # Word meanings
    "grammar": 2,      # POS, word order
    "syntax": 3,       # Hierarchical structure
    "sentences": 3,    # Full comprehension
    "pragmatics": 4,   # Social context
}


class SequentialTrainer:
    """Trains English through developmental stages."""
    
    def __init__(self, max_reps_per_stage=200, max_stall_rounds=3):
        self.lnn = LatticeNN()
        add_identity_anchor(self.lnn)
        self.gates = DevelopmentalGate()
        self.max_reps = max_reps_per_stage
        self.max_stall = max_stall_rounds
        self.stage_results = {}
    
    def train_all_stages(self):
        """Train through all developmental stages sequentially."""
        stages = list(ALL_STAGES.keys())
        
        for stage in stages:
            print(f"\n{'='*60}")
            print(f"STAGE: {stage.upper()}")
            print(f"{'='*60}")
            
            result = self.train_stage(stage)
            self.stage_results[stage] = result
            
            if not result["gate_passed"]:
                print(f"  WARNING: Gate did not pass after {result['total_reps']} reps")
                print(f"  Metric: {result['gate_metric']}")
            else:
                print(f"  GATE PASSED: {result['gate_metric']}")
        
        return self.stage_results
    
    def train_stage(self, stage: str) -> dict:
        """Train a single stage until gate passes or stalls."""
        corpus = ALL_STAGES[stage]
        tau = STAGE_TAU[stage]
        
        # Determine learn_type based on stage
        if stage in ("sensory", "babbling"):
            learn_type = "sensory"
        elif stage in ("phonics", "morphology"):
            learn_type = "sensory"
        else:
            learn_type = "language"
        
        reps = 20  # Start with 20 reps
        stall_rounds = 0
        prev_metric = 0
        total_reps = 0
        data_added = False
        
        while reps <= self.max_reps:
            # Train
            trained = self._train_corpus(corpus, reps, learn_type)
            total_reps += reps
            
            # Add word nodes after training
            self._add_stage_word_nodes(stage, corpus)
            
            # Check gate
            gate_result = self.gates.check_gate(stage, self.lnn)
            metric_value = gate_result.get("actual", 0)
            
            print(f"  Reps: {total_reps}, Gate: {gate_result['metric']}")
            
            if gate_result["passed"]:
                return {
                    "stage": stage,
                    "gate_passed": True,
                    "gate_metric": gate_result["metric"],
                    "total_reps": total_reps,
                    "data_added": data_added,
                }
            
            # Check for stall (no improvement)
            if metric_value <= prev_metric:
                stall_rounds += 1
                if stall_rounds >= self.max_stall:
                    # Add more training data
                    if not data_added:
                        print(f"  Gate stalled, adding more training data...")
                        self._add_more_data(stage, corpus)
                        data_added = True
                        stall_rounds = 0
                    else:
                        print(f"  Gate stalled even with more data, stopping")
                        break
            else:
                stall_rounds = 0
            
            prev_metric = metric_value
            reps += 20  # Increase reps
        
        return {
            "stage": stage,
            "gate_passed": False,
            "gate_metric": gate_result.get("metric", "N/A"),
            "total_reps": total_reps,
            "data_added": data_added,
        }
    
    def _train_corpus(self, corpus, reps: int, learn_type: str):
        """Train on a corpus."""
        if isinstance(corpus, list):
            for _ in range(reps):
                for text in corpus:
                    learn_from_text(self.lnn, text, repetitions=1, learn_type=learn_type)
        elif isinstance(corpus, dict):
            for category, texts in corpus.items():
                if isinstance(texts, list):
                    for _ in range(reps):
                        for text in texts:
                            learn_from_text(self.lnn, text, repetitions=1, learn_type=learn_type)
                elif isinstance(texts, str):
                    for _ in range(reps):
                        learn_from_text(self.lnn, texts, repetitions=1, learn_type=learn_type)
    
    def _add_stage_word_nodes(self, stage: str, corpus):
        """Add word-level nodes for a stage."""
        sentences = []
        if isinstance(corpus, list):
            sentences = corpus
        elif isinstance(corpus, dict):
            for texts in corpus.values():
                if isinstance(texts, list):
                    sentences.extend(texts)
                elif isinstance(texts, str):
                    sentences.append(texts)
        
        if sentences:
            add_word_nodes(self.lnn, sentences)
    
    def _add_more_data(self, stage: str, corpus):
        """Add more training data when gate stalls."""
        # Double the corpus by repeating with variations
        if isinstance(corpus, list):
            # Add reversed versions
            for text in corpus[:]:
                reversed_text = text[::-1]
                corpus.append(reversed_text)
        elif isinstance(corpus, dict):
            for category, texts in corpus.items():
                if isinstance(texts, list):
                    for text in texts[:]:
                        reversed_text = text[::-1]
                        corpus[category].append(reversed_text)
    
    def get_summary(self) -> dict:
        """Get training summary."""
        gate_summary = self.gates.get_summary()
        
        return {
            "stages": self.stage_results,
            "gates": gate_summary,
            "total_nodes": len(self.lnn.nodes),
            "total_springs": len(self.lnn.springs),
        }
