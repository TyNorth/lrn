"""
Sequential English Training Pipeline

Trains the LRN through developmental stages:
1. Sensory/Letters → 2. Babbling → 3. Phonics → 4. Morphology →
5. Sight Words → 6. Vocabulary → 7. Grammar → 8. Syntax →
9. Sentences → 10. Pragmatics

Each stage trains until its developmental gate passes.
After each sentence: REM synthesis forms τ=3 categorical bridges (gravity effect).
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.natural_tokenize import learn_from_text, TAU_BY_TYPE
from lrn.inference import add_word_nodes
from lrn.english_corpus import ALL_STAGES
from lrn.english_gates import DevelopmentalGate
from lrn.rem_synthesis import REMSleep


# Tau assignment per stage
STAGE_TAU = {
    "sensory": 0,      # Letters - constitutive
    "babbling": 1,     # Sound exploration - definitional
    "phonics": 1,      # Letter→sound mapping - definitional
    "morphology": 1,   # Word parts - definitional
    "sight_words": 2,  # Instant recognition - causal
    "vocabulary": 2,   # Word meanings - causal
    "grammar": 2,      # POS, word order - causal
    "syntax": 3,       # Hierarchical structure - categorical
    "sentences": 3,    # Full comprehension - categorical
    "pragmatics": 4,   # Social context - contextual
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
        self.wake_buffer = []  # Last 5 sentences for REM wake context
    
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
        gate_result = {"passed": False, "metric": "N/A", "actual": 0}
        
        while reps <= self.max_reps:
            # Train with REM synthesis after each sentence
            trained = self._train_corpus_with_rem(corpus, reps, learn_type, stage)
            total_reps += reps
            
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
    
    def _train_corpus_with_rem(self, corpus, reps: int, learn_type: str, stage: str):
        """Train on a corpus with REM synthesis after full corpus passes."""
        sentences = self._flatten_corpus(corpus)
        sentence_count = 0
        
        for rep in range(reps):
            for sentence in sentences:
                # 1. Train sentence
                learn_from_text(self.lnn, sentence, repetitions=1, learn_type=learn_type)
                
                # 2. Add word nodes for this sentence
                add_word_nodes(self.lnn, [sentence])
                
                sentence_count += 1
            
            # REM synthesis after each full corpus pass
            if sentences:
                self._run_rem_synthesis(sentences[-1])
                propagate(self.lnn, n_steps=3)
        
        # Final REM and relaxation
        if sentences:
            self._run_rem_synthesis(sentences[-1])
            propagate(self.lnn, n_steps=3)
    
    def _run_rem_synthesis(self, current_sentence: str):
        """Run lightweight REM synthesis - forms τ=3 bridges between co-occurring words."""
        # Update wake buffer
        self.wake_buffer.append(current_sentence)
        if len(self.wake_buffer) > 5:
            self.wake_buffer = self.wake_buffer[-5:]
        
        # Get words from recent sentences
        recent_words = set()
        for sentence in self.wake_buffer:
            for word in sentence.lower().split():
                recent_words.add(f"word:{word}")
        
        # Form τ=3 categorical bridges between all co-occurring words
        # This creates the "gravity" effect - words in same context cluster together
        word_list = list(recent_words)
        for i in range(len(word_list)):
            for j in range(i + 1, len(word_list)):
                a, b = word_list[i], word_list[j]
                key = self.lnn._key(a, b)
                
                if key in self.lnn.springs:
                    # Promote existing spring to τ=3 (categorical)
                    existing = self.lnn.springs[key]
                    if existing.tau > 2:
                        existing.tau = 3
                        existing.stiffness = max(existing.stiffness, 10)
                else:
                    # Create new τ=3 categorical bridge
                    self.lnn.add_or_update_spring(a, b, stiffness=10, tau=3, mode="add")
    
    def _flatten_corpus(self, corpus) -> list:
        """Flatten corpus into list of sentences."""
        sentences = []
        if isinstance(corpus, list):
            sentences = corpus
        elif isinstance(corpus, dict):
            for texts in corpus.values():
                if isinstance(texts, list):
                    sentences.extend(texts)
                elif isinstance(texts, str):
                    sentences.append(texts)
        return sentences
    
    def _add_stage_word_nodes(self, stage: str, corpus):
        """Add word-level nodes for a stage."""
        sentences = self._flatten_corpus(corpus)
        if sentences:
            add_word_nodes(self.lnn, sentences)
    
    def _add_more_data(self, stage: str, corpus):
        """Add more training data when gate stalls."""
        # Double the corpus by repeating with variations
        if isinstance(corpus, list):
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
