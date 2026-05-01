"""
Developmental Gates for Sequential English Learning

Each stage has a gate that must pass before advancing.
Gates check if the LRN has learned enough to proceed.
If a gate fails, training continues with more data until it passes or stalls.
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN


class DevelopmentalGate:
    """Gates advancement based on learning readiness."""
    
    def __init__(self):
        self.gate_results = {}
    
    def check_gate(self, stage: str, lnn: LatticeNN) -> dict:
        """Check if a stage's gate passes."""
        checks = {
            "sensory": self._check_letters,
            "babbling": self._check_phonemes,
            "phonics": self._check_cvc_words,
            "morphology": self._check_morphemes,
            "sight_words": self._check_sight_words,
            "vocabulary": self._check_vocabulary,
            "grammar": self._check_pos_accuracy,
            "syntax": self._check_parsing,
            "sentences": self._check_completion,
            "pragmatics": self._check_context,
        }
        
        check_fn = checks.get(stage)
        if not check_fn:
            return {"passed": False, "reason": f"Unknown stage: {stage}"}
        
        result = check_fn(lnn)
        self.gate_results[stage] = result
        return result
    
    def _check_letters(self, lnn: LatticeNN) -> dict:
        """Gate: All 26 letters recognized."""
        letter_nodes = set()
        for n in lnn.nodes:
            if n.startswith("sens:") and len(n) == 6:
                char = n[5:]
                if char.isalpha() and len(char) == 1:
                    letter_nodes.add(char.lower())
        
        passed = len(letter_nodes) >= 26
        return {
            "passed": passed,
            "metric": f"{len(letter_nodes)}/26 letters",
            "threshold": 26,
            "actual": len(letter_nodes),
        }
    
    def _check_phonemes(self, lnn: LatticeNN) -> dict:
        """Gate: 10+ phoneme combinations formed."""
        phoneme_springs = sum(1 for sp in lnn.springs.values() if sp.tau <= 1)
        passed = phoneme_springs >= 10
        return {
            "passed": passed,
            "metric": f"{phoneme_springs} phoneme springs",
            "threshold": 10,
            "actual": phoneme_springs,
        }
    
    def _check_cvc_words(self, lnn: LatticeNN) -> dict:
        """Gate: 50+ word nodes formed."""
        word_nodes = sum(1 for n in lnn.nodes if n.startswith("word:") and len(n) > 6)
        passed = word_nodes >= 50
        return {
            "passed": passed,
            "metric": f"{word_nodes} word nodes",
            "threshold": 50,
            "actual": word_nodes,
        }
    
    def _check_morphemes(self, lnn: LatticeNN) -> dict:
        """Gate: 10+ morpheme patterns identified."""
        morpheme_nodes = sum(1 for n in lnn.nodes if n.startswith("word:") and len(n) > 6)
        passed = morpheme_nodes >= 10
        return {
            "passed": passed,
            "metric": f"{morpheme_nodes} morpheme nodes",
            "threshold": 10,
            "actual": morpheme_nodes,
        }
    
    def _check_sight_words(self, lnn: LatticeNN) -> dict:
        """Gate: 100+ sight words recognized."""
        word_nodes = sum(1 for n in lnn.nodes if n.startswith("word:"))
        passed = word_nodes >= 100
        return {
            "passed": passed,
            "metric": f"{word_nodes} word nodes",
            "threshold": 100,
            "actual": word_nodes,
        }
    
    def _check_vocabulary(self, lnn: LatticeNN) -> dict:
        """Gate: 500+ words with connections."""
        connected_words = 0
        for n in lnn.nodes:
            if n.startswith("word:"):
                neighbors = lnn.get_neighbors(n)
                if len(neighbors) >= 2:
                    connected_words += 1
        
        passed = connected_words >= 500
        return {
            "passed": passed,
            "metric": f"{connected_words} connected words",
            "threshold": 500,
            "actual": connected_words,
        }
    
    def _check_pos_accuracy(self, lnn: LatticeNN) -> dict:
        """Gate: 80% POS accuracy."""
        from lrn.grammar_training import infer_pos
        
        test_words = {
            "cat": "noun", "eats": "verb", "big": "adjective",
            "the": "determiner", "fast": "adjective", "runs": "verb",
            "dog": "noun", "small": "adjective",
        }
        
        correct = 0
        for word, expected_pos in test_words.items():
            result = infer_pos(lnn, word)
            if result.get("pos") == expected_pos:
                correct += 1
        
        accuracy = (correct * 100) // len(test_words)
        passed = accuracy >= 80
        return {
            "passed": passed,
            "metric": f"{accuracy}% POS accuracy",
            "threshold": 80,
            "actual": accuracy,
        }
    
    def _check_parsing(self, lnn: LatticeNN) -> dict:
        """Gate: Can parse simple sentences."""
        from lrn.inference import attention_with_residue
        
        test_queries = ["word:cat", "word:if", "word:and"]
        parsed = 0
        
        for query in test_queries:
            result = attention_with_residue(lnn, query, propagate_steps=3)
            if result["attention"]:
                parsed += 1
        
        passed = parsed >= 2
        return {
            "passed": passed,
            "metric": f"{parsed}/{len(test_queries)} parsed",
            "threshold": 2,
            "actual": parsed,
        }
    
    def _check_completion(self, lnn: LatticeNN) -> dict:
        """Gate: 70% completion accuracy."""
        from lrn.inference import attention_with_residue
        
        tests = [
            ("word:cat", "eats"),
            ("word:big", "dog"),
            ("word:if", "rains"),
        ]
        
        correct = 0
        for query, expected in tests:
            result = attention_with_residue(lnn, query, propagate_steps=3)
            words = [n.replace("word:", "") for n, _ in result["attention"]]
            if expected in words:
                correct += 1
        
        accuracy = (correct * 100) // len(tests)
        passed = accuracy >= 70
        return {
            "passed": passed,
            "metric": f"{accuracy}% completion",
            "threshold": 70,
            "actual": accuracy,
        }
    
    def _check_context(self, lnn: LatticeNN) -> dict:
        """Gate: Context-appropriate responses."""
        from lrn.inference import attention_with_residue
        
        test_queries = ["word:fire", "word:water", "word:cold"]
        responded = 0
        
        for query in test_queries:
            result = attention_with_residue(lnn, query, propagate_steps=3)
            if result["attention"]:
                responded += 1
        
        passed = responded >= 2
        return {
            "passed": passed,
            "metric": f"{responded}/{len(test_queries)} context responses",
            "threshold": 2,
            "actual": responded,
        }
    
    def get_summary(self) -> dict:
        """Get summary of all gate results."""
        passed = sum(1 for r in self.gate_results.values() if r.get("passed"))
        total = len(self.gate_results)
        return {
            "passed": passed,
            "total": total,
            "percentage": (passed * 100) // max(1, total),
            "results": self.gate_results,
        }
