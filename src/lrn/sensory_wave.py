"""
Sensory Wave Model v2 - Semantic labels for grounded wave patterns

The wave pattern IS the meaning. The word is just a label pointing to it.
Similarity between concepts = similarity between their wave signatures.
No co-occurrence learning needed — the geometry of the waves encodes semantics.

Wave signature = (amplitude, frequency, phase, rise_time, peak_time, fall_time, cooldown)
Two concepts are "similar" if their wave signatures produce similar activation patterns.
"""
import sys
import math
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.charts import progress_bar


class SensoryWave:
    """A sensory event with wave properties.
    
    The wave IS the concept. The label is just a pointer.
    """
    
    def __init__(self, concept, amplitude=1.0, frequency=1.0, phase=0.0,
                 rise_time=3, peak_time=4, fall_time=5, cooldown=10):
        self.concept = concept
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.rise_time = rise_time
        self.peak_time = peak_time
        self.fall_time = fall_time
        self.cooldown = cooldown
        self.cycle_length = rise_time + peak_time + fall_time + cooldown
    
    def value(self, tick):
        """Get wave value at a given tick."""
        cycle_length = self.rise_time + self.peak_time + self.fall_time + self.cooldown
        total_active = self.rise_time + self.peak_time + self.fall_time
        
        cycle_pos = (tick - int(self.phase * cycle_length)) % cycle_length
        if cycle_pos < 0:
            cycle_pos += cycle_length
        
        if cycle_pos < self.rise_time:
            t = cycle_pos / self.rise_time
            envelope = 1.0 - math.exp(-5 * t)
        elif cycle_pos < self.rise_time + self.peak_time:
            envelope = 1.0
        elif cycle_pos < total_active:
            t = (cycle_pos - self.rise_time - self.peak_time) / self.fall_time
            envelope = math.exp(-5 * t)
        else:
            return 0.0
        
        return self.amplitude * envelope
    
    def signature(self):
        """Return the wave signature as a tuple."""
        return (self.amplitude, self.frequency, self.phase,
                self.rise_time, self.peak_time, self.fall_time, self.cooldown)


def wave_similarity(wave_a, wave_b):
    """Compute similarity between two wave signatures.
    
    Returns 0-1 score based on:
    - Amplitude similarity (how intense)
    - Phase alignment (when they fire)
    - Shape similarity (rise/peak/fall timing)
    - Frequency similarity (how often they repeat)
    """
    # Amplitude similarity
    amp_sim = 1.0 - abs(wave_a.amplitude - wave_b.amplitude)
    
    # Phase alignment (circular distance)
    phase_diff = abs(wave_a.phase - wave_b.phase)
    phase_diff = min(phase_diff, 1.0 - phase_diff)  # wrap around
    phase_sim = 1.0 - phase_diff * 2  # 0.5 apart = 0 similarity
    
    # Shape similarity (rise/peak/fall ratios)
    shape_a = (wave_a.rise_time, wave_a.peak_time, wave_a.fall_time)
    shape_b = (wave_b.rise_time, wave_b.peak_time, wave_b.fall_time)
    max_val = max(sum(shape_a), sum(shape_b), 1)
    shape_diff = sum(abs(a - b) for a, b in zip(shape_a, shape_b)) / max_val
    shape_sim = 1.0 - shape_diff
    
    # Frequency similarity
    freq_sim = 1.0 - abs(wave_a.frequency - wave_b.frequency)
    
    # Weighted combination
    # Phase alignment is most important for semantic similarity
    # Shape determines the "feel" of the concept
    # Amplitude determines intensity
    # Frequency determines rhythm
    return (phase_sim * 0.4 + shape_sim * 0.3 + amp_sim * 0.15 + freq_sim * 0.15)


def ground_waves(lnn, concepts, verbose=True):
    """Ground concepts as sensory waves.
    
    Springs form based on wave signature similarity, NOT co-occurrence.
    This means the lattice learns semantic relationships directly from
    the geometry of the sensory experience.
    """
    waves = []
    for c in concepts:
        wave = SensoryWave(
            concept=c["name"],
            amplitude=c.get("amplitude", 1.0),
            frequency=c.get("frequency", 1.0),
            phase=c.get("phase", 0.0),
            rise_time=c.get("rise_time", 3),
            peak_time=c.get("peak_time", 4),
            fall_time=c.get("fall_time", 5),
            cooldown=c.get("cooldown", 10),
        )
        waves.append(wave)
    
    # Create springs based on wave similarity
    for i in range(len(waves)):
        for j in range(i + 1, len(waves)):
            sim = wave_similarity(waves[i], waves[j])
            
            if sim > 0.3:  # Only connect if meaningfully similar
                stiffness = int(sim * 100)
                # Tau based on similarity strength:
                # >0.8: τ=1 (definitional — nearly identical waves)
                # >0.6: τ=2 (causal — strongly related)
                # >0.4: τ=3 (categorical — same category)
                # >0.3: τ=4 (contextual — loosely related)
                if sim > 0.8:
                    tau = 1
                elif sim > 0.6:
                    tau = 2
                elif sim > 0.4:
                    tau = 3
                else:
                    tau = 4
                
                lnn.add_or_update_spring(
                    f"word:{waves[i].concept}",
                    f"word:{waves[j].concept}",
                    stiffness=stiffness,
                    tau=tau,
                    mode="add"
                )
    
    # Also run a brief temporal simulation to capture dynamic overlap
    # This adds a small bonus for waves that actually fire together
    # (constructive interference) and penalty for waves that never overlap
    # (destructive interference)
    total_ticks = sum(w.cycle_length for w in waves) // len(waves) * 10
    
    for tick in range(total_ticks):
        active = [w for w in waves if w.value(tick) > 0.01]
        
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                key = lnn._key(f"word:{active[i].concept}", f"word:{active[j].concept}")
                if key in lnn.springs:
                    sp = lnn.springs[key]
                    # Bonus for constructive overlap
                    overlap = active[i].value(tick) * active[j].value(tick)
                    sp.stiffness += int(overlap * 5)
        
        # Anti-co-occurrence: decay springs for concepts that never fire together
        active_names = set(w.concept for w in active)
        for w in waves:
            if w.concept not in active_names:
                for other in active:
                    key = lnn._key(f"word:{w.concept}", f"word:{other.concept}")
                    if key in lnn.springs:
                        sp = lnn.springs[key]
                        sp.stiffness = max(1, sp.stiffness - 1)
    
    propagate(lnn, n_steps=3)
    
    return lnn


def create_lnn():
    """Create a new LatticeNN with identity anchor."""
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    return lnn
