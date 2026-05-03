"""
Harmonic Video Model - Attaching semantic labels to pre-grounded concepts

The lattice already KNOWS concepts through sensory grounding:
- hot/cold from thermal wave patterns
- big/small from weight wave patterns  
- fast/slow from kinesthetic wave patterns
- red/blue from electromagnetic wave patterns
- b/d from visual geometry wave patterns

The harmonic video doesn't teach concepts. It attaches WORDS to them.
"This feeling you know? We call it 'hot'."

Modality streams phase-lock the label onto the existing sensory pattern:
- VISUAL: the word "hot" appears in red text on screen
- AUDIO: someone says "hot" 
- EMOTIONAL: the feeling of heat (already grounded)
- RHYTHM: the syllable timing

When all modalities converge on the label at the same phase as the
sensory wave → the word becomes the pointer to the concept.
"""
import sys
import math
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, add_identity_anchor, propagate
from lrn.charts import progress_bar


class ModalityStream:
    """A single sensory channel carrying a concept label."""
    
    def __init__(self, concept, modality, amplitude=1.0, frequency=1.0, phase=0.0,
                 rise_time=2, peak_time=3, fall_time=2, cooldown=5):
        self.concept = concept
        self.modality = modality
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.rise_time = rise_time
        self.peak_time = peak_time
        self.fall_time = fall_time
        self.cooldown = cooldown
    
    def value(self, tick):
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


class HarmonicLesson:
    """A segment that attaches labels to pre-grounded concepts."""
    
    def __init__(self, name, streams, duration_ticks=100):
        self.name = name
        self.streams = streams
        self.duration_ticks = duration_ticks
    
    def get_active_concepts(self, tick):
        active = {}
        for stream in self.streams:
            val = stream.value(tick)
            if val > 0.01:
                if stream.concept not in active:
                    active[stream.concept] = {}
                active[stream.concept][stream.modality] = val
        return active


def harmonic_convergence(lnn, lesson, verbose=True):
    """Run a harmonic lesson.
    
    Key insight: the lattice already has sensory-grounded springs.
    This lesson creates word nodes and links them to the existing
    sensory patterns through multi-modal phase alignment.
    
    Spring strength = sum(modalities) × harmonic_multiplier
    Where harmonic_multiplier = number of aligned modalities
    """
    total_ticks = lesson.duration_ticks
    
    for tick in range(total_ticks):
        active = lesson.get_active_concepts(tick)
        
        if not active:
            continue
        
        concepts = list(active.keys())
        
        for concept, modalities in active.items():
            total_activation = sum(modalities.values())
            modality_count = len(modalities)
            
            node_name = f"word:{concept}"
            if node_name not in lnn.nodes:
                lnn.add_node(node_name)
            
            # Cross-modal nodes
            for modality in modalities:
                modal_node = f"{modality}:{concept}"
                if modal_node not in lnn.nodes:
                    lnn.add_node(modal_node)
                
                key = lnn._key(node_name, modal_node)
                if key not in lnn.springs:
                    lnn.add_or_update_spring(node_name, modal_node, 
                                            stiffness=int(total_activation * 20), 
                                            tau=1, mode="add")
                else:
                    sp = lnn.springs[key]
                    sp.stiffness += int(total_activation * 5)
            
            # Springs between co-active concepts
            for i in range(len(concepts)):
                for j in range(i + 1, len(concepts)):
                    a, b = concepts[i], concepts[j]
                    a_total = sum(active[a].values())
                    b_total = sum(active[b].values())
                    
                    a_mods = len(active[a])
                    b_mods = len(active[b])
                    harmonic_mult = (a_mods + b_mods) / 2
                    
                    reinforcement = int(a_total * b_total * harmonic_mult * 10)
                    
                    key = lnn._key(f"word:{a}", f"word:{b}")
                    if key in lnn.springs:
                        sp = lnn.springs[key]
                        sp.stiffness += reinforcement
                        sp.exposure_count += 1
                    else:
                        lnn.add_or_update_spring(f"word:{a}", f"word:{b}",
                                                stiffness=reinforcement, tau=3, mode="add")
    
    propagate(lnn, n_steps=3)
    return lnn


def make_label_lesson(concepts, name, duration_ticks=150):
    """Create a lesson that attaches labels to pre-grounded concepts.
    
    Each concept gets 4 modalities phase-aligned:
    - visual: the word appears on screen
    - audio: the word is spoken
    - emotional: the feeling associated with the concept
    - rhythm: the syllable beat
    
    Concepts are presented sequentially, each getting a time slot.
    """
    streams = []
    
    for i, concept in enumerate(concepts):
        phase = i / len(concepts)
        
        # Visual - word appears on screen
        streams.append(ModalityStream(
            concept=concept, modality="visual",
            amplitude=0.9, frequency=0.8, phase=phase,
            rise_time=1, peak_time=4, fall_time=1, cooldown=3
        ))
        
        # Audio - word is spoken
        streams.append(ModalityStream(
            concept=concept, modality="audio",
            amplitude=0.9, frequency=0.8, phase=phase,
            rise_time=1, peak_time=4, fall_time=1, cooldown=3
        ))
        
        # Emotional - feeling associated with concept
        streams.append(ModalityStream(
            concept=concept, modality="emotional",
            amplitude=0.6, frequency=0.8, phase=phase,
            rise_time=2, peak_time=3, fall_time=2, cooldown=3
        ))
        
        # Rhythm - syllable timing
        streams.append(ModalityStream(
            concept=concept, modality="rhythm",
            amplitude=0.7, frequency=0.8, phase=phase,
            rise_time=1, peak_time=2, fall_time=1, cooldown=4
        ))
    
    return HarmonicLesson(name, streams, duration_ticks=duration_ticks)


def harmonic_training(lnn, lessons, verbose=True):
    """Run multiple harmonic lessons through the lattice."""
    for i, lesson in enumerate(lessons):
        if verbose:
            print(f"  Lesson {i+1}/{len(lessons)}: {lesson.name}")
        harmonic_convergence(lnn, lesson, verbose=False)
    
    propagate(lnn, n_steps=5)
    return lnn


def create_lnn():
    """Create a new LatticeNN with identity anchor."""
    lnn = LatticeNN()
    add_identity_anchor(lnn)
    return lnn
