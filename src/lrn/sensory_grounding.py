"""
Sensory Grounding - Multi-modal concepts grounded before text training

Training pipeline order:
1. INTEROCEPTIVE - internal body states (hunger, comfort, heartbeat, breath)
   The FIRST sense. A newborn experiences the world from inside out.
2. EXTERNAL SENSORY - colors, letters, shapes, temperature, texture, sound
   The world outside the body, perceived through senses.
3. TEXT CORPUS - Pre-K and beyond, building on sensory foundation.

Grounded concepts:
- Interoceptive: hunger, thirst, comfort, pain, heartbeat, breath, fatigue, warmth
- Colors: warm/cool phase alignment (electromagnetic spectrum)
- Letters: visual geometry (strokes, curves, loops, intersections)
- Shapes: geometric properties (sides, angles, symmetry)
- Temperature: thermal spectrum (hot → warm → cool → cold)
- Texture: tactile spectrum (soft/smooth ↔ rough/hard/sticky)
- Sound: auditory spectrum (loud/quiet, high/low pitch)
- Taste: gustatory categories (sweet/sour/salty/bitter)
- Smell: olfactory categories (fragrant/stinky)
- Weight: proprioceptive spectrum (heavy/light)
- Speed: kinesthetic spectrum (fast/slow)
- Distance: spatial spectrum (near/far)
- Brightness: visual spectrum (bright/dim/dark)
"""
import sys
import math
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn.sensory_wave import ground_waves, create_lnn, SensoryWave


def ground_interoceptive(lnn, verbose=True):
    """Ground interoceptive sensations — the FIRST sense.
    
    Interoception is the sense of the internal state of the body.
    Before a baby can see colors or hear sounds, it feels:
    - Hunger pangs (rhythmic, urgent, builds to peak)
    - Fullness/comfort (slow, warm, lingering)
    - Heartbeat (rhythmic, constant, baseline)
    - Breath (cyclical, gentle, life-sustaining)
    - Fatigue/sleepiness (gradual onset, heavy, slow decay)
    - Pain/discomfort (sharp, sudden, demands attention)
    - Warmth from within (steady, comforting, baseline)
    - Energy/alertness (burst, quick, fades)
    
    These form the foundational layer that all other senses build on.
    Emotions are later mapped onto these interoceptive patterns.
    """
    if verbose:
        print("  Grounding: INTEROCEPTIVE (internal body states)")
    
    concepts = [
        # Heartbeat — the first rhythm, constant baseline
        # Fast, rhythmic, never stops
        {"name": "heartbeat", "amplitude": 0.6, "frequency": 0.95, "phase": 0.0,
         "rise_time": 1, "peak_time": 1, "fall_time": 1, "cooldown": 2},
        
        # Breath — cyclical, gentle, life-sustaining
        # Slower than heartbeat, rhythmic but variable
        {"name": "breath", "amplitude": 0.5, "frequency": 0.7, "phase": 0.1,
         "rise_time": 3, "peak_time": 2, "fall_time": 3, "cooldown": 4},
        
        # Hunger — builds slowly, peaks urgently, only relieved by eating
        # Slow rise, sharp peak, slow decay (lingers until fed)
        {"name": "hungry", "amplitude": 0.9, "frequency": 0.3, "phase": 0.3,
         "rise_time": 6, "peak_time": 4, "fall_time": 2, "cooldown": 20},
        {"name": "full", "amplitude": 0.7, "frequency": 0.3, "phase": 0.35,
         "rise_time": 2, "peak_time": 8, "fall_time": 6, "cooldown": 20},
        
        # Thirst — similar to hunger but sharper, more urgent
        {"name": "thirsty", "amplitude": 0.8, "frequency": 0.3, "phase": 0.4,
         "rise_time": 4, "peak_time": 3, "fall_time": 2, "cooldown": 22},
        
        # Comfort — warm, slow, overlapping with full and breath
        # The baseline state when needs are met
        {"name": "comfort", "amplitude": 0.7, "frequency": 0.5, "phase": 0.15,
         "rise_time": 5, "peak_time": 8, "fall_time": 5, "cooldown": 15},
        {"name": "warmth", "amplitude": 0.6, "frequency": 0.5, "phase": 0.17,
         "rise_time": 5, "peak_time": 8, "fall_time": 5, "cooldown": 15},
        
        # Pain/discomfort — sharp, sudden, demands attention
        # Fast onset, slow decay (pain lingers)
        {"name": "pain", "amplitude": 0.95, "frequency": 0.2, "phase": 0.7,
         "rise_time": 1, "peak_time": 2, "fall_time": 6, "cooldown": 25},
        {"name": "tired", "amplitude": 0.8, "frequency": 0.25, "phase": 0.8,
         "rise_time": 8, "peak_time": 6, "fall_time": 4, "cooldown": 20},
        
        # Energy/alertness — bursts, quick, fades
        {"name": "energy", "amplitude": 0.8, "frequency": 0.4, "phase": 0.5,
         "rise_time": 2, "peak_time": 3, "fall_time": 3, "cooldown": 12},
        {"name": "awake", "amplitude": 0.7, "frequency": 0.4, "phase": 0.52,
         "rise_time": 2, "peak_time": 4, "fall_time": 3, "cooldown": 12},
    ]
    
    ground_waves(lnn, concepts, verbose=False)
    
    if verbose:
        pairs = [
            ("heartbeat", "breath"), ("hungry", "full"), ("comfort", "warmth"),
            ("pain", "tired"), ("energy", "awake"), ("hungry", "pain"),
            ("comfort", "full"), ("heartbeat", "comfort"), ("breath", "warmth"),
        ]
        print(f"    Interoceptive pairs:")
        for a, b in pairs:
            key = lnn._key(f"word:{a}", f"word:{b}")
            if key in lnn.springs:
                print(f"      {a}-{b}: {lnn.springs[key].stiffness}")
            else:
                print(f"      {a}-{b}: (no spring)")
    
    return lnn


def ground_colors(lnn, verbose=True):
    """Ground colors as electromagnetic sensory events.
    
    Warm colors (red/orange/yellow) share phase → constructive interference
    Cool colors (blue/green/purple) share phase → constructive interference
    Warm vs cool are out of phase → destructive interference
    Neutrals (black/white/brown/pink) have independent timing
    """
    if verbose:
        print("  Grounding: COLORS (electromagnetic spectrum)")
    
    concepts = [
        # Warm colors - fire/sunset context, phase-aligned
        {"name": "red", "amplitude": 0.9, "frequency": 0.8, "phase": 0.0,
         "rise_time": 3, "peak_time": 5, "fall_time": 4, "cooldown": 12},
        {"name": "orange", "amplitude": 0.8, "frequency": 0.8, "phase": 0.02,
         "rise_time": 3, "peak_time": 5, "fall_time": 4, "cooldown": 12},
        {"name": "yellow", "amplitude": 0.9, "frequency": 0.8, "phase": 0.04,
         "rise_time": 3, "peak_time": 5, "fall_time": 4, "cooldown": 12},
        
        # Cool colors - sky/water context, phase-aligned with each other
        # Phase 0.5 = half cycle offset from warm → minimal temporal overlap
        {"name": "blue", "amplitude": 0.9, "frequency": 0.7, "phase": 0.5,
         "rise_time": 4, "peak_time": 6, "fall_time": 5, "cooldown": 15},
        {"name": "green", "amplitude": 0.8, "frequency": 0.7, "phase": 0.52,
         "rise_time": 4, "peak_time": 6, "fall_time": 5, "cooldown": 15},
        {"name": "purple", "amplitude": 0.7, "frequency": 0.7, "phase": 0.54,
         "rise_time": 4, "peak_time": 6, "fall_time": 5, "cooldown": 15},
        
        # Neutrals - different contexts, low frequency, high cooldown
        {"name": "black", "amplitude": 0.4, "frequency": 0.3, "phase": 0.25,
         "rise_time": 2, "peak_time": 3, "fall_time": 3, "cooldown": 25},
        {"name": "white", "amplitude": 0.4, "frequency": 0.3, "phase": 0.27,
         "rise_time": 2, "peak_time": 3, "fall_time": 3, "cooldown": 25},
        {"name": "brown", "amplitude": 0.5, "frequency": 0.4, "phase": 0.75,
         "rise_time": 3, "peak_time": 4, "fall_time": 4, "cooldown": 20},
        {"name": "pink", "amplitude": 0.3, "frequency": 0.2, "phase": 0.1,
         "rise_time": 2, "peak_time": 2, "fall_time": 2, "cooldown": 30},
    ]
    
    ground_waves(lnn, concepts, verbose=False)
    
    # Report
    warm = {"red", "orange", "yellow"}
    cool = {"blue", "green", "purple"}
    warm_internal = 0
    cool_internal = 0
    cross = 0
    
    colors = [c["name"] for c in concepts]
    for i in range(len(colors)):
        for j in range(i+1, len(colors)):
            key = lnn._key(f"word:{colors[i]}", f"word:{colors[j]}")
            if key in lnn.springs:
                sp = lnn.springs[key]
                a, b = colors[i], colors[j]
                if a in warm and b in warm:
                    warm_internal += sp.stiffness
                elif a in cool and b in cool:
                    cool_internal += sp.stiffness
                else:
                    cross += sp.stiffness
    
    total = warm_internal + cool_internal + cross
    if verbose:
        print(f"    Warm cluster: {warm_internal} ({warm_internal*100//max(1,total)}%)")
        print(f"    Cool cluster: {cool_internal} ({cool_internal*100//max(1,total)}%)")
        print(f"    Cross-cluster: {cross} ({cross*100//max(1,total)}%)")
    
    return lnn


def ground_letters(lnn, verbose=True):
    """Ground letters as visual geometric patterns.
    
    Letters sharing visual features fire in phase:
    - Vertical ascenders: b, d, f, h, k, l, t
    - Closed loops: a, b, d, e, g, o, p, q
    - Curve-only: c, o, s, u
    - Diagonal: k, v, w, x, y, z
    - Descenders: g, j, p, q, y
    - Horizontal: e, f, t, z
    - Simple (1-2 strokes): c, i, j, l, o, s, u, v
    
    Letters in multiple groups get higher amplitude (more visual features).
    """
    if verbose:
        print("  Grounding: LETTERS (visual geometry)")
    
    # Visual feature groups
    groups = {
        "vertical_ascenders": {"b": 0.0, "d": 0.02, "f": 0.04, "h": 0.01, "k": 0.03, "l": 0.0, "t": 0.05},
        "closed_loop": {"a": 0.3, "b": 0.31, "d": 0.32, "e": 0.33, "g": 0.34, "o": 0.3, "p": 0.35, "q": 0.36},
        "curve_only": {"c": 0.6, "o": 0.61, "s": 0.62, "u": 0.63},
        "diagonal": {"k": 0.8, "v": 0.81, "w": 0.82, "x": 0.83, "y": 0.84, "z": 0.85},
        "descenders": {"g": 0.9, "j": 0.91, "p": 0.92, "q": 0.93, "y": 0.94},
        "horizontal": {"e": 0.15, "f": 0.16, "t": 0.17, "z": 0.18},
        "simple": {"c": 0.45, "i": 0.46, "j": 0.47, "l": 0.45, "o": 0.46, "s": 0.48, "u": 0.49, "v": 0.47},
    }
    
    # Merge: letters appear in multiple groups
    letter_data = {}
    for group_name, members in groups.items():
        for letter, phase in members.items():
            if letter not in letter_data:
                letter_data[letter] = {"phases": [], "count": 0}
            letter_data[letter]["phases"].append(phase)
            letter_data[letter]["count"] += 1
    
    # Build concepts
    concepts = []
    for letter, data in letter_data.items():
        avg_phase = sum(data["phases"]) / len(data["phases"])
        # More visual features = higher amplitude
        amplitude = min(1.0, 0.6 + 0.15 * data["count"])
        
        concepts.append({
            "name": letter,
            "amplitude": amplitude,
            "frequency": 0.8,
            "phase": avg_phase,
            "rise_time": 2,
            "peak_time": 4,
            "fall_time": 3,
            "cooldown": 10,
        })
    
    ground_waves(lnn, concepts, verbose=False)
    
    # Report
    if verbose:
        # Check visually similar pairs
        expected_similar = [
            ("b", "d"), ("p", "q"), ("g", "q"), ("c", "s"), ("w", "x"),
            ("b", "l"), ("d", "l"), ("f", "h"), ("g", "p"), ("a", "e"),
        ]
        print(f"    Visually similar pairs:")
        for a, b in expected_similar:
            key = lnn._key(f"word:{a}", f"word:{b}")
            if key in lnn.springs:
                print(f"      {a}-{b}: {lnn.springs[key].stiffness}")
            else:
                print(f"      {a}-{b}: (no spring)")
    
    return lnn


def ground_shapes(lnn, verbose=True):
    """Ground shapes as geometric sensory events.
    
    Shapes sharing geometric properties fire in phase:
    - Circles/ellipses: circle, oval (curved, no corners)
    - Polygons: triangle, square, rectangle, pentagon, hexagon (straight edges, corners)
    - 4-sided: square, rectangle (right angles)
    - 3-sided: triangle (sharp angles)
    """
    if verbose:
        print("  Grounding: SHAPES (geometry)")
    
    concepts = [
        # Curved shapes - no corners, smooth
        {"name": "circle", "amplitude": 0.9, "frequency": 0.8, "phase": 0.0,
         "rise_time": 3, "peak_time": 5, "fall_time": 3, "cooldown": 12},
        {"name": "oval", "amplitude": 0.8, "frequency": 0.8, "phase": 0.02,
         "rise_time": 3, "peak_time": 5, "fall_time": 3, "cooldown": 12},
        
        # Polygons - corners, straight edges
        {"name": "triangle", "amplitude": 0.8, "frequency": 0.7, "phase": 0.5,
         "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 10},
        {"name": "square", "amplitude": 0.9, "frequency": 0.7, "phase": 0.52,
         "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 10},
        {"name": "rectangle", "amplitude": 0.8, "frequency": 0.7, "phase": 0.54,
         "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 10},
        {"name": "pentagon", "amplitude": 0.7, "frequency": 0.7, "phase": 0.56,
         "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 10},
        {"name": "hexagon", "amplitude": 0.7, "frequency": 0.7, "phase": 0.58,
         "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 10},
        
        # Star - unique, sharp points
        {"name": "star", "amplitude": 0.6, "frequency": 0.5, "phase": 0.8,
         "rise_time": 2, "peak_time": 2, "fall_time": 2, "cooldown": 15},
        {"name": "heart", "amplitude": 0.6, "frequency": 0.5, "phase": 0.82,
         "rise_time": 2, "peak_time": 2, "fall_time": 2, "cooldown": 15},
    ]
    
    ground_waves(lnn, concepts, verbose=False)
    
    if verbose:
        shapes = [c["name"] for c in concepts]
        curved = {"circle", "oval"}
        polygon = {"triangle", "square", "rectangle", "pentagon", "hexagon"}
        
        curved_internal = 0
        polygon_internal = 0
        cross = 0
        
        for i in range(len(shapes)):
            for j in range(i+1, len(shapes)):
                key = lnn._key(f"word:{shapes[i]}", f"word:{shapes[j]}")
                if key in lnn.springs:
                    sp = lnn.springs[key]
                    a, b = shapes[i], shapes[j]
                    if a in curved and b in curved:
                        curved_internal += sp.stiffness
                    elif a in polygon and b in polygon:
                        polygon_internal += sp.stiffness
                    else:
                        cross += sp.stiffness
        
        total = curved_internal + polygon_internal + cross
        print(f"    Curved cluster: {curved_internal} ({curved_internal*100//max(1,total)}%)")
        print(f"    Polygon cluster: {polygon_internal} ({polygon_internal*100//max(1,total)}%)")
        print(f"    Cross-cluster: {cross} ({cross*100//max(1,total)}%)")
    
    return lnn


def ground_temperature(lnn, verbose=True):
    """Ground temperature as thermal sensation.
    
    Thermal spectrum: hot → warm → cool → cold
    Adjacent concepts fire in phase (warm overlaps with both hot and cool).
    Opposite ends (hot vs cold) are maximally out of phase.
    """
    if verbose:
        print("  Grounding: TEMPERATURE (thermal sensation)")
    
    concepts = [
        # Hot - intense thermal, fast onset, slow decay
        {"name": "hot", "amplitude": 0.95, "frequency": 0.6, "phase": 0.0,
         "rise_time": 2, "peak_time": 4, "fall_time": 6, "cooldown": 15},
        # Warm - moderate thermal, overlaps with hot
        {"name": "warm", "amplitude": 0.8, "frequency": 0.7, "phase": 0.15,
         "rise_time": 3, "peak_time": 5, "fall_time": 5, "cooldown": 12},
        # Cool - moderate, overlaps with cold
        {"name": "cool", "amplitude": 0.8, "frequency": 0.7, "phase": 0.6,
         "rise_time": 3, "peak_time": 5, "fall_time": 5, "cooldown": 12},
        # Cold - intense, slow onset, fast decay
        {"name": "cold", "amplitude": 0.95, "frequency": 0.6, "phase": 0.75,
         "rise_time": 4, "peak_time": 4, "fall_time": 3, "cooldown": 15},
        # Freezing - extreme cold
        {"name": "freezing", "amplitude": 0.9, "frequency": 0.5, "phase": 0.8,
         "rise_time": 4, "peak_time": 3, "fall_time": 2, "cooldown": 18},
        # Boiling - extreme heat
        {"name": "boiling", "amplitude": 0.9, "frequency": 0.5, "phase": 0.05,
         "rise_time": 2, "peak_time": 3, "fall_time": 4, "cooldown": 18},
    ]
    
    ground_waves(lnn, concepts, verbose=False)
    
    if verbose:
        pairs = [("hot", "warm"), ("warm", "cool"), ("cool", "cold"), ("hot", "cold"), ("freezing", "cold"), ("boiling", "hot")]
        print(f"    Thermal pairs:")
        for a, b in pairs:
            key = lnn._key(f"word:{a}", f"word:{b}")
            if key in lnn.springs:
                print(f"      {a}-{b}: {lnn.springs[key].stiffness}")
            else:
                print(f"      {a}-{b}: (no spring)")
    
    return lnn


def ground_texture(lnn, verbose=True):
    """Ground texture as tactile sensation.
    
    Tactile dimensions:
    - Softness: soft ↔ hard (pressure sensitivity)
    - Smoothness: smooth ↔ rough (surface friction)
    - Adhesion: sticky ↔ slippery (surface tension)
    
    Concepts sharing tactile properties fire in phase.
    """
    if verbose:
        print("  Grounding: TEXTURE (tactile sensation)")
    
    concepts = [
        # Soft things - gentle pressure, slow rise
        {"name": "soft", "amplitude": 0.8, "frequency": 0.7, "phase": 0.0,
         "rise_time": 4, "peak_time": 5, "fall_time": 4, "cooldown": 12},
        {"name": "fuzzy", "amplitude": 0.7, "frequency": 0.7, "phase": 0.02,
         "rise_time": 4, "peak_time": 5, "fall_time": 4, "cooldown": 12},
        {"name": "fluffy", "amplitude": 0.7, "frequency": 0.7, "phase": 0.04,
         "rise_time": 4, "peak_time": 5, "fall_time": 4, "cooldown": 12},
        
        # Hard things - sharp onset, quick decay
        {"name": "hard", "amplitude": 0.8, "frequency": 0.7, "phase": 0.5,
         "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 10},
        {"name": "rough", "amplitude": 0.7, "frequency": 0.7, "phase": 0.52,
         "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 10},
        {"name": "bumpy", "amplitude": 0.6, "frequency": 0.7, "phase": 0.54,
         "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 10},
        
        # Smooth things - moderate, overlaps with soft
        {"name": "smooth", "amplitude": 0.8, "frequency": 0.6, "phase": 0.2,
         "rise_time": 3, "peak_time": 5, "fall_time": 3, "cooldown": 14},
        {"name": "silky", "amplitude": 0.6, "frequency": 0.6, "phase": 0.22,
         "rise_time": 3, "peak_time": 5, "fall_time": 3, "cooldown": 14},
        
        # Sticky things - unique tactile signature
        {"name": "sticky", "amplitude": 0.7, "frequency": 0.5, "phase": 0.75,
         "rise_time": 3, "peak_time": 4, "fall_time": 5, "cooldown": 15},
        {"name": "wet", "amplitude": 0.6, "frequency": 0.5, "phase": 0.77,
         "rise_time": 3, "peak_time": 4, "fall_time": 5, "cooldown": 15},
    ]
    
    ground_waves(lnn, concepts, verbose=False)
    
    if verbose:
        pairs = [("soft", "fuzzy"), ("fuzzy", "fluffy"), ("hard", "rough"), ("smooth", "silky"), ("soft", "hard"), ("sticky", "wet")]
        print(f"    Tactile pairs:")
        for a, b in pairs:
            key = lnn._key(f"word:{a}", f"word:{b}")
            if key in lnn.springs:
                print(f"      {a}-{b}: {lnn.springs[key].stiffness}")
            else:
                print(f"      {a}-{b}: (no spring)")
    
    return lnn


def ground_sound(lnn, verbose=True):
    """Ground sound as auditory sensation.
    
    Auditory dimensions:
    - Volume: loud ↔ quiet (amplitude)
    - Pitch: high ↔ low (frequency)
    - Quality: musical ↔ noisy (harmonic content)
    """
    if verbose:
        print("  Grounding: SOUND (auditory sensation)")
    
    concepts = [
        # Loud sounds - high amplitude, sharp onset
        {"name": "loud", "amplitude": 0.9, "frequency": 0.8, "phase": 0.0,
         "rise_time": 1, "peak_time": 3, "fall_time": 2, "cooldown": 8},
        {"name": "noisy", "amplitude": 0.7, "frequency": 0.8, "phase": 0.02,
         "rise_time": 1, "peak_time": 3, "fall_time": 2, "cooldown": 8},
        {"name": "boom", "amplitude": 0.8, "frequency": 0.8, "phase": 0.04,
         "rise_time": 1, "peak_time": 2, "fall_time": 3, "cooldown": 8},
        
        # Quiet sounds - low amplitude, gentle
        {"name": "quiet", "amplitude": 0.6, "frequency": 0.5, "phase": 0.5,
         "rise_time": 4, "peak_time": 5, "fall_time": 4, "cooldown": 15},
        {"name": "whisper", "amplitude": 0.5, "frequency": 0.5, "phase": 0.52,
         "rise_time": 4, "peak_time": 5, "fall_time": 4, "cooldown": 15},
        {"name": "silent", "amplitude": 0.4, "frequency": 0.3, "phase": 0.55,
         "rise_time": 5, "peak_time": 6, "fall_time": 5, "cooldown": 20},
        
        # High pitch - fast frequency
        {"name": "high", "amplitude": 0.7, "frequency": 0.9, "phase": 0.25,
         "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 6},
        {"name": "squeak", "amplitude": 0.6, "frequency": 0.9, "phase": 0.27,
         "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 6},
        
        # Low pitch - slow frequency
        {"name": "low", "amplitude": 0.7, "frequency": 0.4, "phase": 0.75,
         "rise_time": 4, "peak_time": 5, "fall_time": 4, "cooldown": 12},
        {"name": "rumble", "amplitude": 0.6, "frequency": 0.4, "phase": 0.77,
         "rise_time": 4, "peak_time": 5, "fall_time": 4, "cooldown": 12},
        
        # Musical - harmonic
        {"name": "music", "amplitude": 0.8, "frequency": 0.6, "phase": 0.35,
         "rise_time": 3, "peak_time": 6, "fall_time": 3, "cooldown": 10},
        {"name": "sing", "amplitude": 0.7, "frequency": 0.6, "phase": 0.37,
         "rise_time": 3, "peak_time": 6, "fall_time": 3, "cooldown": 10},
    ]
    
    ground_waves(lnn, concepts, verbose=False)
    
    if verbose:
        pairs = [("loud", "noisy"), ("quiet", "whisper"), ("high", "squeak"), ("low", "rumble"), ("music", "sing"), ("loud", "quiet")]
        print(f"    Auditory pairs:")
        for a, b in pairs:
            key = lnn._key(f"word:{a}", f"word:{b}")
            if key in lnn.springs:
                print(f"      {a}-{b}: {lnn.springs[key].stiffness}")
            else:
                print(f"      {a}-{b}: (no spring)")
    
    return lnn


def ground_taste(lnn, verbose=True):
    """Ground taste as gustatory sensation.
    
    Basic tastes: sweet, sour, salty, bitter, umami
    Sweet and salty often co-occur (food).
    Bitter and sour are warning signals (poison/spoilage) → out of phase with sweet.
    """
    if verbose:
        print("  Grounding: TASTE (gustatory sensation)")
    
    concepts = [
        # Sweet - pleasant, frequent, moderate duration
        {"name": "sweet", "amplitude": 0.9, "frequency": 0.8, "phase": 0.0,
         "rise_time": 2, "peak_time": 5, "fall_time": 3, "cooldown": 10},
        {"name": "sugar", "amplitude": 0.8, "frequency": 0.8, "phase": 0.02,
         "rise_time": 2, "peak_time": 5, "fall_time": 3, "cooldown": 10},
        {"name": "honey", "amplitude": 0.7, "frequency": 0.7, "phase": 0.04,
         "rise_time": 2, "peak_time": 5, "fall_time": 3, "cooldown": 12},
        
        # Salty - often with sweet in food
        {"name": "salty", "amplitude": 0.8, "frequency": 0.7, "phase": 0.1,
         "rise_time": 2, "peak_time": 4, "fall_time": 3, "cooldown": 12},
        
        # Sour - warning signal, different phase
        {"name": "sour", "amplitude": 0.7, "frequency": 0.6, "phase": 0.5,
         "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 15},
        {"name": "lemon", "amplitude": 0.6, "frequency": 0.6, "phase": 0.52,
         "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 15},
        
        # Bitter - warning signal, opposite of sweet
        {"name": "bitter", "amplitude": 0.7, "frequency": 0.5, "phase": 0.75,
         "rise_time": 3, "peak_time": 3, "fall_time": 4, "cooldown": 18},
        
        # Umami - savory, overlaps with salty
        {"name": "savory", "amplitude": 0.7, "frequency": 0.6, "phase": 0.15,
         "rise_time": 3, "peak_time": 5, "fall_time": 3, "cooldown": 14},
    ]
    
    ground_waves(lnn, concepts, verbose=False)
    
    if verbose:
        pairs = [("sweet", "sugar"), ("sugar", "honey"), ("salty", "savory"), ("sour", "lemon"), ("sweet", "bitter"), ("sour", "bitter")]
        print(f"    Taste pairs:")
        for a, b in pairs:
            key = lnn._key(f"word:{a}", f"word:{b}")
            if key in lnn.springs:
                print(f"      {a}-{b}: {lnn.springs[key].stiffness}")
            else:
                print(f"      {a}-{b}: (no spring)")
    
    return lnn


def ground_smell(lnn, verbose=True):
    """Ground smell as olfactory sensation.
    
    Olfactory categories:
    - Pleasant: fragrant, floral, fresh
    - Unpleasant: stinky, rotten, smoky
    - Food-related: spicy, sweet-smelling
    """
    if verbose:
        print("  Grounding: SMELL (olfactory sensation)")
    
    concepts = [
        # Pleasant smells - gentle, lingering
        {"name": "fragrant", "amplitude": 0.8, "frequency": 0.6, "phase": 0.0,
         "rise_time": 4, "peak_time": 6, "fall_time": 5, "cooldown": 12},
        {"name": "fresh", "amplitude": 0.7, "frequency": 0.6, "phase": 0.02,
         "rise_time": 4, "peak_time": 6, "fall_time": 5, "cooldown": 12},
        {"name": "floral", "amplitude": 0.7, "frequency": 0.6, "phase": 0.04,
         "rise_time": 4, "peak_time": 6, "fall_time": 5, "cooldown": 12},
        
        # Unpleasant smells - sharp onset, quick decay
        {"name": "stinky", "amplitude": 0.8, "frequency": 0.5, "phase": 0.5,
         "rise_time": 2, "peak_time": 3, "fall_time": 4, "cooldown": 18},
        {"name": "rotten", "amplitude": 0.7, "frequency": 0.5, "phase": 0.52,
         "rise_time": 2, "peak_time": 3, "fall_time": 4, "cooldown": 18},
        {"name": "smoky", "amplitude": 0.6, "frequency": 0.5, "phase": 0.55,
         "rise_time": 3, "peak_time": 4, "fall_time": 5, "cooldown": 15},
        
        # Spicy - sharp, intense
        {"name": "spicy", "amplitude": 0.7, "frequency": 0.6, "phase": 0.25,
         "rise_time": 2, "peak_time": 3, "fall_time": 3, "cooldown": 14},
    ]
    
    ground_waves(lnn, concepts, verbose=False)
    
    if verbose:
        pairs = [("fragrant", "fresh"), ("fresh", "floral"), ("stinky", "rotten"), ("fragrant", "stinky")]
        print(f"    Smell pairs:")
        for a, b in pairs:
            key = lnn._key(f"word:{a}", f"word:{b}")
            if key in lnn.springs:
                print(f"      {a}-{b}: {lnn.springs[key].stiffness}")
            else:
                print(f"      {a}-{b}: (no spring)")
    
    return lnn


def ground_weight(lnn, verbose=True):
    """Ground weight as proprioceptive sensation.
    
    Weight spectrum: heavy ↔ light
    Heavy things: slow onset, slow decay (inertia)
    Light things: fast onset, fast decay
    """
    if verbose:
        print("  Grounding: WEIGHT (proprioceptive sensation)")
    
    concepts = [
        # Heavy - slow, massive, lingering
        {"name": "heavy", "amplitude": 0.9, "frequency": 0.5, "phase": 0.0,
         "rise_time": 5, "peak_time": 6, "fall_time": 5, "cooldown": 15},
        {"name": "big", "amplitude": 0.8, "frequency": 0.5, "phase": 0.02,
         "rise_time": 5, "peak_time": 6, "fall_time": 5, "cooldown": 15},
        
        # Light - fast, airy, quick
        {"name": "light", "amplitude": 0.9, "frequency": 0.7, "phase": 0.5,
         "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 8},
        {"name": "small", "amplitude": 0.7, "frequency": 0.7, "phase": 0.52,
         "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 8},
        {"name": "tiny", "amplitude": 0.6, "frequency": 0.7, "phase": 0.54,
         "rise_time": 2, "peak_time": 3, "fall_time": 2, "cooldown": 8},
    ]
    
    ground_waves(lnn, concepts, verbose=False)
    
    if verbose:
        pairs = [("heavy", "big"), ("light", "small"), ("small", "tiny"), ("heavy", "light")]
        print(f"    Weight pairs:")
        for a, b in pairs:
            key = lnn._key(f"word:{a}", f"word:{b}")
            if key in lnn.springs:
                print(f"      {a}-{b}: {lnn.springs[key].stiffness}")
            else:
                print(f"      {a}-{b}: (no spring)")
    
    return lnn


def ground_speed(lnn, verbose=True):
    """Ground speed as kinesthetic sensation.
    
    Speed spectrum: fast ↔ slow
    Fast things: rapid onset, quick decay
    Slow things: gradual onset, lingering
    """
    if verbose:
        print("  Grounding: SPEED (kinesthetic sensation)")
    
    concepts = [
        # Fast - rapid onset, quick decay
        {"name": "fast", "amplitude": 0.9, "frequency": 0.9, "phase": 0.0,
         "rise_time": 1, "peak_time": 2, "fall_time": 1, "cooldown": 5},
        {"name": "quick", "amplitude": 0.8, "frequency": 0.9, "phase": 0.02,
         "rise_time": 1, "peak_time": 2, "fall_time": 1, "cooldown": 5},
        {"name": "run", "amplitude": 0.7, "frequency": 0.8, "phase": 0.04,
         "rise_time": 1, "peak_time": 3, "fall_time": 2, "cooldown": 6},
        {"name": "fly", "amplitude": 0.7, "frequency": 0.8, "phase": 0.06,
         "rise_time": 1, "peak_time": 3, "fall_time": 2, "cooldown": 6},
        
        # Slow - gradual onset, lingering
        {"name": "slow", "amplitude": 0.9, "frequency": 0.4, "phase": 0.5,
         "rise_time": 5, "peak_time": 6, "fall_time": 5, "cooldown": 15},
        {"name": "crawl", "amplitude": 0.6, "frequency": 0.4, "phase": 0.52,
         "rise_time": 5, "peak_time": 6, "fall_time": 5, "cooldown": 15},
        {"name": "walk", "amplitude": 0.7, "frequency": 0.5, "phase": 0.54,
         "rise_time": 4, "peak_time": 5, "fall_time": 4, "cooldown": 12},
    ]
    
    ground_waves(lnn, concepts, verbose=False)
    
    if verbose:
        pairs = [("fast", "quick"), ("run", "fly"), ("slow", "crawl"), ("fast", "slow"), ("walk", "slow")]
        print(f"    Speed pairs:")
        for a, b in pairs:
            key = lnn._key(f"word:{a}", f"word:{b}")
            if key in lnn.springs:
                print(f"      {a}-{b}: {lnn.springs[key].stiffness}")
            else:
                print(f"      {a}-{b}: (no spring)")
    
    return lnn


def ground_distance(lnn, verbose=True):
    """Ground distance as spatial sensation.
    
    Distance spectrum: near ↔ far
    Near things: sharp, immediate
    Far things: diffuse, lingering
    """
    if verbose:
        print("  Grounding: DISTANCE (spatial sensation)")
    
    concepts = [
        # Near - immediate, sharp
        {"name": "near", "amplitude": 0.9, "frequency": 0.8, "phase": 0.0,
         "rise_time": 1, "peak_time": 3, "fall_time": 2, "cooldown": 8},
        {"name": "close", "amplitude": 0.8, "frequency": 0.8, "phase": 0.02,
         "rise_time": 1, "peak_time": 3, "fall_time": 2, "cooldown": 8},
        {"name": "here", "amplitude": 0.9, "frequency": 0.8, "phase": 0.01,
         "rise_time": 1, "peak_time": 3, "fall_time": 2, "cooldown": 8},
        
        # Far - diffuse, lingering
        {"name": "far", "amplitude": 0.8, "frequency": 0.5, "phase": 0.5,
         "rise_time": 4, "peak_time": 5, "fall_time": 4, "cooldown": 15},
        {"name": "away", "amplitude": 0.7, "frequency": 0.5, "phase": 0.52,
         "rise_time": 4, "peak_time": 5, "fall_time": 4, "cooldown": 15},
        {"name": "there", "amplitude": 0.8, "frequency": 0.5, "phase": 0.51,
         "rise_time": 4, "peak_time": 5, "fall_time": 4, "cooldown": 15},
    ]
    
    ground_waves(lnn, concepts, verbose=False)
    
    if verbose:
        pairs = [("near", "close"), ("close", "here"), ("far", "away"), ("away", "there"), ("near", "far")]
        print(f"    Distance pairs:")
        for a, b in pairs:
            key = lnn._key(f"word:{a}", f"word:{b}")
            if key in lnn.springs:
                print(f"      {a}-{b}: {lnn.springs[key].stiffness}")
            else:
                print(f"      {a}-{b}: (no spring)")
    
    return lnn


def ground_brightness(lnn, verbose=True):
    """Ground brightness as visual intensity.
    
    Brightness spectrum: bright ↔ dim ↔ dark
    Bright things: high amplitude, fast onset
    Dark things: low amplitude, slow onset, lingering
    """
    if verbose:
        print("  Grounding: BRIGHTNESS (visual intensity)")
    
    concepts = [
        # Bright - intense, sharp
        {"name": "bright", "amplitude": 0.95, "frequency": 0.8, "phase": 0.0,
         "rise_time": 1, "peak_time": 3, "fall_time": 2, "cooldown": 8},
        {"name": "light", "amplitude": 0.8, "frequency": 0.7, "phase": 0.02,
         "rise_time": 2, "peak_time": 4, "fall_time": 3, "cooldown": 10},
        {"name": "shine", "amplitude": 0.7, "frequency": 0.7, "phase": 0.04,
         "rise_time": 2, "peak_time": 4, "fall_time": 3, "cooldown": 10},
        {"name": "glow", "amplitude": 0.6, "frequency": 0.6, "phase": 0.06,
         "rise_time": 3, "peak_time": 5, "fall_time": 3, "cooldown": 12},
        
        # Dim - moderate, gentle
        {"name": "dim", "amplitude": 0.5, "frequency": 0.5, "phase": 0.4,
         "rise_time": 4, "peak_time": 5, "fall_time": 4, "cooldown": 15},
        
        # Dark - low intensity, lingering
        {"name": "dark", "amplitude": 0.9, "frequency": 0.4, "phase": 0.7,
         "rise_time": 5, "peak_time": 6, "fall_time": 5, "cooldown": 18},
        {"name": "shadow", "amplitude": 0.7, "frequency": 0.4, "phase": 0.72,
         "rise_time": 5, "peak_time": 6, "fall_time": 5, "cooldown": 18},
    ]
    
    ground_waves(lnn, concepts, verbose=False)
    
    if verbose:
        pairs = [("bright", "light"), ("light", "shine"), ("shine", "glow"), ("dark", "shadow"), ("bright", "dark")]
        print(f"    Brightness pairs:")
        for a, b in pairs:
            key = lnn._key(f"word:{a}", f"word:{b}")
            if key in lnn.springs:
                print(f"      {a}-{b}: {lnn.springs[key].stiffness}")
            else:
                print(f"      {a}-{b}: (no spring)")
    
    return lnn


def sensory_grounding(lnn=None, verbose=True):
    """Run full sensory grounding pipeline.
    
    Creates the sensory foundation before any text training.
    Returns the grounded LatticeNN.
    """
    if lnn is None:
        lnn = create_lnn()
    
    if verbose:
        print("=" * 60)
        print("SENSORY GROUNDING")
        print("=" * 60)
        print()
    
    ground_colors(lnn, verbose=verbose)
    ground_letters(lnn, verbose=verbose)
    ground_shapes(lnn, verbose=verbose)
    ground_temperature(lnn, verbose=verbose)
    ground_texture(lnn, verbose=verbose)
    ground_sound(lnn, verbose=verbose)
    ground_taste(lnn, verbose=verbose)
    ground_smell(lnn, verbose=verbose)
    ground_weight(lnn, verbose=verbose)
    ground_speed(lnn, verbose=verbose)
    ground_distance(lnn, verbose=verbose)
    ground_brightness(lnn, verbose=verbose)
    
    if verbose:
        print(f"\n  Grounded lattice: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
        print()
    
    return lnn
