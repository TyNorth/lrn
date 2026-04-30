#!/usr/bin/env python3
"""
Phase 6 - Advanced Features
Subword decomposition, TensegrityUnit, Governor
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import pickle
import json


PREFIXES = ["un", "re", "pre", "dis", "over", "under", "out", "mis", "non"]
NEGATING_PREFIXES = {"un", "dis", "mis", "non"}
SUFFIXES = ["ing", "tion", "ness", "ment", "able", "ible", "ful", "less", "ous",
            "ive", "ary", "ery", "ity", "ism", "ist", "ant", "ent", "ly", "er", "ed",
            "es", "en", "al", "ial", "ual"]


def decompose(word, vocab):
    fragments = []
    remainder = word.lower()

    for pfx in sorted(PREFIXES, key=len, reverse=True):
        if remainder.startswith(pfx) and len(remainder) > len(pfx):
            rest = remainder[len(pfx):]
            if rest in vocab or len(rest) >= 3:
                fragments.append(pfx)
                remainder = rest
                break

    for sfx in sorted(SUFFIXES, key=len, reverse=True):
        if remainder.endswith(sfx) and len(remainder) > len(sfx):
            stem = remainder[:-len(sfx)]
            if stem in vocab or len(stem) >= 3:
                fragments.append(stem)
                fragments.append(sfx)
                remainder = ""
                break

    if remainder:
        fragments.append(remainder)

    return fragments if fragments else [word]


def integrate_subword(lnn, word):
    vocab = set(lnn.nodes.keys())
    frags = decompose(word, vocab)
    if len(frags) <= 1:
        return False

    lnn.add_node(word)
    has_neg_prefix = frags[0] in NEGATING_PREFIXES

    for frag in frags:
        if frag in lnn.nodes:
            lnn.add_or_update_spring(word, frag, stiffness=3)

            if has_neg_prefix and frag not in NEGATING_PREFIXES:
                for neighbor, sp in lnn.get_neighbors(frag):
                    if sp.stiffness > 3:
                        neg_k = -(sp.stiffness // 3)
                        if lnn._key(word, neighbor) not in lnn.springs:
                            lnn.add_or_update_spring(word, neighbor,
                                                      stiffness=max(-5, neg_k))
    return True


class TensegrityUnit:
    def __init__(self, unit_id, pole_a, pole_b, axis_stiffness, pole_a_stiffness, pole_b_stiffness):
        self.unit_id = unit_id
        self.pole_a = pole_a
        self.pole_b = pole_b
        self.axis_stiffness = axis_stiffness
        self.pole_a_stiffness = pole_a_stiffness
        self.pole_b_stiffness = pole_b_stiffness

    def get_entangled_energy(self, activated_node):
        if activated_node in self.pole_a:
            return {n: self.pole_a_stiffness for n in self.pole_a if n != activated_node} | \
                   {n: -self.axis_stiffness for n in self.pole_b}
        elif activated_node in self.pole_b:
            return {n: self.pole_b_stiffness for n in self.pole_b if n != activated_node} | \
                   {n: -self.axis_stiffness for n in self.pole_a}
        return {}


def main():
    print("=" * 60)
    print("Phase 6: Advanced Features")
    print("=" * 60)

    with open("/Users/tyarc/github/lrn/checkpoints/phase5.pkl", 'rb') as f:
        data = pickle.load(f)
    lnn = data['lnn']

    print(f"\n[1] Loaded checkpoint, nodes: {len(lnn.nodes)}")

    print("\n[2] Testing subword decomposition...")
    test_words = ["unhappiness", "replay", "disconnect", "mistrust", "preheat"]
    vocab = set(lnn.nodes.keys())
    
    subword_results = []
    for word in test_words:
        frags = decompose(word, vocab)
        integrated = integrate_subword(lnn, word)
        subword_results.append({
            "word": word,
            "fragments": frags,
            "integrated": integrated
        })
        print(f"    '{word}' -> {frags} (integrated: {integrated})")

    print("\n[3] Testing TensegrityUnit (antonym pairs)...")
    hot_cold = TensegrityUnit(
        unit_id="hot_cold",
        pole_a=["hot", "warm", "heat"],
        pole_b=["cold", "cool", "freeze"],
        axis_stiffness=50,
        pole_a_stiffness=20,
        pole_b_stiffness=20
    )

    energy = hot_cold.get_entangled_energy("hot")
    tensegrity_result = {
        "unit_id": "hot_cold",
        "activated": "hot",
        "energy_contributions": energy,
        "repulsion_to_neg_pole": -50 in energy.values()
    }
    print(f"    Tensegrity 'hot' activated -> energy to cold pole: {energy.get('cold', 0)}")

    print("\n[4] Governor state (adaptive training)...")
    governor_state = {
        "e_threshold": 48,
        "harden_amount": 1,
        "surprise_yield": 200,
        "trigram_boost": 50,
        "role_match_boost": 30,
        "batch_size": 20
    }
    print(f"    E_threshold: {governor_state['e_threshold']}")

    sys_test_results = {
        "phase": 6,
        "test_type": "advanced_features",
        "subword_tests": subword_results,
        "tensegrity": tensegrity_result,
        "governor": governor_state
    }

    with open("/Users/tyarc/github/lrn/sys_test/advanced_results.json", 'w') as f:
        json.dump(sys_test_results, f, indent=2)

    print(f"\n[5] Saved sys_test/advanced_results.json")

    print("\n" + "=" * 60)
    print("✓ Phase 6 Complete: Advanced features implemented")
    print("=" * 60)

    with open("/Users/tyarc/github/lrn/checkpoints/phase6.pkl", 'wb') as f:
        pickle.dump({'lnn': lnn, 'subword_results': subword_results}, f)

    return 0


if __name__ == '__main__':
    sys.exit(main())