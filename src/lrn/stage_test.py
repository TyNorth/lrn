"""
Step-by-step training pipeline with reports at each stage.

1. Fresh lattice → interoceptive grounding → report
2. Continue → external sensory grounding → report
3. Continue → Pre-K text training → report
"""
import sys
import os
import json
from datetime import datetime
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn.trainer import create_lnn, train, optimal_rem, prune_springs
from lrn.assessor import assess_level
from lrn.sensory_grounding import (
    ground_interoceptive,
    ground_colors, ground_letters, ground_shapes,
    ground_temperature, ground_texture, ground_sound,
    ground_taste, ground_smell, ground_weight,
    ground_speed, ground_distance, ground_brightness,
)
from lrn.corpora import get_corpus, get_corpus_info
from lrn.charts import bar_chart, tau_distribution, category_density

REPORT_DIR = "/Users/tyarc/github/lrn/reports/stage_test"
os.makedirs(REPORT_DIR, exist_ok=True)


def save_stage_report(lnn, stage_name, assessments=None):
    """Save a report for the current stage."""
    tau_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for sp in lnn.springs.values():
        tau_counts[sp.tau] += 1
    
    report = {
        "stage": stage_name,
        "date": datetime.now().isoformat(),
        "nodes": len(lnn.nodes),
        "springs": len(lnn.springs),
        "tau_distribution": tau_counts,
    }
    
    if assessments:
        total_score = sum(r["score"] for r in assessments.values())
        total_possible = sum(r["possible"] for r in assessments.values())
        report["total_score"] = total_score
        report["total_possible"] = total_possible
        report["mastery_pct"] = (total_score * 100) // total_possible
        report["assessments"] = {
            name: {"score": r["score"], "possible": r["possible"], "pct": r["pct"]}
            for name, r in assessments.items()
        }
    
    report_path = os.path.join(REPORT_DIR, f"{stage_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    return report_path


def assess_sensory(lnn):
    """Quick assessment of sensory grounding quality."""
    results = {}
    
    # Colors
    colors = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "brown", "pink"]
    warm = {"red", "orange", "yellow"}
    cool = {"blue", "green", "purple"}
    warm_internal = cool_internal = cross = 0
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
    results["color_clustering"] = {"score": warm_internal + cool_internal, "possible": total, "pct": (warm_internal + cool_internal) * 100 // max(1, total)}
    
    # Letters - visually similar pairs
    expected_similar = [("b", "d"), ("p", "q"), ("g", "q"), ("c", "s"), ("w", "x"), ("f", "h"), ("g", "p"), ("a", "e")]
    letter_score = 0
    for a, b in expected_similar:
        key = lnn._key(f"word:{a}", f"word:{b}")
        if key in lnn.springs and lnn.springs[key].stiffness > 50:
            letter_score += 1
    results["letter_visual"] = {"score": letter_score, "possible": len(expected_similar), "pct": letter_score * 100 // len(expected_similar)}
    
    # Opposites should have LOW springs
    opposites = [("hot", "cold"), ("soft", "hard"), ("sweet", "bitter"), ("fast", "slow"), ("loud", "quiet"), ("bright", "dark")]
    opp_score = 0
    for a, b in opposites:
        key = lnn._key(f"word:{a}", f"word:{b}")
        if key not in lnn.springs or lnn.springs[key].stiffness < 10:
            opp_score += 1
    results["opposite_separation"] = {"score": opp_score, "possible": len(opposites), "pct": opp_score * 100 // len(opposites)}
    
    return results


# ============================================================
# STAGE 1: Fresh lattice → Interoceptive grounding
# ============================================================
print("=" * 60)
print("STAGE 1: INTEROCEPTIVE GROUNDING")
print("=" * 60)
print()

lnn = create_lnn()
print(f"  Fresh lattice: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
print()

ground_interoceptive(lnn, verbose=True)

sensory_results = assess_sensory(lnn)
print(f"\n  After interoceptive: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
for name, r in sensory_results.items():
    print(f"    {name}: {r['score']}/{r['possible']} ({r['pct']}%)")

path = save_stage_report(lnn, "interoceptive", sensory_results)
print(f"\n  Report: {path}")

# ============================================================
# STAGE 2: Continue → External sensory grounding
# ============================================================
print("\n" + "=" * 60)
print("STAGE 2: EXTERNAL SENSORY GROUNDING")
print("=" * 60)
print()

ground_colors(lnn, verbose=True)
ground_letters(lnn, verbose=True)
ground_shapes(lnn, verbose=True)
ground_temperature(lnn, verbose=True)
ground_texture(lnn, verbose=True)
ground_sound(lnn, verbose=True)
ground_taste(lnn, verbose=True)
ground_smell(lnn, verbose=True)
ground_weight(lnn, verbose=True)
ground_speed(lnn, verbose=True)
ground_distance(lnn, verbose=True)
ground_brightness(lnn, verbose=True)

sensory_results = assess_sensory(lnn)
print(f"\n  After external sensory: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
for name, r in sensory_results.items():
    print(f"    {name}: {r['score']}/{r['possible']} ({r['pct']}%)")

path = save_stage_report(lnn, "external_sensory", sensory_results)
print(f"\n  Report: {path}")

# ============================================================
# STAGE 3: Continue → Pre-K text training
# ============================================================
print("\n" + "=" * 60)
print("STAGE 3: PRE-K TEXT TRAINING")
print("=" * 60)

corpus_info = get_corpus_info("prek")
print(f"  Corpus: {corpus_info['total']} sentences")
print()

import time
t0 = time.time()
corpus = get_corpus("prek")
train(lnn, corpus, reps=3, learn_type="sensory", rem_interval="end")
train_time = time.time() - t0

print(f"\n  Training: {train_time:.1f}s")
print(f"  Final lattice: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")

# Full Pre-K assessment
assessments = assess_level(lnn, "prek")
for name, r in assessments.items():
    print(bar_chart(name.upper(), r["score"], r["possible"]))

total_score = sum(r["score"] for r in assessments.values())
total_possible = sum(r["possible"] for r in assessments.values())
mastery_pct = (total_score * 100) // total_possible
print(f"  {'─'*62}")
print(bar_chart("TOTAL", total_score, total_possible))
status = "MASTERY ✓" if mastery_pct >= 100 else "IN PROGRESS"
print(f"\n  Status: {status} ({mastery_pct}%)")

# Tau distribution
tau_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
for sp in lnn.springs.values():
    tau_counts[sp.tau] += 1
print(tau_distribution(tau_counts))

# Category density
categories_data = []
for cat_name, members in [
    ("animals", ["cat", "dog", "bird", "fish", "horse", "cow", "sheep", "pig", "chicken", "duck"]),
    ("colors", ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "brown", "pink"]),
    ("emotions", ["happy", "sad", "angry", "scared", "surprised", "tired"])
]:
    tau3 = 0
    total = 0
    for i in range(len(members)):
        for j in range(i+1, len(members)):
            key = lnn._key(f"word:{members[i]}", f"word:{members[j]}")
            total += 1
            if key in lnn.springs and lnn.springs[key].tau == 3:
                tau3 += 1
    categories_data.append((cat_name, tau3, total))
print(category_density(categories_data))

path = save_stage_report(lnn, "prek_text", assessments)
print(f"\n  Report: {path}")
