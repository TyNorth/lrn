"""
Scientific Test: Quality vs Quantity for Color Learning

Tests two hypotheses:
1. Sensory-rich corpus (quality) outperforms label-only corpus (quantity)
2. Colors trained as "sensory" learn_type perform better than "language"
"""
import sys
import os
import json
from datetime import datetime
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn.trainer import create_lnn, train
from lrn.assessor import assess_level
from lrn.charts import bar_chart, tau_distribution, category_density
from lrn.corpora.color_quality import QUALITY_COLOR_CORPUS
from lrn.corpora.color_quantity import QUANTITY_COLOR_CORPUS

RESULTS_DIR = "/Users/tyarc/github/lrn/reports/color_test"
os.makedirs(RESULTS_DIR, exist_ok=True)


def test_colors(corpus, name, learn_type, reps=3, rem_interval="end"):
    """Train and assess a color corpus."""
    print(f"\n{'='*60}")
    print(f"TEST: {name} ({learn_type})")
    print(f"{'='*60}")
    print(f"  Sentences: {len(corpus)}")
    print(f"  Learn type: {learn_type}")
    print(f"  Reps: {reps}, REM: {rem_interval}")
    print()
    
    lnn = create_lnn()
    
    import time
    t0 = time.time()
    train(lnn, corpus, reps=reps, learn_type=learn_type, rem_interval=rem_interval)
    train_time = time.time() - t0
    
    # Color clustering assessment
    colors = ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "brown", "pink"]
    tau3 = 0
    total = 0
    for i in range(len(colors)):
        for j in range(i+1, len(colors)):
            key = lnn._key(f"word:{colors[i]}", f"word:{colors[j]}")
            total += 1
            if key in lnn.springs and lnn.springs[key].tau == 3:
                tau3 += 1
    
    # Individual color node existence
    color_nodes = sum(1 for c in colors if f"word:{c}" in lnn.nodes)
    
    # Probe each color - count connections to other colors
    color_connections = {}
    for c in colors:
        node = f"word:{c}"
        if node in lnn.nodes:
            neighbors = lnn.get_neighbors(node)
            other_colors = [n.replace("word:", "") for n, sp in neighbors if n.startswith("word:") and n.replace("word:", "") in colors]
            color_connections[c] = other_colors
    
    tau_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for sp in lnn.springs.values():
        tau_counts[sp.tau] += 1
    
    print(f"\n  Training: {train_time:.1f}s")
    print(f"  Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
    print(f"  Color nodes: {color_nodes}/10")
    print(f"  Color clustering: {tau3}/{total} ({tau3*100//max(1,total)}%)")
    print(tau_distribution(tau_counts))
    
    # Show which colors connect to which
    print(f"\n  Color Connections:")
    for c in colors:
        if c in color_connections:
            others = color_connections[c]
            print(f"    {c}: {', '.join(others) if others else '(none)'}")
    
    return {
        "name": name,
        "learn_type": learn_type,
        "sentences": len(corpus),
        "reps": reps,
        "rem_interval": rem_interval,
        "train_time": train_time,
        "nodes": len(lnn.nodes),
        "springs": len(lnn.springs),
        "color_nodes": color_nodes,
        "color_clustering_tau3": tau3,
        "color_clustering_total": total,
        "color_clustering_pct": tau3 * 100 // max(1, total),
        "tau_distribution": tau_counts,
        "color_connections": color_connections,
    }


# Run all 4 combinations
results = []

# Quality corpus, sensory
results.append(test_colors(QUALITY_COLOR_CORPUS, "quality", "sensory"))

# Quality corpus, language
results.append(test_colors(QUALITY_COLOR_CORPUS, "quality", "language"))

# Quantity corpus, sensory
results.append(test_colors(QUANTITY_COLOR_CORPUS, "quantity", "sensory"))

# Quantity corpus, language
results.append(test_colors(QUANTITY_COLOR_CORPUS, "quantity", "language"))

# Summary comparison
print(f"\n{'='*60}")
print(f"SUMMARY: Quality vs Quantity")
print(f"{'='*60}")

print(f"\n  {'Metric':25s} {'Q+Sensory':>12s} {'Q+Lang':>12s} {'Qty+Sensory':>12s} {'Qty+Lang':>12s}")
print(f"  {'─'*75}")
print(f"  {'Sentences':25s} {results[0]['sentences']:>12d} {results[1]['sentences']:>12d} {results[2]['sentences']:>12d} {results[3]['sentences']:>12d}")
print(f"  {'Train Time (s)':25s} {results[0]['train_time']:>11.1f}s {results[1]['train_time']:>11.1f}s {results[2]['train_time']:>11.1f}s {results[3]['train_time']:>11.1f}s")
print(f"  {'Springs':25s} {results[0]['springs']:>12d} {results[1]['springs']:>12d} {results[2]['springs']:>12d} {results[3]['springs']:>12d}")
print(f"  {'Color Nodes':25s} {results[0]['color_nodes']:>12d} {results[1]['color_nodes']:>12d} {results[2]['color_nodes']:>12d} {results[3]['color_nodes']:>12d}")
print(f"  {'Color Clustering τ=3':25s} {results[0]['color_clustering_pct']:>11d}% {results[1]['color_clustering_pct']:>11d}% {results[2]['color_clustering_pct']:>11d}% {results[3]['color_clustering_pct']:>11d}%")

# Save results
report = {
    "date": datetime.now().isoformat(),
    "test": "color_quality_vs_quantity",
    "results": results,
}

report_path = os.path.join(RESULTS_DIR, f"color_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\n  Report saved: {report_path}")
