"""
Full Cumulative Pipeline: Sensory → Pre-K → K → 1st → 2nd Grade

Each level builds on the lattice from the previous level, simulating
how real children learn cumulatively across years.

Usage:
    python3 -m lrn.pipeline [--reps=N] [--rem-interval=N]
"""
import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn.sensory_grounding import sensory_grounding
from lrn.harmonic_lessons import harmonic_video_training
from lrn.physical_manipulation import physical_manipulation
from lrn.social_interaction import social_interaction
from lrn.trainer import train
from lrn.corpora import get_corpus, get_corpus_info, AVAILABLE_LEVELS
from lrn.assessor import assess_level
from lrn.charts import bar_chart, tau_distribution, confidence_label


REPORT_DIR = "/Users/tyarc/github/lrn/reports"


def run_pipeline(reps=50, rem_interval="end"):
    """Run the full cumulative pipeline."""
    results = {}
    
    # ============================================================
    # PHASE 0: Sensory Grounding (once, at the beginning)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"PHASE 0: SENSORY GROUNDING")
    print(f"{'='*60}")
    print(f"  Interoceptive + 12 external modalities")
    print()
    
    lnn = sensory_grounding(verbose=True)
    
    # ============================================================
    # PHASE 0.5: Physical & Social Simulation (once, foundational)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"PHASE 0.5: FOUNDATIONAL SIMULATIONS")
    print(f"{'='*60}")
    
    physical_manipulation(lnn, verbose=True)
    social_interaction(lnn, verbose=True)
    
    print(f"\n  After foundational simulations: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs")
    
    # ============================================================
    # PHASE 1-N: Cumulative Curriculum Levels
    # ============================================================
    levels = ["prek", "kindergarten", "first_grade", "second_grade", "third_grade"]
    
    for level in levels:
        print(f"\n{'='*60}")
        print(f"PHASE: {level.upper()}")
        print(f"{'='*60}")
        
        corpus_info = get_corpus_info(level)
        print(f"  Corpus: {corpus_info['total']} sentences ({corpus_info['original']} original + {corpus_info['varied']} varied)")
        print(f"  Reps: {reps}, REM: {rem_interval}")
        print()
        
        # Step 1: Harmonic video labeling (level-specific)
        harmonic_video_training(lnn, level, verbose=True)
        
        # Step 2: Text corpus training
        t0 = time.time()
        corpus = get_corpus(level)
        learn_type = "sensory" if level == "prek" else "language"
        train(lnn, corpus, reps=reps, learn_type=learn_type, rem_interval=rem_interval, verbose=False)
        train_time = time.time() - t0
        
        print(f"\n  Training: {train_time:.1f}s")
        print(f"  Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
        
        # Assess
        print(f"\n{'='*60}")
        print(f"ASSESSMENT: {level.upper()}")
        print(f"{'='*60}")
        
        assess_results = assess_level(lnn, level)
        
        # Group by domain
        domains = {}
        for name, r in assess_results.items():
            domain = r.get("domain", "General")
            if domain not in domains:
                domains[domain] = []
            domains[domain].append((name, r))
        
        for domain, skills in domains.items():
            print(f"\n  ── {domain} ──")
            for name, r in skills:
                skill_name = r.get("skill", name.upper())
                print(bar_chart(f"  {skill_name}", r["score"], r["possible"]))
        
        total_score = sum(r["score"] for r in assess_results.values())
        total_possible = sum(r["possible"] for r in assess_results.values())
        mastery_pct = int((total_score * 100) // total_possible)
        
        print(f"\n  {'─'*62}")
        print(bar_chart("TOTAL", total_score, total_possible))
        
        status = "MASTERY ✓" if mastery_pct >= 100 else "IN PROGRESS"
        print(f"\n  Status: {status} ({mastery_pct}%)")
        
        # Store results
        results[level] = {
            "nodes": len(lnn.nodes),
            "springs": len(lnn.springs),
            "total_score": total_score,
            "total_possible": total_possible,
            "mastery_pct": mastery_pct,
            "status": status,
            "train_time": train_time,
            "assessments": {
                name: {
                    "score": r["score"],
                    "possible": r["possible"],
                    "pct": r["pct"],
                    "domain": r.get("domain", "General"),
                    "skill": r.get("skill", name),
                }
                for name, r in assess_results.items()
            },
        }
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print(f"\n{'='*60}")
    print(f"CUMULATIVE PIPELINE SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n  {'Level':15s} {'Nodes':>8s} {'Springs':>8s} {'Score':>10s} {'Mastery':>8s}")
    print(f"  {'─'*55}")
    
    for level in levels:
        r = results[level]
        print(f"  {level:15s} {r['nodes']:8d} {r['springs']:8d} {r['total_score']:6.1f}/{r['total_possible']:3d} {r['mastery_pct']:7d}%")
    
    # Tau distribution
    tau_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for sp in lnn.springs.values():
        tau_counts[sp.tau] += 1
    print(tau_distribution(tau_counts))
    
    # Save final report
    pipeline_dir = os.path.join(REPORT_DIR, "pipeline")
    os.makedirs(pipeline_dir, exist_ok=True)
    
    report = {
        "type": "cumulative_pipeline",
        "date": datetime.now().isoformat(),
        "reps": reps,
        "rem_interval": rem_interval,
        "levels": results,
        "final_nodes": len(lnn.nodes),
        "final_springs": len(lnn.springs),
        "tau_distribution": tau_counts,
    }
    
    report_path = os.path.join(pipeline_dir, f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n  Report saved: {report_path}")
    
    return lnn, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run full cumulative pipeline")
    parser.add_argument("--reps", type=int, default=50, help="Training repetitions per level")
    parser.add_argument("--rem-interval", default="end", help="REM interval (1=every, end=final only)")
    args = parser.parse_args()
    
    run_pipeline(reps=args.reps, rem_interval=args.rem_interval)
