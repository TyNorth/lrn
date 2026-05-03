"""
Iterative Train-Test-Report-Chat-Save-Commit Pipeline

Trains each curriculum level, assesses, generates sample responses,
saves checkpoints, and reports results. Continues through all levels.

Usage:
    python3 -m lrn.iterative_pipeline [--start=prek] [--end=third_grade]
"""
import sys
import os
import json
import time
import pickle
from datetime import datetime

sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, propagate, generate, generate_sequence
from lrn.generate import _score_candidates
from lrn.sensory_grounding import sensory_grounding
from lrn.harmonic_lessons import harmonic_video_training
from lrn.physical_manipulation import physical_manipulation
from lrn.social_interaction import social_interaction
from lrn.trainer import train
from lrn.corpora import get_corpus, get_corpus_info, AVAILABLE_LEVELS
from lrn.assessor import assess_level


CHECKPOINT_DIR = "/Users/tyarc/github/lrn/checkpoints/iterative"
REPORT_DIR = "/Users/tyarc/github/lrn/reports/iterative"
CHAT_DIR = "/Users/tyarc/github/lrn/reports/chat"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CHAT_DIR, exist_ok=True)


LEVELS = ["prek", "kindergarten", "first_grade", "second_grade", "third_grade",
          "fourth_grade", "fifth_grade", "middle_school", "high_school", "college"]


def save_checkpoint(lnn, level_name):
    """Save lattice state to pickle."""
    path = os.path.join(CHECKPOINT_DIR, f"lattice_{level_name}.pkl")
    with open(path, "wb") as f:
        pickle.dump({
            "nodes": lnn.nodes,
            "springs": lnn.springs,
            "trigrams": lnn.trigrams,
        }, f)
    print(f"  Checkpoint saved: {path}")
    return path


def load_checkpoint(level_name):
    """Load lattice state from pickle."""
    path = os.path.join(CHECKPOINT_DIR, f"lattice_{level_name}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    lnn = LatticeNN()
    lnn.nodes = data["nodes"]
    lnn.springs = data["springs"]
    lnn.trigrams = data["trigrams"]
    print(f"  Checkpoint loaded: {path}")
    return lnn


def generate_sample_responses(lnn, level_name, prompts=None):
    """Generate sample responses for a trained lattice."""
    if prompts is None:
        prompts = {
            "prek": ["the cat", "big red", "i am"],
            "kindergarten": ["the cat is", "i can", "we go"],
            "first_grade": ["the story is", "because the", "she felt"],
            "second_grade": ["the character", "empathy means", "the problem"],
            "third_grade": ["the theme is", "force causes", "the government"],
        }
    level_prompts = prompts.get(level_name, ["the", "what is", "how does"])

    results = {}
    for prompt in level_prompts:
        resp = generate_sequence(lnn, prompt, max_tokens=8)
        results[prompt] = resp
        print(f"  '{prompt}' -> {resp}")

    return results


def run_level(lnn, level, reps=50):
    """Train a single curriculum level."""
    print(f"\n{'='*60}")
    print(f"LEVEL: {level.upper()}")
    print(f"{'='*60}")

    t0 = time.time()

    corpus_info = get_corpus_info(level)
    if corpus_info:
        print(f"  Corpus: {corpus_info['total']} sentences")
        harmonic_video_training(lnn, level, verbose=True)
        corpus = get_corpus(level)
        learn_type = "sensory" if level == "prek" else "language"
        train(lnn, corpus, reps=reps, learn_type=learn_type, rem_interval="end", verbose=True)
    else:
        print(f"  No corpus for {level}, skipping text training")

    elapsed = time.time() - t0
    print(f"\n  Training time: {elapsed:.1f}s")
    print(f"  Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")

    return elapsed


def assess_and_report(lnn, level, train_time):
    """Assess level and save report."""
    print(f"\n{'='*60}")
    print(f"ASSESSMENT: {level.upper()}")
    print(f"{'='*60}")

    try:
        assess_results = assess_level(lnn, level)
    except Exception as e:
        print(f"  No assessor for {level}: {e}")
        assess_results = {}

    report = {
        "level": level,
        "date": datetime.now().isoformat(),
        "nodes": len(lnn.nodes),
        "springs": len(lnn.springs),
        "train_time": round(train_time, 2),
        "assessments": {},
    }

    total_score = 0
    total_possible = 0

    for name, r in assess_results.items():
        score = r.get("score", 0)
        possible = r.get("possible", 0)
        pct = int((score * 100) // possible) if possible > 0 else 0
        total_score += score
        total_possible += possible

        report["assessments"][name] = {
            "score": score,
            "possible": possible,
            "pct": pct,
            "domain": r.get("domain", "General"),
            "skill": r.get("skill", name),
        }

        bar = "#" * (pct // 5)
        print(f"  {r.get('skill', name):30s} {score:3d}/{possible:3d} {pct:3d}% {bar}")

    mastery_pct = int((total_score * 100) // total_possible) if total_possible > 0 else 0
    report["total_score"] = total_score
    report["total_possible"] = total_possible
    report["mastery_pct"] = mastery_pct
    report["status"] = "MASTERY" if mastery_pct >= 100 else "IN PROGRESS"

    print(f"\n  TOTAL: {total_score}/{total_possible} = {mastery_pct}%")
    print(f"  Status: {report['status']}")

    report_path = os.path.join(REPORT_DIR, f"report_{level}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {report_path}")

    return report


def run_chat_demo(lnn, level, max_tokens=10):
    """Run a quick chat demo and save results."""
    print(f"\n{'='*60}")
    print(f"CHAT DEMO: {level.upper()}")
    print(f"{'='*60}")

    demo_prompts = {
        "prek": ["the cat", "big dog", "red ball"],
        "kindergarten": ["the cat is", "i can read", "we play"],
        "first_grade": ["the story is about", "because she", "the problem was"],
        "second_grade": ["the character felt", "empathy means", "the solution was"],
        "third_grade": ["the theme of the story", "force causes objects", "the government makes"],
    }

    prompts = demo_prompts.get(level, ["the", "what is", "how does"])
    results = {}

    for prompt in prompts:
        t0 = time.time()
        resp = generate_sequence(lnn, prompt, max_tokens=max_tokens)
        elapsed = time.time() - t0
        results[prompt] = {"response": resp, "time": round(elapsed, 2)}
        print(f"  '{prompt}' -> {resp} ({elapsed:.1f}s)")

    chat_path = os.path.join(CHAT_DIR, f"demo_{level}.json")
    with open(chat_path, "w") as f:
        json.dump({"level": level, "prompts": results}, f, indent=2)

    return results


def run_pipeline(start="prek", end="college", reps=50):
    """Run the full iterative pipeline."""
    start_idx = LEVELS.index(start) if start in LEVELS else 0
    end_idx = LEVELS.index(end) if end in LEVELS else len(LEVELS) - 1

    all_results = []

    for level in LEVELS[start_idx:end_idx + 1]:
        print(f"\n{'#'*60}")
        print(f"# ITERATION: {level.upper()}")
        print(f"{'#'*60}")

        lnn = load_checkpoint(level)
        if lnn is None:
            if level == "prek" or level == LEVELS[start_idx]:
                print(f"\n  Starting fresh lattice...")
                lnn = sensory_grounding(verbose=False)
                physical_manipulation(lnn, verbose=False)
                social_interaction(lnn, verbose=False)
            else:
                prev_level = LEVELS[LEVELS.index(level) - 1]
                lnn = load_checkpoint(prev_level)
                if lnn is None:
                    print(f"  ERROR: No checkpoint for {prev_level}, cannot continue")
                    break
                print(f"\n  Loaded from {prev_level} checkpoint")

        train_time = run_level(lnn, level, reps=reps)
        report = assess_and_report(lnn, level, train_time)
        chat_results = run_chat_demo(lnn, level)
        checkpoint_path = save_checkpoint(lnn, level)

        all_results.append({
            "level": level,
            "report": report,
            "chat": chat_results,
            "checkpoint": checkpoint_path,
        })

    print(f"\n{'='*60}")
    print(f"PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Level':15s} {'Nodes':>8s} {'Springs':>8s} {'Mastery':>8s} {'Status':>12s}")
    print(f"  {'─'*55}")

    for r in all_results:
        rep = r["report"]
        print(f"  {r['level']:15s} {rep['nodes']:8d} {rep['springs']:8d} {rep['mastery_pct']:7d}% {rep['status']:>12s}")

    summary_path = os.path.join(REPORT_DIR, f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, "w") as f:
        json.dump({
            "date": datetime.now().isoformat(),
            "levels": all_results,
        }, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")

    return lnn, all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Iterative training pipeline")
    parser.add_argument("--start", default="prek", help="Starting level")
    parser.add_argument("--end", default="third_grade", help="Ending level")
    parser.add_argument("--reps", type=int, default=50, help="Training repetitions")
    args = parser.parse_args()

    lnn, results = run_pipeline(start=args.start, end=args.end, reps=args.reps)
